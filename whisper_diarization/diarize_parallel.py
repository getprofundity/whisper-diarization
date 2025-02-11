import argparse
import logging
import os
import re
import subprocess
import sys
import json
import numpy as np

import faster_whisper
import torch

from ctc_forced_aligner import (
    generate_emissions,
    get_alignments,
    get_spans,
    load_alignment_model,
    postprocess_results,
    preprocess_text,
)
from deepmultilingualpunctuation import PunctuationModel

from .helpers import (
    cleanup,
    find_numeral_symbol_tokens,
    get_realigned_ws_mapping_with_punctuation,
    get_sentences_speaker_mapping,
    get_speaker_aware_transcript,
    get_words_speaker_mapping,
    langs_to_iso,
    process_language_arg,
    punct_model_langs,
    whisper_langs,
    write_srt,
    write_txt,
)
from .diarization_utils import DiarizationPipeline

mtypes = {"cpu": "int8", "cuda": "float16"}

# Initialize parser
parser = argparse.ArgumentParser()
parser.add_argument(
    "-a", "--audio", help="name of the target audio file", required=True
)
parser.add_argument(
    "--no-stem",
    action="store_false",
    dest="stemming",
    default=True,
    help="Disables source separation."
    "This helps with long files that don't contain a lot of music.",
)

parser.add_argument(
    "--suppress_numerals",
    action="store_true",
    dest="suppress_numerals",
    default=False,
    help="Suppresses Numerical Digits."
    "This helps the diarization accuracy but converts all digits into written text.",
)

parser.add_argument(
    "--whisper-model",
    dest="model_name",
    default="large-v2",
    help="name of the Whisper model to use",
)

parser.add_argument(
    "--batch-size",
    type=int,
    dest="batch_size",
    default=4,
    help="Batch size for batched inference, reduce if you run out of memory, "
    "set to 0 for original whisper longform inference",
)

parser.add_argument(
    "--language",
    type=str,
    default=None,
    choices=whisper_langs,
    help="Language spoken in the audio, specify None to perform language detection",
)

parser.add_argument(
    "--device",
    dest="device",
    default="cuda" if torch.cuda.is_available() else "cpu",
    help="if you have a GPU use 'cuda', otherwise 'cpu'",
)


def main():
    args = parser.parse_args()
    args.language = process_language_arg(args.language, args.model_name)

    # Initialize the pipeline
    pipeline = DiarizationPipeline()

    try:
        # Process audio stemming if needed
        vocal_target = pipeline._process_audio_stemming(args.audio, args)

        # Start NeMo process in parallel
        logging.info(f"Starting Nemo process with vocal_target: {vocal_target}")
        nemo_process = subprocess.Popen(
            [
                sys.executable,  # Use the current Python interpreter
                "-m",
                "whisper_diarization.nemo_process",
                "-a",
                vocal_target,
                "--device",
                args.device,
                "--temp-dir",
                pipeline.TEMP_DIR,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=os.environ.copy(),  # Pass current environment
        )

        # Get transcription and word timestamps
        audio_waveform, word_timestamps, scores, info = pipeline._transcribe_audio(
            vocal_target, args
        )

        # Wait for NeMo process to complete
        nemo_return_code = nemo_process.wait()
        nemo_error_trace = nemo_process.stderr.read()
        assert nemo_return_code == 0, (
            "Diarization failed with the following error:"
            f"\n{nemo_error_trace.decode('utf-8')}"
        )

        # Read speaker timestamps
        speaker_ts = []
        with open(
            os.path.join(pipeline.TEMP_DIR, "pred_rttms", "mono_file.rttm"), "r"
        ) as f:
            lines = f.readlines()
            for line in lines:
                line_list = line.split(" ")
                s = int(float(line_list[5]) * 1000)
                e = s + int(float(line_list[8]) * 1000)
                speaker_conf = float(line_list[10]) if line_list[10] != "<NA>" else 1.0
                speaker_id = int(line_list[11].split("_")[-1])
                speaker_ts.append([s, e, speaker_id, speaker_conf])

        # Create word-speaker mapping
        wsm = get_words_speaker_mapping(word_timestamps, speaker_ts, "start")

        # Add punctuation if available
        wsm = pipeline._add_punctuation(wsm, info, args)

        # Create final output
        pipeline._create_output(wsm, speaker_ts, scores, args.audio)

    finally:
        pipeline.cleanup()


if __name__ == "__main__":
    main()
