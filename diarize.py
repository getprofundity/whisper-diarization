import argparse
import logging
import os
import re
import json
import numpy as np
from pathlib import Path

import faster_whisper
import torch
import torchaudio

from ctc_forced_aligner import (
    generate_emissions,
    get_alignments,
    get_spans,
    load_alignment_model,
    postprocess_results,
    preprocess_text,
)
from deepmultilingualpunctuation import PunctuationModel
from nemo.collections.asr.models.msdd_models import NeuralDiarizer

from helpers import (
    cleanup,
    create_config,
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
)

mtypes = {"cpu": "int8", "cuda": "float16"}

# Set up cache directory for models
CACHE_DIR = os.path.join(os.getcwd(), "checkpoints")
WHISPER_CACHE_DIR = os.path.join(CACHE_DIR, "whisper")
ALIGNMENT_CACHE_DIR = os.path.join(CACHE_DIR, "alignment")
PUNCT_CACHE_DIR = os.path.join(CACHE_DIR, "punctuation")
NEMO_CACHE_DIR = os.path.join(CACHE_DIR, "nemo")

# Create cache directories if they don't exist
os.makedirs(WHISPER_CACHE_DIR, exist_ok=True)
os.makedirs(ALIGNMENT_CACHE_DIR, exist_ok=True)
os.makedirs(PUNCT_CACHE_DIR, exist_ok=True)
os.makedirs(NEMO_CACHE_DIR, exist_ok=True)

# Set NeMo cache directory
os.environ["NEMO_CACHE_DIR"] = NEMO_CACHE_DIR


def load_cached_alignment_model(device, dtype):
    """Load the alignment model with caching support."""
    # Set HF cache directory for the alignment model
    os.environ["TRANSFORMERS_CACHE"] = ALIGNMENT_CACHE_DIR
    os.environ["HF_HOME"] = ALIGNMENT_CACHE_DIR
    # Use the default model path from the library
    return load_alignment_model(
        device=device,
        model_path="MahmoudAshraf/mms-300m-1130-forced-aligner",
        dtype=dtype,
    )


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
    default="medium.en",
    help="name of the Whisper model to use",
)

parser.add_argument(
    "--batch-size",
    type=int,
    dest="batch_size",
    default=8,
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

args = parser.parse_args()
language = process_language_arg(args.language, args.model_name)

if args.stemming:
    # Isolate vocals from the rest of the audio

    return_code = os.system(
        f'python -m demucs.separate -n htdemucs --two-stems=vocals "{args.audio}" -o temp_outputs --device "{args.device}"'
    )

    if return_code != 0:
        logging.warning(
            "Source splitting failed, using original audio file. "
            "Use --no-stem argument to disable it."
        )
        vocal_target = args.audio
    else:
        vocal_target = os.path.join(
            "temp_outputs",
            "htdemucs",
            os.path.splitext(os.path.basename(args.audio))[0],
            "vocals.wav",
        )
else:
    vocal_target = args.audio


# Transcribe the audio file

whisper_model = faster_whisper.WhisperModel(
    args.model_name,
    device=args.device,
    compute_type=mtypes[args.device],
    download_root=WHISPER_CACHE_DIR,
)
whisper_pipeline = faster_whisper.BatchedInferencePipeline(whisper_model)
audio_waveform = faster_whisper.decode_audio(vocal_target)
suppress_tokens = (
    find_numeral_symbol_tokens(whisper_model.hf_tokenizer)
    if args.suppress_numerals
    else [-1]
)

if args.batch_size > 0:
    transcript_segments, info = whisper_pipeline.transcribe(
        audio_waveform,
        language,
        suppress_tokens=suppress_tokens,
        batch_size=args.batch_size,
    )
else:
    transcript_segments, info = whisper_model.transcribe(
        audio_waveform,
        language,
        suppress_tokens=suppress_tokens,
        vad_filter=True,
    )

full_transcript = "".join(segment.text for segment in transcript_segments)

# clear gpu vram
del whisper_model, whisper_pipeline
torch.cuda.empty_cache()

# Forced Alignment
alignment_model, alignment_tokenizer = load_cached_alignment_model(
    args.device,
    torch.float16 if args.device == "cuda" else torch.float32,
)

emissions, stride = generate_emissions(
    alignment_model,
    torch.from_numpy(audio_waveform)
    .to(alignment_model.dtype)
    .to(alignment_model.device),
    batch_size=args.batch_size,
)

del alignment_model
torch.cuda.empty_cache()

tokens_starred, text_starred = preprocess_text(
    full_transcript,
    romanize=True,
    language=langs_to_iso[info.language],
)

segments, scores, blank_token = get_alignments(
    emissions,
    tokens_starred,
    alignment_tokenizer,
)

spans = get_spans(tokens_starred, segments, blank_token)

word_timestamps = postprocess_results(text_starred, spans, stride, scores)


# convert audio to mono for NeMo combatibility
ROOT = os.getcwd()
temp_path = os.path.join(ROOT, "temp_outputs")
os.makedirs(temp_path, exist_ok=True)
torchaudio.save(
    os.path.join(temp_path, "mono_file.wav"),
    torch.from_numpy(audio_waveform).unsqueeze(0).float(),
    16000,
    channels_first=True,
)


# Initialize NeMo MSDD diarization model
msdd_model = NeuralDiarizer(cfg=create_config(temp_path)).to(args.device)
msdd_model.diarize()

del msdd_model
torch.cuda.empty_cache()

# Reading timestamps <> Speaker Labels mapping


speaker_ts = []
with open(os.path.join(temp_path, "pred_rttms", "mono_file.rttm"), "r") as f:
    lines = f.readlines()
    for line in lines:
        line_list = line.split(" ")
        s = int(float(line_list[5]) * 1000)
        e = s + int(float(line_list[8]) * 1000)
        # Extract speaker confidence from RTTM (field 10), handle '<NA>' case
        speaker_conf = float(line_list[10]) if line_list[10] != "<NA>" else 1.0
        speaker_id = int(line_list[11].split("_")[-1])
        speaker_ts.append([s, e, speaker_id, speaker_conf])

wsm = get_words_speaker_mapping(word_timestamps, speaker_ts, "start")

if info.language in punct_model_langs:
    # restoring punctuation in the transcript to help realign the sentences
    punct_model = PunctuationModel(
        model="kredor/punctuate-all", cache_dir=PUNCT_CACHE_DIR
    )

    words_list = list(map(lambda x: x["word"], wsm))

    labled_words = punct_model.predict(words_list, chunk_size=230)

    ending_puncts = ".?!"
    model_puncts = ".,;:!?"

    # We don't want to punctuate U.S.A. with a period. Right?
    is_acronym = lambda x: re.fullmatch(r"\b(?:[a-zA-Z]\.){2,}", x)

    for word_dict, labeled_tuple in zip(wsm, labled_words):
        word = word_dict["word"]
        if (
            word
            and labeled_tuple[1] in ending_puncts
            and (word[-1] not in model_puncts or is_acronym(word))
        ):
            word += labeled_tuple[1]
            if word.endswith(".."):
                word = word.rstrip(".")
            word_dict["word"] = word

else:
    logging.warning(
        f"Punctuation restoration is not available for {info.language} language."
        " Using the original punctuation."
    )

wsm = get_realigned_ws_mapping_with_punctuation(wsm)
ssm = get_sentences_speaker_mapping(wsm, speaker_ts)


# Create word-level output with confidence scores
word_level_output = []
for word_info, score in zip(wsm, scores):
    # Convert log probability to regular probability using exp
    confidence = float(np.exp(score)) if score is not None else 1.0

    # Find the corresponding speaker segment for this word
    word_time = word_info["start_time"]
    speaker_confidence = 1.0  # default value
    for s, e, spk, conf in speaker_ts:
        if s <= word_time <= e and spk == word_info["speaker"]:
            speaker_confidence = conf
            break

    word_entry = {
        "speaker": word_info["speaker"],  # Already a number
        "start": word_info["start_time"] / 1000.0,  # Convert to seconds
        "end": word_info["end_time"] / 1000.0,  # Convert to seconds
        "word": word_info["word"].strip(),
        "confidence": confidence,
        "speaker_confidence": speaker_confidence,
    }
    word_level_output.append(word_entry)

# Create segment-level output with words grouped under each segment
segment_output = []
current_word_idx = 0
for segment in ssm:
    # Extract speaker number from the "Speaker X" format
    speaker_id = int(segment["speaker"].split()[-1])
    segment_entry = {
        "speaker": speaker_id,  # Use the number instead of string
        "start": segment["start_time"] / 1000.0,  # Convert to seconds
        "end": segment["end_time"] / 1000.0,  # Convert to seconds
        "text": segment["text"].strip(),
        "words": [],
    }

    # Add all words that fall within this segment's time range
    while (
        current_word_idx < len(word_level_output)
        and word_level_output[current_word_idx]["start"] <= segment["end_time"] / 1000.0
    ):
        if (
            word_level_output[current_word_idx]["start"]
            >= segment["start_time"] / 1000.0
        ):
            segment_entry["words"].append(word_level_output[current_word_idx])
        current_word_idx += 1

    segment_output.append(segment_entry)

with open(f"{os.path.splitext(args.audio)[0]}.txt", "w", encoding="utf-8-sig") as f:
    get_speaker_aware_transcript(ssm, f)

with open(f"{os.path.splitext(args.audio)[0]}.srt", "w", encoding="utf-8-sig") as srt:
    write_srt(ssm, srt)

with open(
    f"{os.path.splitext(args.audio)[0]}_segments.json", "w", encoding="utf-8-sig"
) as f:
    json.dump(segment_output, f, indent=2, ensure_ascii=False, default=str)


cleanup(temp_path)
