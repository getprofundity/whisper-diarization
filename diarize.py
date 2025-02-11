import argparse
import logging
import os
import re
import json
import numpy as np
from pathlib import Path
import tempfile
import shutil

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
from diarization_utils import DiarizationPipeline

mtypes = {"cpu": "int8", "cuda": "float16"}

# Set up cache directory for models
CACHE_DIR = os.path.join(os.getcwd(), "checkpoints")
WHISPER_CACHE_DIR = os.path.join(CACHE_DIR, "whisper")
ALIGNMENT_CACHE_DIR = os.path.join(CACHE_DIR, "alignment")
PUNCT_CACHE_DIR = os.path.join(CACHE_DIR, "punctuation")
NEMO_CACHE_DIR = os.path.join(CACHE_DIR, "nemo")

# Create temporary directory for processing
TEMP_DIR = tempfile.mkdtemp()

# Create cache directories if they don't exist
os.makedirs(WHISPER_CACHE_DIR, exist_ok=True)
os.makedirs(ALIGNMENT_CACHE_DIR, exist_ok=True)
os.makedirs(PUNCT_CACHE_DIR, exist_ok=True)
os.makedirs(NEMO_CACHE_DIR, exist_ok=True)

# Set NeMo cache directory
os.environ["NEMO_CACHE_DIR"] = NEMO_CACHE_DIR


# Ensure cleanup of temporary directory
def cleanup_temp():
    """Clean up temporary directory and its contents."""
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)


# Register cleanup on normal program exit
import atexit

atexit.register(cleanup_temp)


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


def main():
    args = parser.parse_args()
    args.language = process_language_arg(args.language, args.model_name)

    # Initialize and run the pipeline
    pipeline = DiarizationPipeline()
    try:
        pipeline.process_audio(args.audio, args)
    finally:
        pipeline.cleanup()


if __name__ == "__main__":
    main()
