"""
Whisper Diarization package for speaker diarization using Whisper and NeMo.
"""

from .diarization_utils import DiarizationPipeline
from .helpers import (
    get_words_speaker_mapping,
    get_speaker_aware_transcript,
    get_sentences_speaker_mapping,
)

__version__ = "0.1.0"

__all__ = [
    "DiarizationPipeline",
    "get_words_speaker_mapping",
    "get_speaker_aware_transcript",
    "get_sentences_speaker_mapping",
]
