"""
Whisper Diarization package for speaker diarization using Whisper and NeMo.
"""

from .diarization_utils import DiarizationPipeline, ParallelNemo
from .helpers import (
    get_words_speaker_mapping,
    get_speaker_aware_result,
    get_sentences_speaker_mapping,
)

__version__ = "0.1.1"

__all__ = [
    "DiarizationPipeline",
    "ParallelNemo",
    "get_words_speaker_mapping",
    "get_speaker_aware_result",
    "get_sentences_speaker_mapping",
]
