import os
import re
import json
import logging
import tempfile
import shutil
import numpy as np
import torch
import faster_whisper
from typing import Dict, List, Tuple, Any, Optional
import subprocess
import sys

from nemo.collections.asr.models.msdd_models import NeuralDiarizer, EncDecDiarLabelModel
from nemo.collections.asr.models.classification_models import EncDecClassificationModel
from nemo.collections.asr.models.label_models import EncDecSpeakerLabelModel

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
    find_numeral_symbol_tokens,
    get_realigned_ws_mapping_with_punctuation,
    get_sentences_speaker_mapping,
    get_speaker_aware_transcript,
    get_words_speaker_mapping,
    langs_to_iso,
    write_srt,
    punct_model_langs,
    load_prototype_config,
    create_config,
    get_speaker_timestamps
)

# Model type mapping
mtypes = {"cpu": "int8", "cuda": "float16"}

class ParallelNemo:
    def __init__(self, audio_file_path: str, device: str, temp_dir_path: str, model_cache_path: str):
        self.audio_file_path = audio_file_path
        self.device = device
        self.temp_dir_path = temp_dir_path
        self.model_cache_path = model_cache_path

    def start(self):
        self.nemo_process = subprocess.Popen(
            [
                sys.executable,  # Use the current Python interpreter
                "-m",
                "whisper_diarization.nemo_process",
                "-a",
                self.audio_file_path,
                "--device",
                self.device,
                "--temp-dir",
                self.temp_dir_path,
                "--model-cache-dir",
                self.model_cache_path
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=os.environ.copy(),  # Pass current environment
        )

    def wait_for_results(self):
        # Wait for NeMo process to complete
        nemo_return_code = self.nemo_process.wait()
        nemo_error_trace = self.nemo_process.stderr.read()
        assert nemo_return_code == 0, (
            "Diarization failed with the following error:"
            f"\n{nemo_error_trace.decode('utf-8')}"
        )

        speaker_ts = get_speaker_timestamps(self.temp_dir_path)
        return speaker_ts


class DiarizationPipeline:
    def __init__(self, cache_root: str = None):
        """Initialize the diarization pipeline with optional cache directory."""
        if cache_root is None:
            cache_root = os.path.join(os.getcwd(), "checkpoints")

        self.CACHE_DIR = cache_root
        self.WHISPER_CACHE_DIR = os.path.join(self.CACHE_DIR, "whisper")
        self.ALIGNMENT_CACHE_DIR = os.path.join(self.CACHE_DIR, "alignment")
        self.PUNCT_CACHE_DIR = os.path.join(self.CACHE_DIR, "punctuation")
        self.NEMO_CACHE_DIR = os.path.join(self.CACHE_DIR, "nemo")

        # Create cache directories
        for directory in [
            self.WHISPER_CACHE_DIR,
            self.ALIGNMENT_CACHE_DIR,
            self.PUNCT_CACHE_DIR,
            self.NEMO_CACHE_DIR,
        ]:
            os.makedirs(directory, exist_ok=True)

        # Set NeMo cache directory
        os.environ["NEMO_CACHE_DIR"] = self.NEMO_CACHE_DIR

        # Create temporary directory
        self.TEMP_DIR = tempfile.mkdtemp()

    def cleanup(self):
        """Clean up temporary directory."""
        if os.path.exists(self.TEMP_DIR):
            shutil.rmtree(self.TEMP_DIR)

    def load_cached_alignment_model(self, device: str, dtype: torch.dtype):
        """Load alignment model with caching support."""
        os.environ["TRANSFORMERS_CACHE"] = self.ALIGNMENT_CACHE_DIR
        os.environ["HF_HOME"] = self.ALIGNMENT_CACHE_DIR
        return load_alignment_model(
            device=device,
            model_path="MahmoudAshraf/mms-300m-1130-forced-aligner",
            dtype=dtype,
        )

    def process_audio(self, audio_path: str, args: Any) -> Dict:
        """Process audio file and return diarization results."""
        # Handle audio stemming if needed
        vocal_target = self._process_audio_stemming(audio_path, args)

        # Get transcription and word timestamps
        audio_waveform, word_timestamps, scores, info = self._transcribe_audio(
            vocal_target, args
        )

        diarizer_config = self.create_diarizer_config()
        # Get speaker timestamps
        speaker_timestamps = self._get_speaker_timestamps(diarizer_config, vocal_target, args)

        # Create word-speaker mapping
        wsm = get_words_speaker_mapping(word_timestamps, speaker_timestamps, "start")

        # Add punctuation if available
        wsm = self._add_punctuation(wsm, info, args)

        # Create final output
        return self._create_output(wsm, speaker_timestamps, scores, audio_path)

    def _process_audio_stemming(self, audio_path: str, args: Any) -> str:
        """Process audio stemming if enabled."""
        if not args.stemming:
            return audio_path

        return_code = os.system(
            f'python -m demucs.separate -n htdemucs --two-stems=vocals "{audio_path}" -o "{self.TEMP_DIR}" --device "{args.device}"'
        )

        if return_code != 0:
            logging.warning(
                "Source splitting failed, using original audio file. "
                "Use --no-stem argument to disable it."
            )
            return audio_path

        return os.path.join(
            self.TEMP_DIR,
            "htdemucs",
            os.path.splitext(os.path.basename(audio_path))[0],
            "vocals.wav",
        )

    def _transcribe_audio(
        self, audio_path: str, args: Any
    ) -> Tuple[np.ndarray, List, List, Any]:
        """Transcribe audio and get word timestamps."""
        whisper_model = faster_whisper.WhisperModel(
            args.model_name,
            device=args.device,
            compute_type=mtypes[args.device],
            download_root=self.WHISPER_CACHE_DIR,
        )
        whisper_pipeline = faster_whisper.BatchedInferencePipeline(whisper_model)
        audio_waveform = faster_whisper.decode_audio(audio_path)

        suppress_tokens = (
            find_numeral_symbol_tokens(whisper_model.hf_tokenizer)
            if args.suppress_numerals
            else [-1]
        )

        if args.batch_size > 0:
            transcript_segments, info = whisper_pipeline.transcribe(
                audio_waveform,
                args.language,
                suppress_tokens=suppress_tokens,
                batch_size=args.batch_size,
            )
        else:
            transcript_segments, info = whisper_model.transcribe(
                audio_waveform,
                args.language,
                suppress_tokens=suppress_tokens,
                vad_filter=True,
            )

        full_transcript = "".join(segment.text for segment in transcript_segments)

        # Get word timestamps through forced alignment
        word_timestamps, scores = self._get_word_timestamps(
            audio_waveform, full_transcript, info, args
        )

        # Cleanup
        del whisper_model, whisper_pipeline
        torch.cuda.empty_cache()

        return audio_waveform, word_timestamps, scores, info

    def download_diarization_model(self):
        msdd_model = EncDecDiarLabelModel.from_pretrained(model_name="diar_msdd_telephonic", return_model_file=True)
        vad_model = EncDecClassificationModel.from_pretrained(model_name="vad_multilingual_marblenet", return_model_file=True)
        # speaker_model = EncDecSpeakerLabelModel.from_pretrained(model_name="titanet_large", return_model_file=True)
        nemo_prototype_model_config_path = load_prototype_config(self.NEMO_CACHE_DIR)
        return nemo_prototype_model_config_path, msdd_model, vad_model

    def create_diarizer_config(self):
        config_path, msdd_model, vad_model = self.download_diarization_model()
        return create_config(
            output_dir=self.TEMP_DIR,
            model_config_path=config_path,
            msdd_model_path=msdd_model,
            vad_model_path=vad_model)

    def _get_speaker_timestamps(self, diarizer_config: Any, audio_path: str, device: str) -> List:
        """Get speaker timestamps from NeMo diarization."""
        import torchaudio

        # Convert audio to mono for NeMo compatibility
        audio_waveform = faster_whisper.decode_audio(audio_path)
        os.makedirs(self.TEMP_DIR, exist_ok=True)
        torchaudio.save(
            os.path.join(self.TEMP_DIR, "mono_file.wav"),
            torch.from_numpy(audio_waveform).unsqueeze(0).float(),
            16000,
            channels_first=True,
        )

        # Initialize NeMo MSDD diarization model
        msdd_model = NeuralDiarizer(cfg=diarizer_config).to(device)
        msdd_model.diarize()

        del msdd_model
        torch.cuda.empty_cache()

        return get_speaker_timestamps(self.TEMP_DIR)


    def _get_word_timestamps(
        self, audio_waveform: np.ndarray, transcript: str, info: Any, args: Any
    ) -> Tuple[List, List]:
        """Get word-level timestamps through forced alignment."""
        alignment_model, alignment_tokenizer = self.load_cached_alignment_model(
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
            transcript,
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

        return word_timestamps, scores

    def _add_punctuation(self, wsm: List, info: Any, args: Any) -> List:
        """Add punctuation to the transcript if available."""
        if info.language not in punct_model_langs:
            logging.warning(
                f"Punctuation restoration is not available for {info.language} language."
                " Using the original punctuation."
            )
            return wsm

        punct_model = PunctuationModel(
            model="kredor/punctuate-all",
            cache_dir=self.PUNCT_CACHE_DIR,
        )

        words_list = list(map(lambda x: x["word"], wsm))
        labeled_words = punct_model.predict(words_list, chunk_size=230)

        ending_puncts = ".?!"
        model_puncts = ".,;:!?"
        is_acronym = lambda x: re.fullmatch(r"\b(?:[a-zA-Z]\.){2,}", x)

        for word_dict, labeled_tuple in zip(wsm, labeled_words):
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

        return get_realigned_ws_mapping_with_punctuation(wsm)

    def _create_output(
        self, wsm: List, speaker_ts: List, scores: List, audio_path: str | None = None
    ) -> Dict:
        """Create final output with word and segment level information."""
        ssm = get_sentences_speaker_mapping(wsm, speaker_ts)

        # Create word-level output
        word_level_output = []
        for word_info, score in zip(wsm, scores):
            confidence = float(np.exp(score)) if score is not None else 1.0
            word_entry = {
                "speaker": word_info["speaker"],
                "start": word_info["start_time"] / 1000.0,
                "end": word_info["end_time"] / 1000.0,
                "word": word_info["word"].strip(),
                "confidence": confidence,
            }
            word_level_output.append(word_entry)

        # Create segment-level output
        segment_output = []
        current_word_idx = 0
        for segment in ssm:
            speaker_id = int(segment["speaker"].split()[-1])
            segment_entry = {
                "speaker": speaker_id,
                "start": segment["start_time"] / 1000.0,
                "end": segment["end_time"] / 1000.0,
                "text": segment["text"].strip(),
                "words": [],
            }

            while (
                current_word_idx < len(word_level_output)
                and word_level_output[current_word_idx]["start"]
                <= segment["end_time"] / 1000.0
            ):
                if (
                    word_level_output[current_word_idx]["start"]
                    >= segment["start_time"] / 1000.0
                ):
                    segment_entry["words"].append(word_level_output[current_word_idx])
                current_word_idx += 1

            segment_output.append(segment_entry)

        if audio_path is not None:
            # Write output files
            base_path = os.path.splitext(audio_path)[0]
            with open(f"{base_path}.txt", "w", encoding="utf-8-sig") as f:
                get_speaker_aware_transcript(ssm, f)

            with open(f"{base_path}.srt", "w", encoding="utf-8-sig") as srt:
                write_srt(ssm, srt)

            with open(f"{base_path}_segments.json", "w", encoding="utf-8-sig") as f:
                json.dump(segment_output, f, indent=2, ensure_ascii=False, default=str)

        return {
            "segments": segment_output,
            "word_timestamps": word_level_output,
        }
