import argparse
import os

import torch

from nemo.collections.asr.models.msdd_models import NeuralDiarizer
from pydub import AudioSegment
from faster_whisper import decode_audio
import torchaudio

from .helpers import create_config, load_prototype_config
from .diarization_utils import DiarizationPipeline

parser = argparse.ArgumentParser()
parser.add_argument(
    "-a", "--audio", help="name of the target audio file", required=True
)
parser.add_argument(
    "--device",
    dest="device",
    default="cuda" if torch.cuda.is_available() else "cpu",
    help="if you have a GPU use 'cuda', otherwise 'cpu'",
)
parser.add_argument(
    "--temp-dir",
    dest="temp_dir",
    help="temporary directory for processing",
    required=True,
)
parser.add_argument(
    "--model-cache-dir",
    dest="model_cache_dir",
    help="temporary directory for processing",
    required=False,
)
args = parser.parse_args()

pipeline = DiarizationPipeline(args.model_cache_dir)
pipeline.TEMP_DIR = args.temp_dir
pipeline._get_speaker_timestamps(pipeline.create_diarizer_config(), args.audio, args.device)


#audio_waveform = decode_audio(args.audio)
#os.makedirs(args.temp_dir, exist_ok=True)
#torchaudio.save(
#    os.path.join(args.temp_dir, "mono_file.wav"),
#    torch.from_numpy(audio_waveform).unsqueeze(0).float(),
#    16000,
#    channels_first=True,
#)
#
## Initialize NeMo MSDD diarization model
#
#pipeline = DiarizationPipeline(args.model_cache_dir)
#pipeline.TEMP_DIR = args.temp_dir
#diarizer_config = pipeline.create_diarizer_config()
#
#msdd_model = NeuralDiarizer(cfg=diarizer_config).to(args.device)
#msdd_model.diarize()
