import argparse
import os

import torch

from nemo.collections.asr.models.msdd_models import NeuralDiarizer
from pydub import AudioSegment

from .helpers import create_config

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
args = parser.parse_args()

# convert audio to mono for NeMo compatibility
sound = AudioSegment.from_file(args.audio).set_channels(1)
os.makedirs(args.temp_dir, exist_ok=True)
sound.export(os.path.join(args.temp_dir, "mono_file.wav"), format="wav")

# Initialize NeMo MSDD diarization model
msdd_model = NeuralDiarizer(cfg=create_config(args.temp_dir)).to(args.device)
msdd_model.diarize()
