"""
Download a recording from xeno-canto.
"""

import os
import urllib.request
from pathlib import Path

from pydub import AudioSegment

from paths import ROOT_PATH

# Define the URL of the recording to download
url: str = "https://xeno-canto.org/771373/download"

# Define the directory to save the recording
dataset_path: Path = ROOT_PATH / 'datasets' / 'xeno-canto'

# Create the directory if it doesn't exist
os.makedirs(dataset_path, exist_ok=True)

# Define the full path to save the recording
recording_path: Path = dataset_path / 'XC771373.mp3'

# Download the recording
urllib.request.urlretrieve(url, recording_path)
print(f"Downloaded recording to {recording_path}")

# Load the mp3 file
sound: AudioSegment = AudioSegment.from_mp3(recording_path)

# Convert to 48 kHz
sound = sound.set_frame_rate(48000)

# Convert to mono
sound = sound.set_channels(1)

# cut the first 3 seconds
start_time: int = 0  # milliseconds
end_time: int = 3000  # milliseconds
sound = sound[start_time:end_time]

# Export the sound to a wav file
sound.export(recording_path.with_suffix('.wav'), format="wav")
