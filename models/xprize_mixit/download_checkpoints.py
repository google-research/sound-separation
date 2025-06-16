"""
Download the model checkpoint files using python and the standard library
"""
from pathlib import Path

import os
import urllib.request

from paths import ROOT_PATH

# Define the URLs of the files to download
urls: list[str] = [
    "https://huggingface.co/XPRIZE-Brazilian-Team/XPRIZE-MixIT/resolve/main/model.ckpt-1305906.data-00000-of-00001",
    "https://huggingface.co/XPRIZE-Brazilian-Team/XPRIZE-MixIT/resolve/main/model.ckpt-1305906.index",
    "https://huggingface.co/XPRIZE-Brazilian-Team/XPRIZE-MixIT/resolve/main/model.ckpt-1305906.meta",
]

# Define the directory to save the checkpoint files
checkpoints_path: Path = ROOT_PATH / 'models' / 'xprize_mixit' / 'checkpoints'

# Create the checkpoint file that TensorFlow expects
checkpoint_file_path: Path = checkpoints_path / 'checkpoint'
checkpoint_content: str = (
    'model_checkpoint_path: "model.ckpt-1305906"\n'
    'all_model_checkpoint_paths: "model.ckpt-1305906"\n'
)
with open(checkpoint_file_path, 'w') as f:
    f.write(checkpoint_content)
print(f"Created checkpoint file at {checkpoint_file_path}")

# Create the directory if it doesn't exist
os.makedirs(checkpoints_path, exist_ok=True)

# Download each file
for url in urls:
    # Extract the filename from the URL
    filename: str = os.path.basename(url)

    # Define the full path to save the file
    file_path: Path = checkpoints_path / filename

    # Download the file
    urllib.request.urlretrieve(url, file_path)
    print(f"Downloaded {filename} to {file_path}")

    # Check if the file is not a .meta file
    if not filename.endswith('.meta'):
        continue

    # Rename the .meta file to inference.meta, because the model expects it
    inference_meta_path: Path = checkpoints_path / 'inference.meta'
    os.rename(file_path, inference_meta_path)
