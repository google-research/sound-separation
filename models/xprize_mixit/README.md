## XPRIZE MixIT Model

This model was trained for the XPRIZE rainforest competition.
It is a mixture-invariant training (MixIT) model that can separate up to 4 sources.

## Dataset

This model was trained on a dataset that was created for the competition.
The dataset was collected from Xeno-Canto and collaborating institutions.
In particular, the collaborating institutions are:
- [FNJV](https://www2.ib.unicamp.br/fnjv/)
The dataset contains all available audio from Amazon basin.

## Install Prerequisites

FFMPEG

- Ubuntu/Debian: `sudo apt update && sudo apt install ffmpeg`
- macOS: `brew install ffmpeg`
- Windows: Download from the official ffmpeg website and add it to your system's PATH.

## Create a virtual environment

```bash
python3.10 -m venv venv
```

## Install packaging tool

```bash
venv/bin/python -m pip install --timeout 1 --upgrade pip
```

## Install requirements

```bash
venv/bin/python -m pip install --timeout 1 -r ../../requirements.in pydub
```

## Download the model

```bash
venv/bin/python download_checkpoints.py
```

## Download a recording

```bash
venv/bin/python download_recording.py
```

## Separate channels of a recording

```bash
PYTHONPATH=../.. venv/bin/python channels_of_file_separating.py \
    --channel_count 8 \
    --input_filepath ../../datasets/xeno-canto/XC771373.wav \
    --output_filepath ../../datasets/xeno-canto/XC771373.wav \
    --model_name xprize_mixit
```

## List all nodes and their dimensions

```bash
PYTHONPATH=../.. venv/bin/python layer_dimension_listing.py
```

## Create embedding of a recording

```bash
PYTHONPATH=../.. venv/bin/python embedding_creation.py \
    --input_filepath ../../datasets/xeno-canto/XC771373.wav \
    --model_name xprize_mixit
```
