import multiprocessing

from functools import partial
from librosa.feature import melspectrogram

from pathlib import Path

import librosa
import librosa.display
import numpy
from matplotlib import pyplot
import soundfile

from models.xprize_mixit.paths import ROOT_PATH


def create_mel_spectrogram(
    *,
    channel: int | None,
    input_file: Path,
    n_mels: int,
):
    audio: numpy.ndarray
    sample_rate: int
    if channel is not None:
        input_file: Path = (
            input_file.parent / (input_file.stem + f"_channel_{channel}.wav")
        )
    audio, sample_rate = soundfile.read(
        file=input_file,
    )

    mel_spectrogram: numpy.ndarray = (
        melspectrogram(y=audio, sr=sample_rate, n_mels=n_mels)
    )
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=numpy.max)

    # Save the spectrogram as an image
    pyplot.figure(figsize=(10, 4))
    librosa.display.specshow(
        mel_spectrogram_db,
        sr=sample_rate,
        x_axis='time',
        y_axis='mel',
        # cmap='viridis',
    )
    pyplot.colorbar(format='%+2.0f dB')
    if channel is None:
        pyplot.title(f'MEL Spectrogram - {input_file.stem}')
    else:
        pyplot.title(f'MEL Spectrogram - Channel {channel}')
    pyplot.tight_layout()

    output_file: Path = (
        input_file.parent / (input_file.stem + f'_mel_spectrogram.png')
    )
    pyplot.savefig(output_file)
    pyplot.close()

    if channel is None:
        print(f"Saved MEL spectrogram for {input_file.stem} to {output_file}")
    else:
        print(f"Saved MEL spectrogram for channel {channel} to {output_file}")

def main():
    input_file: Path = ROOT_PATH / f'datasets/xeno-canto/XC771373.wav'

    create_mel_spectrogram(
        channel=None,
        input_file=input_file,
        n_mels=256,
    )

    # with multiprocessing.Pool() as pool:
    for i in range(0, 8):
        create_mel_spectrogram_partial: partial = partial(
            create_mel_spectrogram,
            channel=i,
            input_file=input_file,
            n_mels=256,
        )
        create_mel_spectrogram_partial()
        #pool.apply_async(func=create_mel_spectrogram_partial)
    #pool.close()
    #pool.join()


if __name__ == "__main__":
    main()
