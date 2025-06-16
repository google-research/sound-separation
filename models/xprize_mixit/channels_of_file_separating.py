from argparse import ArgumentParser
from argparse import Namespace
from pathlib import Path

from paths import MODELS_PATH
from channel_separation_shim import \
    separate_channels_using_mixit
from tensorflow_shim import get_determinism


def separate_channels_of_file(
    *,
    channel_count: int,
    input_filepath: Path,
    output_filepath: Path,
    model_name: str,
) -> None:
    input_file: str = str(input_filepath)
    model_directory: str
    if model_name == 'xprize_mixit':
        if channel_count == 8:
            model_directory = str(
                MODELS_PATH / 'xprize_mixit' / 'checkpoints'
            )
        else:
            raise NotImplementedError()
    else:
        raise NotImplementedError(
            f"Model {model_name} not implemented for channel count {channel_count}"
        )

    checkpoint: str
    if model_name == 'xprize_mixit':
        if channel_count == 8:
            checkpoint = str(
                MODELS_PATH / 'xprize_mixit' / 'checkpoints' / 'model.ckpt-1305906'
            )
        else:
            raise NotImplementedError()
    else:
        raise NotImplementedError()

    input_channels: int = 0
    scale_input: bool = False
    input_tensor: str = 'input_audio/receiver_audio:0'
    output_tensor: str = 'denoised_waveforms:0'
    output_channels: int = 0
    output_file: str = str(output_filepath)
    write_outputs_separately: bool = True

    try:
        separate_channels_using_mixit(
            checkpoint=checkpoint,
            input_channels=input_channels,
            input_file=input_file,
            input_tensor=input_tensor,
            model_directory=model_directory,
            channel_count=channel_count,
            output_channels=output_channels,
            output_file=output_file,
            output_tensor=output_tensor,
            scale_input=scale_input,
            write_outputs_separately=write_outputs_separately,
        )
    except ValueError as exception:
        print(
            f'Skipping separation of {input_file} '
            f'due to exception: {exception}',
            flush=True,
        )
        return


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--channel_count')
    parser.add_argument('--input_filepath')
    parser.add_argument('--output_filepath')
    parser.add_argument('--model_name')
    args: Namespace = parser.parse_args()
    print(args)

    get_determinism()

    separate_channels_of_file(
        channel_count=int(args.channel_count),
        input_filepath=args.input_filepath,
        output_filepath=args.output_filepath,
        model_name=args.model_name,
    )
