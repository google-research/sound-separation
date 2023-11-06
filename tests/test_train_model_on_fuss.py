import datetime
from pathlib import Path

import pytest

from models.neurips2020_mixit.train_model_on_fuss import train_model_on_fuss
from tests._paths import ROOT_PATH


@pytest.mark.parametrize(
    ('dataset_name', 'dataset_path'),
    [
        ('fuss', 'fuss/fuss_dev/ssdata'),
        ('anuraset', 'AnuraSet_Dev/anuraset_dev/ssdata'),
        ('rfcx', 'rfcx_dev/'),
        # Fails:
        # ('congo_soundscapes_dev', 'congo_soundscapes_dev/10_random_examples'),
    ],
)
def test_train_model_on_fuss(dataset_name, dataset_path):
    # Configuration
    line_limit = 4
    train_steps = 4

    # Arrange
    root_data_audio_path: Path = ROOT_PATH / '3-datasets' / 'audio'
    train_example_file_path = (
            root_data_audio_path / dataset_path / 'train_example_list.txt'
    )
    test_train_example_file_path = (
            root_data_audio_path / dataset_path / 'test_train_example_list.txt'
    )
    _copy_first_lines(
        input_file_path=train_example_file_path,
        output_file_path=test_train_example_file_path,
        line_limit=line_limit,
    )

    validation_example_file_path = (
            root_data_audio_path / dataset_path / 'validation_example_list.txt'
    )
    test_evaluation_example_file_path = (
            root_data_audio_path / dataset_path / 'test_validation_example_list.txt'
    )
    _copy_first_lines(
        input_file_path=validation_example_file_path,
        output_file_path=test_evaluation_example_file_path,
        line_limit=line_limit,
    )

    test_validation_list_file_path_string = str(
        root_data_audio_path / dataset_path / 'test_validation_example_list.txt'
    )
    test_train_list_file_path_string = str(
        root_data_audio_path / dataset_path / 'test_train_example_list.txt'
    )
    datetime_slug = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    datetime_slug += '-test'
    root_models_path: Path = ROOT_PATH / '5-models' / 'models'
    model_dir = str(
        root_models_path / 'neurips2020_mixit' / dataset_name / datetime_slug
    )

    # Act
    train_model_on_fuss(
        train_list=test_train_list_file_path_string,
        validation_list=test_validation_list_file_path_string,
        model_dir=model_dir,
        train_steps=train_steps,
    )


def _copy_first_lines(
    input_file_path,
    output_file_path,
    line_limit,
):
    with open(input_file_path, 'r') as input_file:
        with open(output_file_path, 'w') as output_file:
            for _ in range(line_limit):
                output_file.write(input_file.readline())
