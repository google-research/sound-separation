from models.neurips2020_mixit.train_model_on_fuss import train_model_on_fuss
from tests._paths import ROOT_PATH


def test_train_model_on_fuss():
    # Configuration
    line_limit = 4
    train_steps = 4

    # Arrange
    train_example_file_path = (
        ROOT_PATH /
        'datasets/fuss/data/fuss_dev/ssdata/train_example_list.txt'
    )
    test_train_example_file_path = (
        ROOT_PATH /
        'datasets/fuss/data/fuss_dev/ssdata/test_train_example_list.txt'
    )
    _copy_first_lines(
        input_file_path=train_example_file_path,
        output_file_path=test_train_example_file_path,
        line_limit=line_limit,
    )

    validation_example_file_path = (
        ROOT_PATH /
        'datasets/fuss/data/fuss_dev/ssdata/validation_example_list.txt'
    )
    test_evaluation_example_file_path = (
        ROOT_PATH /
        'datasets/fuss/data/fuss_dev/ssdata/test_validation_example_list.txt'
    )
    _copy_first_lines(
        input_file_path=validation_example_file_path,
        output_file_path=test_evaluation_example_file_path,
        line_limit=line_limit,
    )

    test_validation_list_file_path_string = str(
        ROOT_PATH /
        'datasets/fuss/data/fuss_dev/ssdata/test_validation_example_list.txt'
    )
    test_train_list_file_path_string = str(
        ROOT_PATH /
        'datasets/fuss/data/fuss_dev/ssdata/test_train_example_list.txt'
    )
    model_dir = str(
        ROOT_PATH / 'models/neurips2020_mixit/model_dir'
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
