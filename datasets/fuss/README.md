# README for the Free Universal Sound Separation (FUSS) Dataset
This data is used in
<a href="http://dcase.community/challenge2020/task-sound-event-detection-and-separation-in-domestic-environments">
DCASE2020 Challenge Task 4: Sound Event Detection and Separation in Domestic
Environments</a>, and uses
<a href="https://github.com/justinsalamon/scaper">scaper</a> for mixing.

You can use the scripts in this directory for data preparation for source
separation training in the DCASE2020 Challenge Task 4, or for stand-alone
separation experiments. Please read the scripts to understand what they do.
Below are pointers to the top-level scripts to get you started.

## Data Preparation

### Option A: Install baseline prepared data (recommended)

You can download the prepared baseline training, validation and eval data by
running

```
    bash ./get_dev_data.sh

```

This downloads and installs both of the following:

*   ssdata: the dry mixture and sources (no reverberation - optional)
*   ssdata_reverb: the reverberated mixture and sources

These data are the exact data used to train the baseline separation model. They
were prepared using labels to ensure that no mixture has more than one
overlapping source with the same label.

### Option B: Data augmentation (advanced)

To generate more training and validation data for data augmentation, you can run
the following script.

```
    bash ./run_data_augmentation.sh

```

This script downloads raw data and processes them to generate 20000 train and
1000 validation examples by default. Random elements of the dry mixture
generation and reverberated mixture generation are controlled by a single
RANDOM_SEED variable.

The purpose of the RANDOM_SEED, NUM_TRAIN and NUM_VAL variables are to allow
participants to experiment with different data augmentations. When using these
variables, please ablate your systems to compare the results with the baseline
data (Option A).

These data are prepared without using original event labels, so in theory some
mixtures may contain multiple events from the same original label, although this
should happen very rarely. Otherwise the generated folders are going to be
similar to the prepared data provided in Option A.

You can check inside the scripts to change certain parameters. Especially, you
probably should change desired folder names for downloading, extracting and
processing data. You can also change the RANDOM_SEED, NUM_TRAIN, NUM_VAL
variables to generate different versions and quantities of the augmented
training and validation data.

## Data License

See the
<a href=https://github.com/google-research/sound-separation/blob/master/datasets/fuss/FUSS_license_doc/README.md>
FUSS_license_doc/README.md</a> for information about the license for data
downloaded by the scripts in this directory.
