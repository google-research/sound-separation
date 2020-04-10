
# Free Universal Sound Separation (FUSS) Dataset

The Free Universal Sound Separation (FUSS) Dataset is a database of arbitrary sound mixtures and source-level references, for use in experiments on arbitrary sound separation.

This is the official sound separation data for the DCASE2020 Challenge Task 4: Sound Event Detection and Separation in Domestic Environments.

## Citation

If you use the FUSS dataset or part of it, please cite our paper describing the dataset and baseline [1].
FUSS is based on <a href="https://annotator.freesound.org/fsd/">FSD50K corpus</a> so please also cite [2]:


### Data Curators

Scott Wisdom, Hakan Erdogan, Dan Ellis, John R. Hershey (Google Research)

### Collaborators

Eduardo Fonseca (Universitat Pompeu Fabra), Frederic Font Corbera (Universitat Pompeu Fabra), Romain Serizel (LORIA), Nicolas Turpault (INRIA), Justin Salamon (Adobe Research), Prem Seetharaman (Northwestern University)

### Contact

You are welcome to contact Scott Wisdom should you have any questions at scottwisdom@google.com.

## About the FUSS dataset

### Overview
FUSS audio data is sourced from a prerelease of <a href="https://annotator.freesound.org/fsd/">Freesound dataset</a> known as (FSD50K), a sound event dataset composed of Freesound content annotated with labels from the AudioSet Ontology. Using the FSD50K labels, these source files have been screened such that they likely only contain a single type of sound. Labels are not provided for these source files, and are not considered part of the challenge. For the purpose of the DCASE Task4 Sound Separation and Event Detection challenge, systems should not use FSD50K labels, even though they may become available upon FSD50K release.

To create mixtures, 10 second clips of sources are convolved with simulated room impulse responses and added together. Each 10 second mixture contains between 1 and 4 sources. Source files longer than 10 seconds are considered "background" sources. Every mixture contains one background source, which is active for the entire duration.
We provide: a software recipe to create the dataset, the room impulse responses, and the original source audio.

### Motivation for use in <a href="http://dcase.community/challenge2020/task-sound-event-detection-and-separation-in-domestic-environments">DCASE2020 Challenge Task 4</a>

This dataset provides a platform to investigate how source separation may help with event detection and vice versa.  Previous work has shown that universal sound separation (separation of arbitrary sounds) is possible [3], and that event detection can help with universal sound separation [4].  It remains to be seen whether sound separation can help with event detection. Event detection is more difficult in noisy environments, and so separation could be a useful pre-processing step. Data with strong labels for event detection are relatively scarce, especially when restricted to specific classes within a domain. In contrast, source separation data needs no event labels for training, and may be more plentiful. In this setting, the idea is to utilize larger unlabeled separation data to train separation systems, which can serve as a front-end to event-detection systems trained on more limited data.

### Room simulation

Room impulse responses are simulated using the image method with frequency-dependent walls. Each impulse corresponds to a rectangular room of random size with random wall materials, where a single microphone and up to 4 sources are placed at random spatial locations.

### Recipe for data creation

The data creation recipe starts with scripts, based on <a href="https://github.com/justinsalamon/scaper">scaper</a> [5],
to generate mixtures of events with random timing of source events, along with a background source that spans the duration of the mixture clip.
The repo for this is at <a href="https://github.com/google-research/sound-separation/tree/master/datasets/fuss">this GitHub repo</a>.
The constituent source files for each mixture are also generated for use as references for training and evaluation.
The data are reverberated using a different room simulation for each mixture.
In this simulation each source has its own reverberation corresponding to a different spatial location.
The reverberated mixtures are created by summing over the reverberated sources.
The dataset recipe scripts support modification, so that participants may remix and augment the training data as desired.

### Format

All audio clips are provided as uncompressed PCM 16 bit, 16 kHz, mono audio files.

### Data split

The FUSS dataset is partitioned into "train", "validation", and "eval" sets, following the same splits used in FSD data. Specifically, the train and validation sets are sourced from the FSD50K dev set, and we have ensured that clips in train come from different uploaders than the clips in validation. The eval set is sourced from the FSD50K eval split.

## Baseline System

A baseline system for the FUSS dataset is available at <a href="https://github.com/google-research/sound-separation/tree/master/models/dcase2020_fuss_baseline">dcase2020_fuss_baseline</a>.

## License
All audio clips (i.e., in  FUSS_fsd_data.tar.gz) used in the preparation of Free Universal Source Separation (FUSS) dataset are designated Creative Commons (CC0) and were obtained from <a href="https://freesound.org">freesound</a>.  The source data in FUSS_fsd_data.tar.gz were selected using labels from the <a href="https://annotator.freesound.org/fsd/">FSD50K corpus</a>, which is licensed as Creative Commons Attribution 4.0 International (CC BY 4.0) License.

The FUSS dataset as a whole, is a curated, reverberated, mixed, and partitioned preparation, and is released under the Creative Commons Attribution 4.0 International (CC BY 4.0) License. This license is specified in the `LICENSE-DATASET` file downloaded with the `FUSS_license_doc.tar.gz` file.

## Files & Download

- FUSS_license_doc.tar.gz           License and this readme
- FUSS_fsd_data.tar.gz              Original input source files
- FUSS_ssdata.tar.gz                Un-reverberated mixtures and reference sources produced by scaper
- FUSS_rir_data.tar.gz              Simulated room impulse responses (RIRs)
- FUSS_ssdata_reverb.tar.gz         Reverberated mixtures and reference sources
- FUSS_baseline_model.tar.gz        Baseline separation model


## References
[1] Scott Wisdom, Hakan Erdogan, Daniel P. W. Ellis, Romain Serizel, Nicolas Turpault, Eduardo Fonseca, Justin Salamon, Prem Seetharaman, John R. Hershey,
"What's All the FUSS About Free Universal Sound Separation Data?", 2020, in preparation.

[2] Eduardo Fonseca, Jordi Pons, Xavier Favory, Frederic Font Corbera, Dmitry Bogdanov, Andrés Ferraro, Sergio Oramas, Alastair Porter, and Xavier Serra. "Freesound Datasets: A Platform for the Creation of Open Audio Datasets."  International Society for Music Information Retrieval Conference (ISMIR), pp. 486–493. Suzhou, China, 2017.

[3] Ilya Kavalerov, Scott Wisdom, Hakan Erdogan, Brian Patton, Kevin Wilson, Jonathan Le Roux, and John R. Hershey. "Universal Sound Separation." IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA), pp. 175-179. New Paltz, NY, USA, 2019.

[4] Efthymios Tzinis, Scott Wisdom, John R. Hershey, Aren Jansen, and Daniel P. W. Ellis. "Improving Universal Sound Separation Using Sound Classification." IEEE International Conference on Acoustics Speech and Signal Processing (ICASSP), 2020.

[5] J. Salamon, D. MacConnell, M. Cartwright, P. Li, and J. P. Bello., "Scaper: A Library for Soundscape Synthesis and Augmentation", In IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA), New Paltz, NY, USA, Oct. 2017.


