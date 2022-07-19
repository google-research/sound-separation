# README for AudioScope YFCC100m clip lists.
This is a recipe for data based on YFCC100m [1] used in
our ICLR 2021 paper [2] on audio-visual on-screen sound separation.

Specifically, we provide CSVs that describe the exact videos and timestamps used
for labeled and unlabeled train, validation, and test clips, as well as specification of pairs of clips used to create mixture of mixtures (MoM) validation and test sets.
The same train/validation/test splits are used as in <a href="https://github.com/google-research/sound-separation/tree/master/datasets/yfcc100m">these lists</a>. All clips referenced by these lists have been filtered by an unsupervised audio-visual coincidence model trained on AudioSet [3] using a threshold of 0.8 on the predicted coincidence probability.

| Split     | Label           | Count       | CSV name                                       |
|-----------|-----------------|-------------|------------------------------------------------|
|Train      | None            | 324970      | filtered_train_clips.csv                       |
|Train      | On-screen only  | 836         | filtered_train_onscreen_unanimous_clips.csv    |
|Train      | Off-screen only | 3672*       | filtered_train_offscreen_unanimous_clips.csv   |
|Validation | None            | 6429        | filtered_validate_clips.csv                    |
|Validation | On-screen only  | 735         | filtered_validate_onscreen_unanimous_clips.csv |
|Validation | Off-screen only | 836         | filtered_validate_offscreen_unanimous_clips.csv|
|Test       | None            | 3293        | filtered_test_clips.csv                        |
|Test       | On-screen only  | 295         | filtered_test_onscreen_unanimous_clips.csv     |
|Test       | Off-screen only | 370         | filtered_test_offscreen_unanimous_clips.csv     |

\* The paper [2] gives the incorrect count of 3681.

Addionally, we provide lists of the pairs of clips used to create MoMs for validation and test, where the MoM video uses video frames from the first clip, and a soundtrack that is the sum of the audio from both clips.

| Split     | Label                   | Count       | CSV name                                                                    |
|-----------|-------------------------|-------------|-----------------------------------------------------------------------------|
|Validation | On-screen + off-screen  | 3675        | filtered_validate_onscreen_unanimous_plus_offscreen_unanimous_mom_clips.csv |
|Validation | Off-screen + off-screen | 4180        | filtered_validate_offscreen_unanimous_plus_offscreen_unanimous_mom_clips.csv|
|Test       | On-screen + off-screen  | 1475        | filtered_test_onscreen_unanimous_plus_offscreen_unanimous_mom_clips.csv     |
|Test       | Off-screen + off-screen | 1850        | filtered_test_offscreen_unanimous_plus_offscreen_unanimous_mom_clips.csv    |

## Download instructions

The CSVs are hosted on Google Cloud. They can be downloaded using the following command:

```
gsutil -m cp -r gs://gresearch/sound_separation/audioscope_yfcc100m_clip_lists .
```

which will copy the CSVs to the current folder.

## CSV format

The train, validation, and test CSVs contain the following columns:

* Video path: string, path to MP4 video in YFCC100m. Here is an example of a video path:
```data/videos/mp4/827/f6b/827f6b53467db2d5218ed8247418c4c.mp4```
* Input start time: float, clip start time in seconds.
* Input end time: float, clip end time in seconds.

The validation and test MoM CSVs contain the following columns, with each row describing two
clips that are used to construct the MoM:

* Video path 1
* Input start time 1
* Input end time 1
* Video path 2
* Input start time 2
* Input end time 2

## Data License

These lists are released under a <a href="https://creativecommons.org/licenses/by/4.0/">Creative Commons license (CC-BY 4.0)</a>.

## References

<a href="https://dl.acm.org/doi/pdf/10.1145/2812802">[1] Bart Thomee, David Shamma, Gerald Friedland, Benjamin Elizalde, Karl Ni, Dough Poland, Damian Borth, Li-Jia Li, "YFCC100M: The New Data in Multimedia Research", Communications of the ACM, 59(2), pp. 64-73, 2016.</a>

<a href="https://openreview.net/forum?id=MDsQkFP1Aw">[2] Efthymios Tzinis, Scott Wisdom, Aren Jansen, Shawn Hershey, Tal Remez, Daniel P. W. Ellis, John R. Hershey, "Into the Wild with AudioScope: Unsupervised Audio-Visual Separation of On-Screen Sounds", International Conference on Learning Representations (ICLR), 2021.</a>

<a href="https://arxiv.org/abs/1911.05894">[3] Aren Jansen, Daniel P. W. Ellis, Shawn Hershey, R. Channing Moore, Manoj Plakal, Ashok C. Popat,
and Rif A. Saurous, "Coincidence, categorization, and consolidation: Learning to recognize sounds
with minimal supervision", IEEE International Conference on Acoustics, Speech, and
Signal Processing (ICASSP), pp. 121–125, 2020.</a>
