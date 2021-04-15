# README for YFCC100m clip lists.
This is a recipe for data based on YFCC100m [1] used in
our NeurIPS 2020 paper [2] on unsupervised sound separation for mixture invariant training.

Specifically, we provide CSVs that describe the exact videos and timestamps used
for train, validation, and test clips, as well as specification of pairs of clips used to create the test set of mixtures of mixtures (MoMs), used for universal sound separation experiments in [2].

## Download instructions

The CSVs are hosted on Google Cloud. They can be downloaded using the following command:

```
gsutil -m cp -r gs://gresearch/sound_separation/yfcc100m_clip_lists .
```

which will copy the CSVs to the current folder.

## CSV format

The train, validation, and test CSVs contain the following columns:

* Video path: string, path to MP4 video in YFCC100m. Here is an example of a video path:
```data/videos/mp4/827/f6b/827f6b53467db2d5218ed8247418c4c.mp4```
* Input start time: float, clip start time in seconds.
* Input end time: float, clip end time in seconds.
* Output offset: float, offset of input clip start from the beginning of 10-second output clip. This is nonzero only for input clips shorter than 10 seconds, i.e. input end time minus input start time is less than 10. Input clips that are shorter than 10s are padded with zeros on either side, and start at the specified offset within the 10 second output clip.

The test MoM CSV contains the following columns, with each row describing two
clips that are used to construct the MoM:

* Video path 1
* Input start time 1
* Input end time 1
* Output offset 1
* Video path 2
* Input start time 2
* Input end time 2
* Output offset 2

## Data License

These lists are released under a <a href="https://creativecommons.org/licenses/by/4.0/">Creative Commons license (CC-BY 4.0)</a>.

## References

[1] Bart Thomee, David Shamma, Gerald Friedland, Benjamin Elizalde, Karl Ni, Dough Poland, Damian Borth, Li-Jia Li, "YFCC100M: The New Data in Multimedia Research", Communications of the ACM, 59(2), pp. 64-73, 2016.

<a href="https://arxiv.org/abs/2006.12701">[2] Scott Wisdom, Efthymios Tzinis, Hakan Erdogan, Ron J. Weiss, Kevin Wilson, John R. Hershey, "Unsupervised Sound Separation Using Mixture Invariant Training", Advances in Neural Information Processing Systems, 33, pp. 3846--3857, 2020.</a>
