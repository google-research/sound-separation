# README for synthetic AMI dataset recipe.
This is a recipe for data based on AMI [1] used in
our ICASSP 2022 paper [2] on using unsupervised sound separation with mixture invariant training (MixIT) [3] for adaptation of speech separation models to real-world meeting data.

Specifically, we provide a CSV and code that provide an exact recipe to recreate the synthetic AMI evaluation set used in [2].

## How to generate the dataset

### 1. Download the audio for the AMI dataset.
You can do this from [this download page](https://groups.inf.ed.ac.uk/ami/download/). Under scenario meetings, you need to select a, b, c, and d for all Edinburgh, IDIAP, and TNO meetings. Under non-scenario meetings, also select all Edinburgh meetings. Select "Individual headsets" and "Microphone array" under "Audio related media streams". The total estimated download size should be 185 GB.

### 2. Run the python generation code.
Run the following command, given the bash variables AMI_DIRECTORY and OUTPUT_DIRECTORY are set to the path for the downloaded AMI dataset and desired output directory, respectively:

```
python3 make_synthetic_ami.py -a ${AMI_DIRECTORY} -o ${OUTPUT_DIRECTORY}
```

## Description of generation procedure

A "segment" is a section of the meeting that is single-speaker, as indicated by the AMI annotations. The wav file path from which a segment is extracted is given by `wav_<bg,fg>`, and the start and end times of the segments are given by `seg_start_<bg,fg>` and `seg_end_<bg,fg>`. For each segment, we use least-squares to estimate the best linear time-invariant finite impulse response (FIR) filter that maps single-speaker headset audio to distant microphone audio. This provides clean reverberant versions of the anechoic headset audio, which can then be mixed together.

This filtering procedure also provides an estimate of the background noise. Given headset audio  $ x $, distant microphone audio  $ y $, and inferred filter $\hat{h}$, the filtered headset is $x*\hat{h}$, and the residual $y - x*\hat{h}$ is an estimate of the background noise. Note that the residual may still contain some speech and thus is an imperfect reference for background noise, since the linear filtering is not perfect.

To construct the synthetic AMI mixtures with their corresponding references, we extract shorter "clips" from the segments decribed above. The offset of a clip within a segment is given by `offset_<bg,fg>`, and the duration of a clip from this offset is given by `duration_<bg,fg>`. Each synthetic AMI example is constructed from two clips: a "background" clip, which is always 5 seconds long, and a "foreground" clip, which has duration less than or equal to 5 seconds. For the background clip, two sources are created: reverberant filtered headset $x*\hat{h}$, and the reverberant residual $y-\hat{h}*x$ that serves as an imperfect reference for background noise. For the foreground clip, a single source is created: the reverberant filtered headset, shifted by `shift_fg`.

The outputs of the `make_synthetic_ami.py` script are a folder for each example, where each folder contains 3 wav files:

* `receiver_audio.wav`: single-channel audio of distant microphone.
* `source_images.wav`: 3-channel audio of reverberant sources, in order of imperfect background noise, background source, foreground source.
* `source_audio.wav`: 3-channel audio of original headset audio. First source is all-zeros, second source is original background headset audio, and third source is original foreground headset audio.

## CSV format

The CSV provides the mixing recipe, and contains the following columns:

* wav_bg: string, relative path to background headset wav.
* seg_start_bg: float, start time in seconds of background clip.
* seg_end_bg: float, end time in seconds of background clip.
* offset_bg: float, offset of background clip within segment.
* duration_bg: float, duration in seconds of background clip (always 5).
* wav_fg: string, relative path to foreground headset wav.
* seg_start_fg: float, start time in seconds of foreground segment.
* seg_end_fg: float, end time in seconds of foreground segment.
* offset_fg: float, offset of foreground clip within segment.
* duration_fg: float, duration in seconds of foreground clip.
* shift_fg: float, shift of fg clip relative to bg clip in seconds.

## Data License

The CSV is released under a <a href="https://creativecommons.org/licenses/by/4.0/">Creative Commons license (CC-BY 4.0)</a>.

Code is under Apache 2.0 license.

## References

<a href="https://link.springer.com/chapter/10.1007/11677482_3">[1] Jean Carletta, Simone Ashby, Sebastien Bourban, Mike Flynn, Mael Guillemot, Thomas Hain, Jaroslav Kadlec, Vasilis Karaiskos, Wessel Kraaij, Melissa Kronenthal, Guillaume Lathoud, Mike Lincoln, Agnes Lisowska, Iain McCowan, Wilfried Post, Dennis Reidsma, Pierre Wellner, "The AMI Meeting Corpus: A Pre-announcement", In: Renals, S., Bengio, S. (eds) Machine Learning for Multimodal Interaction. MLMI 2005. Lecture Notes in Computer Science, vol 3869. Springer, Berlin, Heidelberg, 2006.</a>

<a href="https://arxiv.org/abs/2110.10739">[2] Aswin Sivaraman, Scott Wisdom, Hakan Erdogan, John R. Hershey, "Adapting Speech Separation to Real-World Meetings Using Mixture Invariant Training", Proc. IEEE International Conference on Audio, Speech, and Signal Processing (ICASSP), 2022.</a>

<a href="https://arxiv.org/abs/2006.12701">[3] Scott Wisdom, Efthymios Tzinis, Hakan Erdogan, Ron J. Weiss, Kevin Wilson, John R. Hershey, "Unsupervised Sound Separation Using Mixture Invariant Training", Advances in Neural Information Processing Systems, 33, pp. 3846--3857, 2020.</a>
