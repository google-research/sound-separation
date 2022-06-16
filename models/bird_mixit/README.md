# Models for Unsupervised Sound Separation of Bird Calls Using Mixture Invariant Training.

These are instructions for using models trained on environmental recordings of bird calls with mixture invariant training (MixIT) [1], as described in [2].

If you find this code useful, please cite [1] and [2].

## Model checkpoints

Two model checkpoints, one with 4 output sources and one with 8 output sources, are available on Google Cloud. These models assume input audio sampled at 22.05 kHz. The models can be downloaded using the following command, which will copy the model checkpoint files to the current folder:

```
gsutil -m cp -r gs://gresearch/sound_separation/bird_mixit_model_checkpoints .
```

This model checkpoint is licensed under <a href="https://www.apache.org/licenses/LICENSE-2.0.txt">the Apache 2.0 license</a>.


## Install TensorFlow
Follow the instructions
<a href="https://www.tensorflow.org/install">here</a>.


## Run the model on a wav file.

Once you have installed TensorFlow, you can run the 4-output model on a wav file using the following:

```
python3 ../tools/process_wav.py \
--model_dir bird_mixit_model_checkpoints/output_sources4 \
--checkpoint bird_mixit_model_checkpoints/output_sources4/model.ckpt-3223090 \
--num_sources 4 \
--input <input name>.wav \
--output <output_name>.wav
```
which will result in 4 wav files `<output_name>_source0.wav`, ... , `<output_name>_source3.wav`.

The 8-output model can be run using the following:

```
python3 ../tools/process_wav.py \
--model_dir bird_mixit_model_checkpoints/output_sources8 \
--checkpoint bird_mixit_model_checkpoints/output_sources8/model.ckpt-2178900 \
--num_sources 8 \
--input <input name>.wav \
--output <output_name>.wav
```
which will result in 8 wav files `<output_name>_source0.wav`, ... , `<output_name>_source7.wav`.

## References

<a href="https://arxiv.org/pdf/2006.12701.pdf">[1] Scott Wisdom, Efthymios Tzinis, Hakan Erdogan, Ron J. Weiss, Kevin Wilson, John R. Hershey, "Unsupervised Sound Separation Using Mixture Invariant Training", Advances in Neural Information Processing Systems, 2020.</a>

<a href="https://arxiv.org/pdf/2110.03209.pdf">[2] Tom Denton, Scott Wisdom, John R. Hershey, "Improving Bird Classification with Unsupervised Sound Separation", Proc. IEEE International Conference on Audio, Speech, and Signal Processing (ICASSP), 2022.</a>
