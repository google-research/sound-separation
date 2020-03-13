# README for the DCASE2020 FUSS baseline model.
This model is the source separation baseline for DCASE2020 Challenge Task 4:
Sound Event Detection and Separation in Domestic Environments (SEDSDE).

You can use the code in this directory to train and evaluate the baseline model
from scratch on FUSS data. If you find this code or model to be useful, please
cite [1].

This baseline model consists of a TDCN++ masking network [2, 3] using STFT
analysis/synthesis and weighted mixture consistency [4], where the weights are
predicted by the network, with one scalar per source. The training loss is
thresholded negative signal-to-noise ratio (SNR).

The model is able to handle
variable number sources by using different loss functions for active and
inactive reference sources. For active reference sources (i.e. non-zero
reference source signals), the threshold for negative SNR is 30 dB, equivalent
to the error power being below the reference
power by 30 dB. For non-active reference sources (i.e. all-zero reference
source signals), the threshold is 20 dB measured relative to the mixture power,
which means gradients are clipped when the error power is 20 dB below the mixture power.

## Install TensorFlow
Follow the instructions
<a href="https://www.tensorflow.org/install">here</a>.

## Evaluate a model
You can download and evaluate the pretrained baseline model using the following:

```
  ./run_baseline_model_evaluate.sh
```

To evaluate a model you have trained yourself, e.g. with ```run_baseline_model_train.sh```, see ```./run_baseline_model_evaluate.sh```
for an example of calling ```evaluate.py```. Please only use the official eval
set of FUSS to report results, which is defined by:

```ssdata_reverb/eval_example_list.txt```

## Train a model
A pretrained baseline model is provided, but you can also train a model yourself using the following:

```
  ./run_baseline_model_train.sh
```

Training and validation performance can be visualized during training using the
following:

```
  tensorboard --logdir=<your_model_directory>
```

The default model directory is set to

```
${ROOT_DIR}/dcase2020_fuss/baseline_train/${DATE}
```

where ```ROOT_DIR``` is defined in ```setup.sh```, and ```DATE``` has the
following format:

```
<year>-<month>-<day>_<hour>-<minute>-<second>
```

## References
[1] Scott Wisdom, Hakan Erdogan, Daniel P. W. Ellis, Romain Serizel, Nicolas Turpault, Eduardo Fonseca, Justin Salamon, Prem Seetharaman, John R. Hershey,
"What's All the FUSS About Free Universal Sound Separation Data?", 2020, in preparation.

[2] Ilya Kavalerov, Scott Wisdom, Hakan Erdogan, Brian Patton, Kevin Wilson, Jonathan Le Roux, and John R. Hershey. "Universal Sound Separation." IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA), pp. 175-179. New Paltz, NY, USA, 2019.

[3] Efthymios Tzinis, Scott Wisdom, John R. Hershey, Aren Jansen, and Daniel P. W. Ellis. "Improving Universal Sound Separation Using Sound Classification." IEEE International Conference on Acoustics Speech and Signal Processing (ICASSP), 2020.

[4] Scott Wisdom, John R. Hershey, Kevin Wilson, Jeremy Thorpe, Michael Chinen, Brian Patton, Rif A. Saurous. "Differentiable Consistency Constraints for Improved Deep Speech Enhancement." IEEE International Conference on Acoustics Speech and Signal Processing (ICASSP), 2019.
