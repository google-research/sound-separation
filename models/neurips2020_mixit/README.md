# Code for Unsupervised Sound Separation Using Mixture Invariant Training.

You can use the code in this directory to train the model from [1] using mixture
invariant training (MixIT) from scratch on the Free Universal Sound Separation
(FUSS) dataset [2]. This is equivalent to the model in the second-to-last row
of Table 1 in [1].

If you find this code useful, please cite [1], and [2] for FUSS.

## YFCC100m data recipe

Though training on YFCC100m is not yet supported with this code, we have released <a href="https://github.com/google-research/sound-separation/blob/master/datasets/yfcc100m/README.md">clip lists</a> that replicate the train, validation, and test clips, as well as the mixtures of mixtures (MoM) test set in [1].

## YFCC100m model checkpoint

A model checkpoint is available on Google Cloud that has been trained on YFCC100m. It can be downloaded using the following command which will copy the model checkpoint files to the current folder:

```
gsutil -m cp -r gs://gresearch/sound_separation/yfcc100m_mixit_model_checkpoint .
```

This model checkpoint is licensed under <a href="https://www.apache.org/licenses/LICENSE-2.0.txt">the Apache 2.0 license</a>.


## Install TensorFlow
Follow the instructions
<a href="https://www.tensorflow.org/install">here</a>.


## Train the model on FUSS

Once you have installed TensorFlow and downloaded the FUSS data (<a href="https://github.com/google-research/sound-separation/blob/master/datasets/fuss/README.md">instructions here</a>), you can train a model yourself using the following:

```
  ./run_train_model_on_fuss.sh
```

Training and validation performance can be visualized during training using the
following:

```
  tensorboard --logdir=<your_model_directory>
```

The default model directory is set to

```
${ROOT_DIR}/mixit/fuss/${DATE}
```

where ```ROOT_DIR``` is defined in ```setup.sh```, and ```DATE``` has the
following format:

```
<year>-<month>-<day>_<hour>-<minute>-<second>
```

## References

<a href="https://arxiv.org/pdf/2006.12701.pdf">[1] Scott Wisdom, Efthymios Tzinis, Hakan Erdogan, Ron J. Weiss, Kevin Wilson, John R. Hershey, "Unsupervised Sound Separation Using Mixture Invariant Training", Advances in Neural Information Processing Systems, 2020.</a>

<a href="https://arxiv.org/pdf/2011.00803.pdf">[2] Scott Wisdom, Hakan Erdogan, Daniel P. W. Ellis, Romain Serizel, Nicolas Turpault, Eduardo Fonseca, Justin Salamon, Prem Seetharaman, John R. Hershey, "What's All the FUSS About Free Universal Sound Separation Data?", ICASSP 2021.</a>
