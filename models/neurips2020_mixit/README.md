# Code for Unsupervised Sound Separation Using Mixture Invariant Training.

You can use the code in this directory to train the model from [1] using mixture
invariant training (MixIT) from scratch on the Free Universal Sound Separation
(FUSS) dataset [2]. This is equivalent to the model in the second-to-last row
of Table 1 in [1].

If you find this code useful, please cite [1], and [2] for FUSS.

## Install TensorFlow
Follow the instructions
<a href="https://www.tensorflow.org/install">here</a>.


## Train the model
You can also train a model yourself using the following:

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
<a href="https://arxiv.org/pdf/2006.12701.pdf">
  [1] Scott Wisdom, Efthymios Tzinis, Hakan Erdogan, Ron J. Weiss, Kevin Wilson, John R. Hershey, Unsupervised Sound Separation Using Mixture Invariant Training", Advances in Neural Information Processing Systems, 2020.
</a>

<a href="https://arxiv.org/pdf/2011.00803.pdf">
[2] Scott Wisdom, Hakan Erdogan, Daniel P. W. Ellis, Romain Serizel, Nicolas Turpault, Eduardo Fonseca, Justin Salamon, Prem Seetharaman, John R. Hershey,
"What's All the FUSS About Free Universal Sound Separation Data?", 2020, in preparation.
</a>
