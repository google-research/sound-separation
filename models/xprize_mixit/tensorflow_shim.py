import tensorflow

def get_determinism() -> None:
    tensorflow.keras.utils.set_random_seed(42)
    tensorflow.config.experimental.enable_op_determinism()
