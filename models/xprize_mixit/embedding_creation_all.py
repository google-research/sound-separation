import numpy
from pathlib import Path

import tensorflow.compat.v1 as tf
from tensorflow import float32
from tensorflow.python.framework.ops import SymbolicTensor

from models.tools import inference
from models.xprize_mixit.paths import ROOT_PATH


def store_tensor_activations(
    tensor_name: str,
    model_directory: Path = ROOT_PATH / 'models' / 'xprize_mixit' / 'checkpoints',
    input_tensor: str = 'input_audio/receiver_audio:0',
    input_file: str = str(ROOT_PATH / 'datasets' / 'xeno-canto' / 'XC771373.wav'),
    input_channels: int = 8,
    scale_input: bool = False,
    downsample_factor: int = 2,  # Added to handle exact 2:1 ratio in error
):
    """
    Store the activations of a given tensor in a file.
    Args:
        tensor_name (str): The name of the tensor to store.
    """
    print(tf.executing_eagerly())
    meta_graph_filename = model_directory / 'inference.meta'
    tf.logging.info('Importing meta graph: %s', meta_graph_filename)
    with tf.Graph().as_default() as graph:
        meta_graph_def = graph.as_graph_def()

        # Read the audio input
        input_wav, sample_rate = inference.read_wav_file(
            input_file, input_channels, scale_input,
        )
        print(f"Original audio shape: {input_wav.shape}, Sample rate: {sample_rate}")

        # Downsample the audio if needed to match model expectations
        if downsample_factor > 1:
            # Taking every nth sample (simple downsampling)
            # This preserves channels but reduces time dimension
            with tf.Session() as sess:
                input_wav_val = sess.run(input_wav)
                input_wav = tf.constant(input_wav_val[:, ::downsample_factor])
                print(f"Downsampled audio shape: {input_wav.shape}")

        input_wav = tf.transpose(input_wav)
        input_wav = tf.expand_dims(input_wav, axis=0)  # shape: [1, mics, samples]
        print(f"Input tensor shape: {input_wav.shape}")

        embedding: SymbolicTensor
        embedding, = tf.import_graph_def(
            meta_graph_def,
            name='',
            input_map={input_tensor: input_wav},
            return_elements=[tensor_name]
        )
        print(f"Embedding tensor: {embedding}")
        print(f"Embedding shape: {embedding.shape}")

        with tf.Session(graph=graph) as session:
            session.run(tf.global_variables_initializer())

            try:
                embedding_np = session.run(embedding)
                print(f"Embedding numpy shape: {embedding_np.shape}")
                print(f"Total elements: {numpy.prod(embedding_np.shape)}")

                # Store the tensor value in a file
                with open(
                        ROOT_PATH / 'datasets' / 'xeno-canto' / 'tensor_activations' / f'{tensor_name}.txt',
                        'w'
                ) as f:
                    f.write(str(embedding_np))

                # if embedding_output_file:
                #     numpy.save(embedding_output_file, embedding_np)
                #     print(f"Embedding saved to {embedding_output_file}")
                #
                # if csv_output_file:
                #     numpy.savetxt(csv_output_file, embedding_np.reshape(-1), delimiter=',')
                #     print(f"Embedding saved to {csv_output_file}")

                return embedding_np

            except tf.errors.InvalidArgumentError as e:
                print(f"Error running the model: {e}")
                print("Try adjusting downsample_factor to match model expectations.")
                raise

def main():
    candidate_tensors = (
        # {
        #     'name': 'improved_tdcn/conv_block_31/tdcn_block/normact2/instance_norm/moments/mean_191:0',
        #     'shape': (1, 1, 1, 512),
        #     'dtype': float32,
        # },
        # {
        #     'name': 'improved_tdcn/conv_block_31/tdcn_block/normact2/instance_norm/moments/StopGradient_192:0',
        #     'shape': (1, 1, 1, 512),
        #     'dtype': float32,
        # },
        # {
        #     'name': 'improved_tdcn/conv_block_31/tdcn_block/normact2/instance_norm/moments/variance_195:0',
        #     'shape': (1, 1, 1, 512),
        #     'dtype': float32,
        # },
        # {
        #     'name': 'improved_tdcn/conv_block_31/tdcn_block/normact2/instance_norm/batchnorm/add_198:0',
        #     'shape': (1, 1, 1, 512),
        #     'dtype': float32,
        # },
        {
            'name': 'improved_tdcn/conv_block_31/tdcn_block/normact2/instance_norm/batchnorm/Rsqrt_198:0',
            'shape': (1, 1, 1, 512),
            'dtype': float32,
        },
        {
            'name': 'improved_tdcn/conv_block_31/tdcn_block/normact2/instance_norm/batchnorm/mul_202:0',
            'shape': (1, 1, 1, 512),
            'dtype': float32,
        },
        {
            'name': 'improved_tdcn/conv_block_31/tdcn_block/normact2/instance_norm/batchnorm/mul_2_202:0',
            'shape': (1, 1, 1, 512),
            'dtype': float32,
        },
        {
            'name': 'improved_tdcn/conv_block_31/tdcn_block/normact2/instance_norm/batchnorm/sub_204:0',
            'shape': (1, 1, 1, 512),
            'dtype': float32,
        },
        {
            'name': 'improved_tdcn/conv_block_31/tdcn_block/normact2/instance_norm/moments/mean_191:0',
            'shape': (1, 1, 1, 512),
            'dtype': float32,
        },
        {
            'name': 'improved_tdcn/conv_block_31/tdcn_block/normact2/instance_norm/moments/StopGradient_192:0',
            'shape': (1, 1, 1, 512),
            'dtype': float32,
        },
        {
            'name': 'improved_tdcn/conv_block_31/tdcn_block/normact2/instance_norm/moments/variance_195:0',
            'shape': (1, 1, 1, 512),
            'dtype': float32,
        },
        {
            'name': 'improved_tdcn/conv_block_31/tdcn_block/normact2/instance_norm/batchnorm/add_198:0',
            'shape': (1, 1, 1, 512),
            'dtype': float32,
        },
        {
            'name': 'improved_tdcn/conv_block_31/tdcn_block/normact2/instance_norm/batchnorm/Rsqrt_198:0',
            'shape': (1, 1, 1, 512),
            'dtype': float32,
        },
        {
            'name': 'improved_tdcn/conv_block_31/tdcn_block/normact2/instance_norm/batchnorm/mul_202:0',
            'shape': (1, 1, 1, 512),
            'dtype': float32,
        },
        {
            'name': 'improved_tdcn/conv_block_31/tdcn_block/normact2/instance_norm/batchnorm/mul_2_202:0',
            'shape': (1, 1, 1, 512),
            'dtype': float32,
        },
        {
            'name': 'improved_tdcn/conv_block_31/tdcn_block/normact2/instance_norm/batchnorm/sub_204:0',
            'shape': (1, 1, 1, 512),
            'dtype': float32,
        },
    )
    for candidate_tensor in candidate_tensors:
        store_tensor_activations(tensor_name=candidate_tensor['name'])

if __name__ == '__main__':
    main()
