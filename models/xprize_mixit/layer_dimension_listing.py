import numpy
import tensorflow.compat.v1 as tf
from tensorflow.python.framework.ops import SymbolicTensor, Operation
from models.tools import inference
from models.xprize_mixit.paths import ROOT_PATH
from os import path


def create_embeddings_using_mixit(
    *,
    checkpoint: str = None,
    input_channels: int = 8,
    input_file: str = str(ROOT_PATH / 'datasets' / 'xeno-canto' / 'XC771373.wav'),
    input_tensor: str = 'input_audio/receiver_audio:0',
    model_directory: str = ROOT_PATH / 'models' / 'xprize_mixit' / 'checkpoints',
    channel_count: int = 1,
    output_channels: int = 8,
    output_file: str = str(ROOT_PATH / 'datasets' / 'xeno-canto' / 'XC771473.wav'),
    # output_tensor: str = 'improved_tdcn/conv_block_31/tdcn_block/Add:0',
    # output_tensor: str = 'improved_tdcn/conv_block_0/tdcn_block/normact1/instance_norm/batchnorm/mul:0',
    # output_tensor: str = 'improved_tdcn/conv_block_0/tdcn_block/normact1/prelu/sub:0',
    # output_tensor: str = 'conv_encoder/dense/Relu:0',
    # output_tensor: str = 'initial_dense/dense/BiasAdd:0',
    # output_tensor: str = 'improved_tdcn/conv_block_0/tdcn_block/normact2/instance_norm/batchnorm/sub_373:0',  #  (1, 1, 1, 512)
    scale_input: bool = False,
    write_outputs_separately: bool = True,
    embedding_output_file: str = None,
    csv_output_file: str = None,
    downsample_factor: int = 2,
):
    print(tf.executing_eagerly())
    meta_graph_filename = path.join(model_directory, 'inference.meta')
    tf.logging.info('Importing meta graph: %s', meta_graph_filename)
    with tf.Graph().as_default() as g:
        meta_graph_def = g.as_graph_def()

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

        # This stupid approach is slow, but it works.
        embedding: SymbolicTensor
        embedding, = tf.import_graph_def(
            meta_graph_def,
            name='',
            input_map={input_tensor: input_wav},
            return_elements=[output_tensor]
        )
        node: Operation
        for node in meta_graph_def.node:
            # Skip nodes that are not tensors
            if node.op == '':
                print(f"Skipping node {node.name} of type {node.op}")
                continue

            try:
                embedding_candidate, = tf.import_graph_def(
                    meta_graph_def,
                    name='',
                    input_map={input_tensor: input_wav},
                    return_elements=[node.name + ':0'],
                )
                print(f"Embedding candidate tensor: {embedding_candidate}")
                print(f"Embedding candidate tensor shape: {embedding_candidate.shape}")
            except Exception as e:
                print(f"WARN_ Skipping node due to error: {e}")
                continue

        with tf.Session(graph=g) as session:
            session.run(tf.global_variables_initializer())

            try:
                embedding_np = session.run(embedding_candidate)
                print(f"Embedding numpy shape: {embedding_np.shape}")
                print(f"Total elements: {numpy.prod(embedding_np.shape)}")

                # Save as numpy array
                if embedding_output_file:
                    numpy.save(embedding_output_file, embedding_np)
                    print(f"Embedding saved to {embedding_output_file}")

                # Save as CSV
                if csv_output_file:
                    # Reshape for CSV - flatten all dimensions into a 1D array
                    flattened = embedding_np.flatten()
                    numpy.savetxt(csv_output_file, flattened, delimiter=',')
                    print(f"Embedding saved as CSV to {csv_output_file}")

                return embedding_np

            except tf.errors.InvalidArgumentError as e:
                print(f"Error running the model: {e}")
                print("Try adjusting downsample_factor to match model expectations.")
                raise


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Create embeddings using MixIT model')
    parser.add_argument('--input_filepath', default=str((ROOT_PATH / 'datasets' / 'xeno-canto' / 'XC771373.wav')),
                        help='Input audio file path')
    parser.add_argument('--model_name', default='xprize_mixit', help='Model name')
    parser.add_argument('--embedding_output_file', help='Output file path for embedding (.npy)')
    parser.add_argument('--csv_output_file', help='Output file path for embedding as CSV (.csv)')
    parser.add_argument('--downsample_factor', type=int, default=2,
                        help='Factor to downsample audio (2 = half the samples)')

    args = parser.parse_args()

    # Default embedding filename if not provided
    if not args.embedding_output_file:
        args.embedding_output_file = args.input_filepath.replace('.wav', '.npy')

    # Default CSV filename if not provided but CSV output is requested
    if args.csv_output_file is None:
        args.csv_output_file = args.input_filepath.replace('.wav', '.csv')

    create_embeddings_using_mixit(
        input_file=args.input_filepath,
        embedding_output_file=args.embedding_output_file,
        csv_output_file=args.csv_output_file,
        downsample_factor=args.downsample_factor
    )
