from os import path

import tensorflow.compat.v1 as tf

from models.tools import inference


def separate_channels_using_mixit(
    *,
    checkpoint: str,
    input_channels: int,
    input_file: str,
    input_tensor: str,
    model_directory: str,
    channel_count: int,
    output_channels: int,
    output_file: str,
    output_tensor: str,
    scale_input: bool,
    write_outputs_separately: bool,
):
    tf.disable_v2_behavior()
    meta_graph_filename = path.join(model_directory, 'inference.meta')
    tf.logging.info('Importing meta graph: %s', meta_graph_filename)
    with tf.Graph().as_default() as g:
        saver = tf.train.import_meta_graph(meta_graph_filename)
        meta_graph_def = g.as_graph_def()
    tf.reset_default_graph()
    input_wav, sample_rate = inference.read_wav_file(
        input_file, input_channels, scale_input)
    input_wav = tf.transpose(input_wav)
    input_wav = tf.expand_dims(input_wav, axis=0)  # shape: [1, mics, samples]
    output_wav, = tf.import_graph_def(
        meta_graph_def,
        name='',
        input_map={input_tensor: input_wav},
        return_elements=[output_tensor])
    # output_wav = tf.squeeze(output_wav, 0)  # shape: [sources, samples]
    # output_wav = tf.transpose(output_wav)
    # if output_channels > 0:
    #     output_wav = output_wav[:, :output_channels]
    # write_output_ops = inference.write_wav_file(
    #     output_file, output_wav, sample_rate=sample_rate,
    #     num_channels=channel_count,
    #     output_channels=output_channels,
    #     write_outputs_separately=write_outputs_separately,
    #     channel_name='channel_',
    # )
    if not checkpoint:
        checkpoint = tf.train.latest_checkpoint(model_directory)
    with tf.Session() as session:
        tf.logging.info('Restoring from checkpoint: %s', checkpoint)
        saver.restore(session, checkpoint)
        session.run(write_output_ops)
