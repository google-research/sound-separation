from pathlib import Path

import tensorflow.compat.v1 as tf

from paths import ROOT_PATH

tf.disable_v2_behavior() # Match the script's environment

# Define the directory to save the checkpoint files
checkpoints_path: Path = ROOT_PATH / 'models' / 'xprize_mixit' / 'checkpoints'

meta_graph_path = checkpoints_path / 'inference.meta'

with tf.Session() as sess:
    print(f"Loading meta graph from: {meta_graph_path}")
    saver = tf.train.import_meta_graph(meta_graph_path)
    graph = tf.get_default_graph()

    print("\nPossible input placeholders (look for one expecting audio shape like [?, 1, ?]):")
    for op in graph.get_operations():
        if 'input' in op.name:
            print(f"Name: {op.name}")

        # print(f'  Operation:', op)
        if str(op.type) in ('AssignVariableOp', 'Assert', 'NoOp', 'ResourceApplyAdam', 'AssignAddVariableOp' 'SaveV2'):
            continue

        if len(op.outputs) == 0:
            continue

        for output in op.outputs:
            # if the rank of the shape is <unknown>, skip it
            if output.shape.ndims is None:
                continue

            for dimension in output.shape:
                if dimension == 144000:
                    print(
                        f"  Operation type: {op.type}, "
                        f"Outputs: {len(op.outputs)}, "
                        f"Name: {op.name}, "
                        f"DType: {output.dtype}, "
                        f"Shape: {output.shape}"
                    )

            # You might need to access the tensor name directly:
            # print(f"  Tensor Name: {op.outputs[0].name}")
