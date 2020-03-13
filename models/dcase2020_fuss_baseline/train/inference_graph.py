# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Write an inference graph for separation models.

Example usage, using tf.estimator compatible functions:

def input_fn(params):
  ...

def model_fn(features, labels, mode, params):
  ...

inference_graph.write_inference_graph(model_fn, input_fn, params)
"""

import copy
import os

import tensorflow.compat.v1 as tf


def write(model_fn, input_fn, params, directory):
  """Writes an inference graph."""
  input_fn_params = copy.deepcopy(params)
  input_fn_params['inference'] = True
  input_fn_params['batch_size'] = 1

  model_fn_params = copy.deepcopy(params)
  model_fn_params['batch_size'] = 1

  with tf.Graph().as_default() as graph:
    features = input_fn(input_fn_params)
    model_fn(features=features,
             labels=None,
             mode=tf.estimator.ModeKeys.PREDICT,
             params=model_fn_params)

    tf.train.Saver()
    graph_def = graph.as_graph_def(add_shapes=True)
    tf.train.write_graph(graph_def, directory, 'inference.pbtxt')
    meta_graph_name = os.path.join(directory, 'inference.meta')
    tf.train.export_meta_graph(filename=meta_graph_name)
