# Copyright 2019-2021 Canaan Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# pylint: disable=invalid-name, unused-argument, import-outside-toplevel

import pytest
import onnx
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto, numpy_helper
from onnx_test_runner import OnnxTestRunner


def _make_module(in_shape, scale, zp, output_type):
    inputs = []
    outputs = []
    initializers = []
    attributes_dict = {}

    # input
    input = helper.make_tensor_value_info('input', TensorProto.FLOAT, in_shape)
    inputs.append('input')

    # scale
    scale = helper.make_tensor(
        'scale',
        TensorProto.FLOAT,
        dims=[len(scale)],
        vals=scale
    )
    inputs.append('scale')
    initializers.append(scale)

    # zero point
    if zp is not None:
        zero_point = helper.make_tensor(
            'zero_point',
            output_type,
            dims=[len(zp)],
            vals=zp
        )
        inputs.append('zero_point')
        initializers.append(zero_point)

    # output
    output = helper.make_tensor_value_info('output', output_type, in_shape)
    outputs.append('output')

    node = onnx.helper.make_node(
        'QuantizeLinear',
        inputs=inputs,
        outputs=outputs,
        **attributes_dict
    )

    nodes = []
    nodes.append(node)

    graph_def = helper.make_graph(
        nodes,
        'test-model',
        [input],
        [output],
        initializer=initializers)

    model_def = helper.make_model(graph_def, producer_name='onnx')

    return model_def


in_shapes = [
    [1, 3, 16, 16]
]

scales = [
    [0.02],
]

zero_points = [
    [100],
]

output_types = [
    TensorProto.UINT8,
    TensorProto.INT8
]

@pytest.mark.parametrize('in_shape', in_shapes)
@pytest.mark.parametrize('scale', scales)
@pytest.mark.parametrize('zero_point', zero_points)
@pytest.mark.parametrize('output_type', output_types)
def test_quantizelinear(in_shape, scale, zero_point, output_type, request):
    model_def = _make_module(in_shape, scale, zero_point, output_type)

    runner = OnnxTestRunner(request.node.name)
    model_file = runner.from_onnx_helper(model_def)
    runner.run(model_file)


if __name__ == "__main__":
    pytest.main(['-vv', 'test_quantizelinear.py'])
