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

import os
import pytest
from huggingface_test_runner import HuggingfaceTestRunner, download_from_huggingface
from transformers import AutoModelForCausalLM, AutoTokenizer


def test_qwen3(request):
    num_layers = int(os.getenv("NNCASE_QWEN3_NUM_LAYERS", "-1"))
    cfg = f"""
    [compile_opt]
    dump_ir = true
    shape_bucket_enable = false
    shape_bucket_range_info = {{ }}
    shape_bucket_segments_count = 0
    shape_bucket_segments = {{ }}
    shape_bucket_fix_var_map = {{ "sequence_length"=1 }}
    
    [huggingface_options]
    output_logits = true
    output_hidden_states = false
    num_layers = {num_layers}
    tensor_type = "bfloat16"

    [paged_attention_config]
    kv_type = "bfloat16"
    key_lanes = [8]
    value_lanes = [8]

    [generator]
    [generator.inputs]
    method = 'text'
    number = 1
    batch = 1

    [generator.inputs.text]
    args = 'tests/importer/huggingface_/prompt.txt'
    sequence_length = 1

    [generator.calibs]
    method = 'text'
    number = 1
    batch = 1

    [generator.calibs.text]
    args = 'tests/importer/huggingface_/prompt.txt'
    sequence_length = 1
    """
    runner = HuggingfaceTestRunner(request.node.name, overwrite_configs=cfg)

    model_name = "Qwen/Qwen3-0.6B"

    if os.path.exists(os.path.join(os.path.dirname(__file__), model_name)):
        model_file = os.path.join(os.path.dirname(__file__), model_name)
    else:
        model_file = download_from_huggingface(
            AutoModelForCausalLM, AutoTokenizer, model_name, need_save=True)

    runner.run(model_file)


if __name__ == "__main__":
    pytest.main(['-vv', __file__])
