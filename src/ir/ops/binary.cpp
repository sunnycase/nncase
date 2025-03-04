/* Copyright 2019-2021 Canaan Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <nncase/ir/op_utils.h>
#include <nncase/ir/ops/binary.h>
#include <xtensor/xarray.hpp>

using namespace nncase;
using namespace nncase::ir;

binary::binary(binary_op_t binary_op, datatype_t input_type, shape_t input_a_shape, shape_t input_b_shape, value_range<float> input_fused_activation)
    : binary_op_(binary_op), fused_activation_(input_fused_activation)
{
    add_input("input_a", input_type, input_a_shape);
    add_input("input_b", input_type, input_b_shape);
    add_output("output", input_type, get_binary_output_shape(input_a_shape, input_b_shape));
}

bool binary::properties_equal(node &other) const
{
    auto &r = static_cast<binary &>(other);
    return binary_op() == r.binary_op() && fused_activation() == r.fused_activation();
}
