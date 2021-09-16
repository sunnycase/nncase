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
#include <nncase/ir/tensors/cast.h>

using namespace nncase;
using namespace nncase::ir;
using namespace nncase::ir::tensors;

cast_node::cast_node(datatype_t new_type) : new_type_(new_type) {
    add_parameter("input");
}

cast::cast(datatype_t new_type) : object_t(std::in_place, new_type) {}

type cast_node::infer_invoke_result_type(type_infer_context &context) {
    CHECK_ARGUMENT_AS_TENSOR(input);
    return tensor_type(new_type(), input_t->shape());
}
