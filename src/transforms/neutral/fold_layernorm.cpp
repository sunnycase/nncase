/* Copyright 2020 Canaan Inc.
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
#include <nncase/ir/ops/binary.h>
#include <nncase/ir/ops/bitcast.h>
#include <nncase/ir/ops/constant.h>
#include <nncase/ir/ops/layernorm.h>
#include <nncase/ir/ops/reduce.h>
#include <nncase/ir/ops/unary.h>
#include <nncase/ir/visitor.h>
#include <nncase/runtime/datatypes.h>
#include <nncase/transforms/neutral/fold_layernorm.h>

using namespace nncase;
using namespace nncase::ir;
using namespace nncase::ir::transforms;

bool fold_layernorm_pattern1_transform::on_try_match(node &node, transform_context &context)
{
    reduce *rd1 = nullptr, *rd2 = nullptr;
    binary *sub = nullptr, *pow = nullptr, *add_eps = nullptr, *div = nullptr, *mul = nullptr, *add_beta = nullptr;
    unary *sqrt = nullptr;
    bitcast *rshape1 = nullptr, *rshape2 = nullptr;

    if ((add_beta = node_cast<binary>(node)) and add_beta->binary_op() == binary_op_t::binary_add
        and (mul = try_get_direct_parent<binary>(*add_beta)) and mul->binary_op() == binary_op_t::binary_mul
        and (rshape2 = try_get_direct_parent<bitcast>(*mul))
        and (div = try_get_direct_parent<binary>(*rshape2)) and div->binary_op() == binary_op_t::binary_div
        and (sqrt = try_get_direct_parent<unary>(*div)) and sqrt->unary_op() == unary_op_t::unary_sqrt
        and (add_eps = try_get_direct_parent<binary>(*sqrt)) and add_eps->binary_op() == binary_op_t::binary_add
        and (rd2 = try_get_direct_parent<reduce>(*add_eps)) and rd2->reduce_op() == reduce_op_t::reduce_mean
        and (pow = try_get_direct_parent<binary>(*rd2)) and pow->binary_op() == binary_op_t::binary_pow
        and (sub = try_get_direct_parent<binary>(*pow)) and sub->binary_op() == binary_op_t::binary_sub
        and (rd1 = try_get_direct_parent<reduce>(*sub)) and rd1->reduce_op() == reduce_op_t::reduce_mean
        and (rshape1 = try_get_direct_parent<bitcast>(*rd1))
        and (sub->input_a().connection() == rd1->input().connection() or sub->input_b().connection() == rd1->input().connection())
        and try_get_direct_parent<binary>(*div) == sub)
    {
        context.inputs.emplace_back(&rshape1->input());
        context.outputs.emplace_back(&add_beta->output());

        context.matched_nodes.emplace_back(rshape1);
        context.matched_nodes.emplace_back(rd1);
        context.matched_nodes.emplace_back(sub);
        context.matched_nodes.emplace_back(pow);
        context.matched_nodes.emplace_back(rd2);
        context.matched_nodes.emplace_back(add_eps);
        context.matched_nodes.emplace_back(sqrt);
        context.matched_nodes.emplace_back(div);
        context.matched_nodes.emplace_back(rshape2);
        context.matched_nodes.emplace_back(mul);
        context.matched_nodes.emplace_back(add_beta);

        return true;
    }

    return false;
}

void fold_layernorm_pattern1_transform::process(transform_context &context)
{
    auto &output = *context.inputs[0]->connection();
    auto inputs = context.outputs[0]->connections();

    auto eps = node_cast<constant>(context.matched_nodes[5]->input_at(1).connection()->owner());
    auto gamma = node_cast<constant>(context.matched_nodes[9]->input_at(1).connection()->owner());
    auto beta = node_cast<constant>(context.matched_nodes[10]->input_at(1).connection()->owner());

    auto axis = output.shape().size() - gamma->output().shape().size();
    auto ln = context.graph.emplace<layernorm>(output.type(), output.shape(), axis, *reinterpret_cast<const float *>(eps->data().data()));
    ln->name(output.name() + "/layernorm");

    ln->input().connect(output);
    ln->scale().connect(gamma->output());
    ln->bias().connect(beta->output());

    for (auto &in : dup(inputs))
        in->connect(ln->output());
}

bool fold_layernorm_pattern2_transform::on_try_match(node &node, transform_context &context)
{
    reduce *rd1 = nullptr, *rd2 = nullptr;
    binary *sub = nullptr, *pow = nullptr, *add_eps = nullptr, *div = nullptr, *mul = nullptr, *add_beta = nullptr;
    unary *sqrt = nullptr;

    if ((add_beta = node_cast<binary>(node)) and add_beta->binary_op() == binary_op_t::binary_add
        and (mul = try_get_direct_parent<binary>(*add_beta)) and mul->binary_op() == binary_op_t::binary_mul
        and (div = try_get_direct_parent<binary>(*mul)) and div->binary_op() == binary_op_t::binary_div
        and (sqrt = try_get_direct_parent<unary>(*div)) and sqrt->unary_op() == unary_op_t::unary_sqrt
        and (add_eps = try_get_direct_parent<binary>(*sqrt)) and add_eps->binary_op() == binary_op_t::binary_add
        and (rd2 = try_get_direct_parent<reduce>(*add_eps)) and rd2->reduce_op() == reduce_op_t::reduce_mean
        and (pow = try_get_direct_parent<binary>(*rd2)) and pow->binary_op() == binary_op_t::binary_pow
        and ((sub = try_get_direct_parent<binary>(*pow, 0)) or (sub = try_get_direct_parent<binary>(*pow, 1))) and sub->binary_op() == binary_op_t::binary_sub
        and (rd1 = try_get_direct_parent<reduce>(*sub)) and rd1->reduce_op() == reduce_op_t::reduce_mean
        and (sub->input_a().connection() == rd1->input().connection() or sub->input_b().connection() == rd1->input().connection())
        and try_get_direct_parent<binary>(*div) == sub)
    {
        context.inputs.emplace_back(&rd1->input());
        context.outputs.emplace_back(&add_beta->output());

        context.matched_nodes.emplace_back(rd1);
        context.matched_nodes.emplace_back(sub);
        context.matched_nodes.emplace_back(pow);
        context.matched_nodes.emplace_back(rd2);
        context.matched_nodes.emplace_back(add_eps);
        context.matched_nodes.emplace_back(sqrt);
        context.matched_nodes.emplace_back(div);
        context.matched_nodes.emplace_back(mul);
        context.matched_nodes.emplace_back(add_beta);

        return true;
    }

    return false;
}

void fold_layernorm_pattern2_transform::process(transform_context &context)
{
    auto &output = *context.inputs[0]->connection();
    auto inputs = context.outputs[0]->connections();

    auto eps = node_cast<constant>(context.matched_nodes[4]->input_at(1).connection()->owner());
    auto gamma = node_cast<constant>(context.matched_nodes[7]->input_at(1).connection()->owner());
    auto beta = node_cast<constant>(context.matched_nodes[8]->input_at(1).connection()->owner());

    auto axis = output.shape().size() - gamma->output().shape().size();
    auto ln = context.graph.emplace<layernorm>(output.type(), output.shape(), axis, *reinterpret_cast<const float *>(eps->data().data()));
    ln->name(output.name() + "/layernorm");

    ln->input().connect(output);
    ln->scale().connect(gamma->output());
    ln->bias().connect(beta->output());

    for (auto &in : dup(inputs))
        in->connect(ln->output());
}