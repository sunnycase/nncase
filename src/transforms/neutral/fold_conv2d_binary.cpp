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
#include <nncase/ir/ops/binary.h>
#include <nncase/ir/ops/concat.h>
#include <nncase/ir/ops/constant.h>
#include <nncase/ir/ops/conv2d.h>
#include <nncase/ir/visitor.h>
#include <nncase/transforms/neutral/fold_conv2d_binary.h>

using namespace nncase;
using namespace nncase::ir;
using namespace nncase::ir::transforms;

bool fold_conv2d_biasadd_transform::on_try_match(node &node, transform_context &context)
{
    conv2d *conv = nullptr;
    binary *add = nullptr;
    constant *bias_constant = nullptr;
    constant *add_constant = nullptr;
    if ((add = node_cast<binary>(node))
        && (add->binary_op() == binary_add)
        && (((conv = try_get_direct_parent<conv2d>(*add, 0)) && (add_constant = try_get_direct_parent<constant>(*add, 1))) || ((conv = try_get_direct_parent<conv2d>(*add, 1)) && (add_constant = try_get_direct_parent<constant>(*add, 0))))
        && (conv->fused_activation() == value_range<float>::full())
        && (bias_constant = node_cast<constant>(conv->bias().connection()->owner()))
        && (bias_constant->data().size() == add_constant->data().size()))
    {
        context.inputs.emplace_back(&conv->input());
        context.inputs.emplace_back(&conv->weights());
        context.outputs.emplace_back(&add->output());

        context.matched_nodes.emplace_back(conv);
        context.matched_nodes.emplace_back(bias_constant);
        context.matched_nodes.emplace_back(add);
        context.matched_nodes.emplace_back(add_constant);
        return true;
    }

    return false;
}

void fold_conv2d_biasadd_transform::process(transform_context &context)
{
    auto old_conv = node_cast<conv2d>(*context.matched_nodes[0]);
    auto old_bias = node_cast<constant>(*context.matched_nodes[1]);
    const float *p = reinterpret_cast<const float *>(old_bias->data().data());

    auto add = node_cast<binary>(*context.matched_nodes[2]);
    auto add_bias = node_cast<constant>(*context.matched_nodes[3]);
    const float *q = reinterpret_cast<const float *>(add_bias->data().data());

    // update bias
    size_t channels = old_conv->output_channels();
    std::vector<float> bias(channels, 0.f);
    for (size_t i = 0; i < channels; i++)
    {
        bias[i] = p[i] + q[i];
    }

    // create new bias
    auto new_bias = context.graph.emplace<constant>(dt_float32, shape_t { channels }, std::span(bias));
    new_bias->name(old_bias->name());

    // update act
    auto act = old_conv->fused_activation();
    auto add_act = add->fused_activation();
    act.min = std::max(act.min, add_act.min);
    act.max = std::min(act.max, add_act.max);

    // create new conv2d
    auto new_conv = context.graph.emplace<conv2d>(old_conv->input().shape(), old_conv->weights().shape(), old_conv->groups(), old_conv->padding_h(), old_conv->padding_w(), old_conv->stride_h(), old_conv->stride_w(), old_conv->dilation_h(), old_conv->dilation_w(), act);

    new_conv->input().connect(*context.inputs[0]->connection());
    new_conv->weights().connect(*context.inputs[1]->connection());
    new_conv->bias().connect(new_bias->output());
    new_conv->name(old_conv->name());

    auto inputs = context.outputs[0]->connections();
    for (auto &in : dup(inputs))
        in->connect(new_conv->output());
}

bool fold_conv2d_mul_transform::on_try_match(node &node, transform_context &context)
{
    conv2d *conv = nullptr;
    binary *mul = nullptr;
    constant *c = nullptr;
    constant *weights = nullptr;
    constant *bias = nullptr;
    if ((conv = node_cast<conv2d>(node))
        && (conv->fused_activation() == value_range<float>::full())
        && (weights = try_get_direct_parent<constant>(*conv, 1))
        && (bias = try_get_direct_parent<constant>(*conv, 2))
        && (mul = try_get_direct_child<binary>(*conv))
        && (mul->binary_op() == binary_mul)
        && (c = try_get_direct_parent<constant>(*mul)))
    {
        auto &c_shape = c->output().shape();
        if ((c_shape.size() == 3
                && c_shape == shape_t { (size_t)conv->output_channels(), 1, 1 })
            || (c_shape.size() == 4
                && c_shape == shape_t { 1, (size_t)conv->output_channels(), 1, 1 }))
        {
            context.inputs.emplace_back(&conv->input());
            context.outputs.emplace_back(&mul->output());

            context.matched_nodes.emplace_back(conv);
            context.matched_nodes.emplace_back(mul);
            context.matched_nodes.emplace_back(c);
            context.matched_nodes.emplace_back(weights);
            context.matched_nodes.emplace_back(bias);
            return true;
        }
    }

    return false;
}

void fold_conv2d_mul_transform::process(transform_context &context)
{
    auto old_conv = node_cast<conv2d>(*context.matched_nodes[0]);
    auto old_mul = node_cast<binary>(*context.matched_nodes[1]);
    auto old_c_node = node_cast<constant>(*context.matched_nodes[2]);
    auto old_weights_node = node_cast<constant>(*context.matched_nodes[3]);
    auto old_bias_node = node_cast<constant>(*context.matched_nodes[4]);

    auto old_c = as_span<const float>(old_c_node->data());
    auto old_weights = as_span<const float>(old_weights_node->data());
    auto old_bias = as_span<const float>(old_bias_node->data());

    // update weights & bias
    size_t channels = old_conv->output_channels();
    size_t weights_per_channel = old_weights.size() / channels;
    std::vector<float> weights(old_weights.begin(), old_weights.end());
    std::vector<float> bias(old_bias.begin(), old_bias.end());
    for (size_t i = 0; i < channels; i++)
    {
        auto scale = old_c[i];
        for (size_t j = 0; j < weights_per_channel; j++)
            weights[i * weights_per_channel + j] *= scale;
        bias[i] *= scale;
    }

    // create new weights & bias
    auto new_weights = context.graph.emplace<constant>(dt_float32, old_weights_node->output().shape(), weights);
    new_weights->name(old_weights_node->name());
    auto new_bias = context.graph.emplace<constant>(dt_float32, old_bias_node->output().shape(), bias);
    new_bias->name(old_bias_node->name());

    // create new conv2d
    auto new_conv = context.graph.emplace<conv2d>(old_conv->input().shape(), old_conv->weights().shape(), old_conv->groups(), old_conv->padding_h(), old_conv->padding_w(), old_conv->stride_h(), old_conv->stride_w(), old_conv->dilation_h(), old_conv->dilation_w(), old_mul->fused_activation());

    new_conv->input().connect(*context.inputs[0]->connection());
    new_conv->weights().connect(new_weights->output());
    new_conv->bias().connect(new_bias->output());
    new_conv->name(old_conv->name());

    auto inputs = context.outputs[0]->connections();
    for (auto &in : dup(inputs))
        in->connect(new_conv->output());
}

bool fold_two_conv2d_add_transform::on_try_match(node &node, transform_context &context)
{
    conv2d *conv1, *conv2;
    constant *weights1, *weights2;
    constant *bias1, *bias2;
    binary *add;
    if ((add = node_cast<binary>(node))
        && (add->binary_op() == binary_add)
        && (conv1 = try_get_direct_parent<conv2d>(*add, 0))
        && (conv2 = try_get_direct_parent<conv2d>(*add, 1))
        && (conv1->fused_activation() == value_range<float>::full())
        && (conv2->fused_activation() == value_range<float>::full())
        && (conv1->padding_h() == conv2->padding_h())
        && (conv1->padding_w() == conv2->padding_w())
        && (conv1->dilation_h() == conv2->dilation_h())
        && (conv1->dilation_w() == conv2->dilation_w())
        && (conv1->stride_h() == conv2->stride_h())
        && (conv1->stride_w() == conv2->stride_w())
        && (conv1->groups() == conv2->groups())
        && (conv1->groups() == 1)
        && (conv1->output_channels() == conv2->output_channels())
        && (conv1->output_channels() <= 256)
        && (weights1 = try_get_direct_parent<constant>(*conv1, 1))
        && (weights2 = try_get_direct_parent<constant>(*conv2, 1))
        && (weights1->output().shape()[2] == weights2->output().shape()[2])
        && (weights1->output().shape()[3] == weights2->output().shape()[3])
        && (bias1 = try_get_direct_parent<constant>(*conv1, 2))
        && (bias2 = try_get_direct_parent<constant>(*conv2, 2)))
    {
        context.inputs.emplace_back(&conv1->input());
        context.inputs.emplace_back(&conv2->input());
        context.outputs.emplace_back(&add->output());

        context.matched_nodes.emplace_back(conv1);
        context.matched_nodes.emplace_back(conv2);
        context.matched_nodes.emplace_back(weights1);
        context.matched_nodes.emplace_back(weights2);
        context.matched_nodes.emplace_back(bias1);
        context.matched_nodes.emplace_back(bias2);
        context.matched_nodes.emplace_back(add);
        return true;
    }

    return false;
}

void fold_two_conv2d_add_transform::process(transform_context &context)
{
    auto old_conv = node_cast<conv2d>(*context.matched_nodes[0]);
    auto old_weights1 = node_cast<constant>(*context.matched_nodes[2]);
    auto old_weights2 = node_cast<constant>(*context.matched_nodes[3]);
    auto old_bias1 = node_cast<constant>(*context.matched_nodes[4]);
    auto old_bias2 = node_cast<constant>(*context.matched_nodes[5]);
    auto old_add = node_cast<binary>(*context.matched_nodes[6]);

    // create new weights & bias
    std::array<shape_t, 2> concat_weights_shapes { old_weights1->output().shape(), old_weights2->output().shape() };
    auto new_weights = context.graph.emplace<concat>(dt_float32, concat_weights_shapes, 1);
    new_weights->input_at(0).connect(old_weights1->output());
    new_weights->input_at(1).connect(old_weights2->output());
    new_weights->name(old_weights1->name());

    auto new_bias = context.graph.emplace<binary>(binary_add, old_bias1->output().shape(), old_bias2->output().shape(), value_range<float>::full());
    new_bias->input_a().connect(old_bias1->output());
    new_bias->input_b().connect(old_bias2->output());
    new_bias->name(old_bias1->name());

    // create new conv2d
    std::array<shape_t, 2> concat_in_shapes { context.inputs[0]->shape(), context.inputs[1]->shape() };
    auto new_input = context.graph.emplace<concat>(dt_float32, concat_in_shapes, 1);
    new_input->input_at(0).connect(*context.inputs[0]->connection());
    new_input->input_at(1).connect(*context.inputs[1]->connection());

    auto new_conv = context.graph.emplace<conv2d>(new_input->output().shape(), new_weights->output().shape(), old_conv->groups(), old_conv->padding_h(), old_conv->padding_w(), old_conv->stride_h(), old_conv->stride_w(), old_conv->dilation_h(), old_conv->dilation_w(), old_add->fused_activation());
    new_conv->input().connect(new_input->output());
    new_conv->weights().connect(new_weights->output());
    new_conv->bias().connect(new_bias->output());
    new_conv->name(old_conv->name());

    auto inputs = context.outputs[0]->connections();
    for (auto &in : dup(inputs))
        in->connect(new_conv->output());
}
