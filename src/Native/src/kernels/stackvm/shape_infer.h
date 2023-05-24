/* This file is generated by tools/stackvm_gen/IsaGen at 2022/5/14 下午6:47:34
 * +08:00.
 *
 * Copyright 2019-2021 Canaan Inc.
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
#pragma once
#include "nncase/kernels/kernel_utils.h"
#include <nncase/kernels/kernel_context.h>
#include <nncase/runtime/datatypes.h>
#include <nncase/runtime/error.h>
#include <nncase/runtime/result.h>
#include <nncase/runtime/runtime_op_utility.h>
#include <nncase/runtime/stackvm/opcode.h>
#include <nncase/runtime/util.h>
#include <nncase/tensor.h>
#include <nncase/value.h>
#include <numeric>

BEGIN_NS_NNCASE_KERNELS_MODULE(stackvm)

inline dims_t conv2d_infer_shape(const dims_t &in_shape,
                                 const dims_t &weights_shape,
                                 const dims_t &stride, const dims_t &dilation,
                                 const paddings_t &paddings) {
    auto new_shape = in_shape;
    new_shape[1] = weights_shape[0];
    new_shape[2] = kernels::detail::get_windowed_output_size(
        in_shape[2], weights_shape[2], stride[0], dilation[0], paddings[0]);
    new_shape[3] = kernels::detail::get_windowed_output_size(
        in_shape[3], weights_shape[3], stride[1], dilation[1], paddings[1]);
    return new_shape;
}

inline dims_t concat_infer_shape(std::vector<dims_t> shapes, int axis) {
    auto new_shape = shapes[0];
    new_shape[axis] = std::accumulate(
        shapes.begin(), shapes.end(), 0,
        [&](auto sum, auto in_shape) -> int { return sum + in_shape[axis]; });
    return new_shape;
}

inline dims_t gather_infer_shape(const dims_t &in_shape,
                                 const dims_t &index_shape, int axis) {
    if (in_shape.size() == 1 && index_shape.size() == 0) {
        // scalar
        return dims_t();
    }
    auto index_shape_copy = index_shape;
    for (size_t i = 0; i < index_shape.size(); ++i) {
        if (index_shape[i] < 0) {
            index_shape_copy[i] += in_shape[axis];
        }
    }
    auto new_shape = in_shape;
    auto indices_shape = index_shape.size() == 0 ? dims_t() : index_shape_copy;
    new_shape.erase(new_shape.begin() + axis);
    new_shape.insert(new_shape.begin() + axis, indices_shape.begin(),
                     indices_shape.end());
    return new_shape;
}

inline dims_t gather_nd_infer_shape(const dims_t &in_shape,
                                    const dims_t &index_shape,
                                    size_t batch_dims) {
    auto new_shape = index_shape;
    new_shape.pop_back();
    new_shape.insert(new_shape.end(),
                     in_shape.begin() + index_shape.back() + batch_dims,
                     in_shape.end());
    if (new_shape.empty()) {
        new_shape.push_back(1);
    }
    return new_shape;
}

inline dims_t slice_infer_shape(const dims_t &in_shape, const axes_t &begins,
                                const axes_t &ends, const axes_t &strides) {
    auto new_shape = dims_t();
    for (size_t i = 0; i < strides.size(); i++) {
        auto stride = strides[i];
        auto begin_val = begins[i];
        auto end_val = std::min(ends[i], (int64_t)in_shape[i]);
        auto dim = (int)std::ceil(
            ((float)std::abs(end_val - begin_val) / (float)std::abs(stride)));
        new_shape.push_back(dim);
    }

    return new_shape.size() ? new_shape : dims_t{1};
}

inline std::vector<dims_t>
split_shape_infer(const dims_t &in_shape, size_t axis, const dims_t &sections) {
    auto result = std::vector<dims_t>();
    for (size_t i = 0; i < sections.size(); ++i) {
        auto shape = in_shape;
        shape[axis] = sections[i];
        result.push_back(shape);
    }
    return result;
}

inline dims_t reshape_shape_infer(const dims_t &in_shape,
                                  const axes_t &new_shape) {
    auto neg_index = -1;
    auto sum = 1;
    for (size_t i = 0; i < new_shape.size(); ++i) {
        if (new_shape[i] != -1) {
            sum *= new_shape[i];
        } else {
            neg_index = i;
        }
    }
    if (neg_index == -1) {
        return dims_t(new_shape.begin(), new_shape.end());
    } else {
        auto result_shape = new_shape;
        auto in_size = std::accumulate(in_shape.begin(), in_shape.end(), 1,
                                       std::multiplies<int64_t>{});
        result_shape[neg_index] = in_size / sum;
        return dims_t(result_shape.begin(), result_shape.end());
    }
}

inline dims_t stack_infer_shape(dims_t shape0, int input_count, int axis) {
    shape0.insert(shape0.begin() + axis, input_count);
    return shape0;
}

inline dims_t unsqueeze_infer_shape(const dims_t &in_shape,
                                    const axes_t &axes) {
    if (in_shape.size() == 0 && axes.size() == 1) {
        return dims_t{1};
    }
    auto new_shape = in_shape.size() == 0 ? dims_t{1} : in_shape;
    for (size_t i = 0; i < axes.size(); i++) {
        if (axes[i] >= 0) {
            new_shape.insert(new_shape.begin() + axes[i], 1);
        } else {
            new_shape.insert(new_shape.end() + axes[i] + 1, 1);
        }
    }
    return new_shape;
}

inline dims_t flatten_infer_shape(const dims_t &in_shape, size_t axis) {
    auto first =
        (size_t)std::accumulate(in_shape.begin(), in_shape.begin() + axis, 1,
                                std::multiplies<size_t>());
    auto second = (size_t)std::accumulate(
        in_shape.begin() + axis, in_shape.end(), 1, std::multiplies<size_t>());
    return dims_t{first, second};
}

inline dims_t squeeze_infer_shape(const dims_t &in_shape, const dims_t &axes) {
    auto result_rank = in_shape.size() - axes.size();
    if (result_rank == 0) {
        return dims_t();
    }
    // todo:error
    auto tmp_out_shpae = in_shape;
    auto max = std::numeric_limits<size_t>::max();
    for (size_t i = 0; i < axes.size(); ++i) {
        tmp_out_shpae[axes[i]] = max;
    }
    auto out_shape = dims_t();
    for (auto d : tmp_out_shpae) {
        if (d != max) {
            out_shape.push_back(d);
        }
    }
    return out_shape;
}

inline dims_t where_infer_shape(const dims_t &cond_shape, const dims_t &x_shape,
                                const dims_t &y_shape) {
    return kernels::detail::get_binary_output_shape(
        kernels::detail::get_binary_output_shape(cond_shape, x_shape), y_shape);
}

inline dims_t tile_infer_shape(const dims_t &in_shape, const dims_t &repeats) {
    auto out_shape = dims_t(in_shape.size());
    for (size_t i = 0; i < out_shape.size(); ++i) {
        out_shape[i] = in_shape[i] * repeats[i];
    }
    return out_shape;
}

inline dims_t reduce_infer_shape(const dims_t &in_shape, const dims_t &axes,
                                 bool keep_dims) {
    auto tmp_shape = in_shape;
    for (size_t i = 0; i < axes.size(); ++i) {
        auto d = keep_dims ? 1 : 0;
        tmp_shape[axes[i]] = d;
    }
    auto new_shape = dims_t();
    for (auto d : tmp_shape) {
        if (d != 0) {
            new_shape.push_back(d);
        }
    }
    return new_shape;
}

inline std::vector<dims_t>
lstm_infer_shape(const dims_t &x_shape, const dims_t &init_h_shape,
                 const dims_t &init_c_shape,
                 runtime::stackvm::lstmdirection_t direction,
                 runtime::stackvm::lstmlayout_t layout, size_t hidden_size,
                 size_t out_size) {
    auto num_directions =
        direction == runtime::stackvm::lstmdirection_t::bidirectional ? 2 : 1;
    auto seq_len_index = layout == runtime::stackvm::lstmlayout_t::zero ? 0 : 1;
    auto y_shape = x_shape;
    y_shape.insert(y_shape.begin() + seq_len_index + 1, num_directions);
    *(y_shape.end() - 1) = hidden_size;
    if (out_size == 1) {
        return {y_shape};
    } else if (out_size == 2) {
        return {y_shape, init_h_shape};
    } else {
        return {y_shape, init_h_shape, init_c_shape};
    }
}

inline dims_t transpose_infer_shape(const dims_t &in_shape,
                                    const dims_t &perm) {
    auto new_shape = in_shape;
    for (size_t i = 0; i < in_shape.size(); ++i) {
        new_shape[i] = in_shape[perm[i]];
    }
    return new_shape;
}

inline dims_t pad_infer_shape(const dims_t &in_shape, const paddings_t &pads) {
    auto d = pads.size();
    auto new_shape = in_shape;
    for (size_t i = 0; i < d; ++i) {
        new_shape[in_shape.size() - d + i] += pads[i].sum();
    }
    return new_shape;
}

inline dims_t space_to_batch_shape_infer(const dims_t &in_shape,
                                         const dims_t &block_shape,
                                         const paddings_t &paddings) {
    auto batch = in_shape[0] * runtime::compute_size(block_shape);
    auto out_shape = dims_t{batch};
    auto m = block_shape.size();
    for (size_t i = 0; i < m; ++i) {
        auto d = (in_shape[i + 1] + paddings[i].sum()) / block_shape[i];
        out_shape.push_back(d);
    }
    auto remain_size = in_shape.size() - 1 - m;
    if (remain_size > 0) {
        out_shape.insert(out_shape.end(), in_shape.end() - remain_size,
                         in_shape.end());
    }
    return out_shape;
}

inline dims_t onehot_infer_shape(const dims_t &indices_shape, size_t depth,
                                 size_t axis) {
    auto new_shape = indices_shape;
    new_shape.insert(new_shape.begin() + axis, depth);
    return new_shape;
}

inline result<dims_t> matmul_infer_shape(const dims_t &lhs_shape,
                                         const dims_t &rhs_shape) {
    if (lhs_shape.size() == 2 && rhs_shape.size() == 2) {
        auto new_shape = dims_t{lhs_shape[0], rhs_shape[1]};
        return ok(new_shape);
    }

    auto new_a_shape = runtime::to_4d(lhs_shape);
    auto new_b_shape = runtime::to_4d(rhs_shape);
    auto big_shape = std::max(lhs_shape.size(), rhs_shape.size());
    auto new_shape = dims_t();
    for (size_t i = 0; i < big_shape - 2; ++i) {
        new_shape.push_back(std::max(new_a_shape[i + 4 - big_shape],
                                     new_b_shape[i + 4 - big_shape]));
    }
    new_shape.push_back(lhs_shape[lhs_shape.size() - 2]);
    new_shape.push_back(rhs_shape.back());
    return ok(new_shape);
}

inline dims_t topk_infer_shape(const dims_t &x, int k, int axis) {
    auto result = x;
    result[axis] = k;
    return result;
}
END_NS_NNCASE_KERNELS_MODULE