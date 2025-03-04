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
#pragma once
#include <nncase/runtime/k210/compiler_defs.h>
#include <nncase/transforms/transform.h>

namespace nncase::ir::transforms::k210
{
class NNCASE_MODULES_K210_API kpu_conv2d_transform : public transform
{
public:
    kpu_conv2d_transform(bool use_mse_quant_w) noexcept
        : use_mse_quant_w_(use_mse_quant_w) { }
    void process(transform_context &context) override;

protected:
    bool skip_self_contained_check() const noexcept override { return true; }
    bool on_try_match(ir::node &node, transform_context &context) override;

private:
    bool use_mse_quant_w_;
};
}
