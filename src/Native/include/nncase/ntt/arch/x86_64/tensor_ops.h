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
#include "../../tensor_ops.h"
#include "arch_types.h"
#include "avx_mathfun.h"

namespace nncase::ntt::tensor_ops {
template <> struct tload_scalar<ntt::vector<float, 8>> {
    ntt::vector<float, 8> operator()(float v) const noexcept {
        return _mm256_set1_ps(v);
    }
};
template <> struct tload_scalar<ntt::vector<float, 4, 4>> {
    ntt::vector<float, 4, 4> operator()(float v) const noexcept {
        ntt::vector<float, 4, 4> out;
        for (size_t i = 0; i < 4; i++) {
            for (size_t j = 0; j < 4; j++) {
                out(i, j) = v;
            }
        }
        return out;
    }
};

template <> struct tload_scalar<ntt::vector<float, 8, 8>> {
    ntt::vector<float, 8, 8> operator()(float v) const noexcept {
        ntt::vector<float, 8, 8> out;
        out(0) = _mm256_set1_ps(v);
        out(1) = _mm256_set1_ps(v);
        out(2) = _mm256_set1_ps(v);
        out(3) = _mm256_set1_ps(v);
        out(4) = _mm256_set1_ps(v);
        out(5) = _mm256_set1_ps(v);
        out(6) = _mm256_set1_ps(v);
        out(7) = _mm256_set1_ps(v);
        return out;
    }
};
} // namespace nncase::ntt::tensor_ops
