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
#include <cstddef>
#include <type_traits>

namespace nncase::ntt::detail {
template <class TLoadPrevious>
inline constexpr bool has_load_previous_v =
    !std::is_same_v<std::remove_cvref_t<TLoadPrevious>, std::nullptr_t>;

template <class TLoadPrevious>
constexpr bool load_previous(const TLoadPrevious &value) noexcept {
    if constexpr (!has_load_previous_v<TLoadPrevious>) {
        return false;
    } else if constexpr (requires { value(); }) {
        return static_cast<bool>(value());
    } else {
        return static_cast<bool>(value);
    }
}
} // namespace nncase::ntt::detail
