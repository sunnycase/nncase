/* Copyright 2019-2020 Canaan Inc.
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
#include <memory>
#include <nncase/runtime/error.h>
#include <nncase/runtime/model.h>
#include <nncase/runtime/result.h>
#include <nncase/runtime/runtime.h>

BEGIN_NS_NNCASE_RUNTIME

typedef void (*runtime_activator_t)(result<std::unique_ptr<runtime_base>> &result);

#define RUNTIME_ACTIVATOR_NAME create_runtime
#define SIMULATOR_ACTIVATOR_NAME create_simulator

extern "C" NNCASE_API void create_runtime(const model_target_t &target_id, result<std::unique_ptr<runtime_base>> &result);

END_NS_NNCASE_RUNTIME
