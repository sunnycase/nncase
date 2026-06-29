# AGENTS.md

This file is for automated agents and developers working in this repository.
Unless a subdirectory provides a more specific `AGENTS.md`, this file applies to
the entire repository.

## Project Background

nncase is a neural network compiler for AI accelerators. This repository
contains the C++ native runtime and kernels, the C# compiler and tools, Python
bindings, and tests. The main goals are:

- Import and compile models from TFLite, ONNX, Caffe, NCNN, and related formats
  to supported targets.
- Support Canaan chips such as K210/K230, plus CPU, Vulkan, CUDA, and other
  runtime capabilities.
- Keep the compiler, runtime, Python package, and test infrastructure
  consistent with each other.

Before changing code, understand the relevant layer: IR, importer, transform,
codegen, runtime, kernel, Python binding, or test infrastructure. Do not patch
around cross-layer contract issues at a local call site.

## Repository Directory Index

- `.github/workflows/`: CI definitions for compiler builds, runtime builds,
  Python wheel builds, notebook tests, and formatting.
- `src/`: main C# compiler stack and native bridge. Key projects include
  `Nncase.Core`, `Nncase.Importer`, `Nncase.Passes`, `Nncase.Evaluator`,
  `Nncase.Schedule`, `Nncase.CodeGen`, `Nncase.Compiler`, `Nncase.Studio`,
  `Nncase.Tests`, and `Native`.
- `modules/`: optional compiler/runtime modules, including K210, NTT, StackVM,
  and native K210/Vulkan module code.
- `targets/`: target registration and target-specific implementation for CPU,
  K210, Vulkan, and related backends.
- `ntt/`: native tensor template runtime code, kernels, benchmarks, and CTest
  coverage for NTT.
- `python/`: Python packages, type stubs, and native FFI bindings for `nncase`
  and `nncaseruntime`.
- `tests/`: Python integration tests, importer tests, model tests, accuracy
  helpers, schedule tests, transform tests, and LLM-related tests.
- `benchmark/`: benchmark build targets and benchmark model assets.
- `examples/`: user-facing examples, notebooks, audio samples, and sample
  models.
- `docs/`: user documentation, design notes, operator support tables, images,
  and Studio assets.
- `cmake/`: shared CMake helper files, dependency wiring, runtime setup, and
  test helpers.
- `toolchains/`: Conan profiles and CMake toolchain files for native and cross
  builds.
- `tools/`: formatting scripts, source generators, coverage configuration,
  diagnostic utilities, and maintenance scripts.
- `csharp/`: native simulator bridge, CMake integration, and NuGet packaging
  files.
- `third_party/`: vendored or mirrored third-party source. Avoid changing this
  unless the task explicitly requires dependency maintenance.

Treat `build/`, `install/`, `tests_output/`, `wheelhouse/`, caches, and
language-specific `bin/`/`obj/` directories as local outputs. Do not edit or
commit them as source changes unless the task explicitly asks for generated
artifacts.

## Documentation Index

- [`README.md`](README.md): project overview, high-level feature summary,
  install notes, and source build quickstart.
- [`docs/readme_ZH.md`](docs/readme_ZH.md): Chinese project overview and user
  entry point.
- [`docs/build.md`](docs/build.md): legacy/source build notes, Python test
  setup notes, and packaging notes.
- [`docs/USAGE_v2_EN.md`](docs/USAGE_v2_EN.md) and
  [`docs/USAGE_v2.md`](docs/USAGE_v2.md): K230 usage guides in English and
  Chinese.
- [`docs/FAQ_EN.md`](docs/FAQ_EN.md) and [`docs/FAQ_ZH.md`](docs/FAQ_ZH.md):
  troubleshooting and frequently asked questions.
- Operator support tables: [`docs/tflite_ops.md`](docs/tflite_ops.md),
  [`docs/onnx_ops.md`](docs/onnx_ops.md), [`docs/caffe_ops.md`](docs/caffe_ops.md),
  [`docs/ncnn_ops.md`](docs/ncnn_ops.md), and
  [`docs/paddle_ops.md`](docs/paddle_ops.md).
- [`docs/MixQuant.md`](docs/MixQuant.md): mixed quantization guidance.
- [`docs/shape_bucket.md`](docs/shape_bucket.md): dynamic shape bucket
  configuration and behavior.
- [`docs/nncase_studio.md`](docs/nncase_studio.md): nncase Studio usage notes
  and related UI assets.
- [`examples/user_guide/k230_simulate-EN.ipynb`](examples/user_guide/k230_simulate-EN.ipynb)
  and [`examples/user_guide/k230_simulate-ZH.ipynb`](examples/user_guide/k230_simulate-ZH.ipynb):
  runnable K230 simulation notebooks.
- [`csharp/readme.md`](csharp/readme.md): C# native simulator bridge notes.

## Development Environment

Use the conda environment `nncase` for local development by default:

```sh
conda activate nncase
```

CI mainly uses Python 3.10, .NET 8, Ninja, Conan 2.6.0, and CMake 3.30.3.
Linux builds use `gcc-14`/`g++-14`. Keep the local environment close to CI:

```sh
python -m pip install --upgrade pip
python -m pip install conan==2.6.0 cmake==3.30.3
conan remote add sunnycase https://conan.sunnycase.moe --index 0
conan remote update conancenter --url "https://center2.conan.io"
```

If the `sunnycase` remote already exists, inspect it with `conan remote list`
and update it to the URL above. Do not hide environment problems with commands
that ignore failures.

For day-to-day development, exact tool versions do not need to be force-pinned
unless the issue is version-sensitive. It is more important to keep the local
build/test flow aligned with the workflow stages and artifact layout.

## Build Instructions

Prefer `.github/workflows/*.yml` as the source of truth. Local commands should
reuse the same profiles, options, presets, and test entry points as CI whenever
possible.

### Developer Compiler Flow

When changing compiler targets, codegen, Python bindings, or PyNTT, keep the
local development flow aligned with `compiler-build.yml`: build and install the
native Python bridge first, then build/publish the C# compiler, then run Python
tests from `install/`. Debug builds are fine for this loop.

```sh
conda activate nncase
export CC=gcc-14
export CXX=g++-14

conan install . --build=missing \
  -s build_type=Debug \
  -pr:a=toolchains/x86_64-linux.profile.jinja \
  -o "&:runtime=False" \
  -o "&:python=True" \
  -o "&:tests=False"

cmake --preset conan-debug
cmake --build build/Debug --config Debug
cmake --install build/Debug --prefix install

dotnet restore -r linux-x64
dotnet build -c Debug --no-restore
dotnet publish src/Nncase.Compiler -c Debug --no-restore --sc false -r linux-x64

cp -a src/Nncase.Compiler/bin/Debug/net8.0/linux-x64/publish/. install/
cp install/lib/*.so install/
```

Always point Python tests at the installed native bridge and compiler publish
artifact. Do not validate a compiler/codegen change with only `dotnet build`;
that can leave Python using stale `_nncase` or stale managed assemblies.

```sh
export PYTHONPATH="$PWD/install/lib:$PWD/install/python:$PWD/tests:${PYTHONPATH}"
export LD_LIBRARY_PATH="$PWD/install/lib:${LD_LIBRARY_PATH}"
export NNCASE_COMPILER="$PWD/install/Nncase.Compiler.dll"
export NNCASE_TILING_MAX_SOLUTIONS=1

python - <<'PY'
import nncase
print("cpu", nncase.check_target("cpu"))
print("cuda", nncase.check_target("cuda"))
PY
```

For PyNTT development, also add the outer `pyntt/` package directory to
`PYTHONPATH`, because the importable package is nested under `pyntt/pyntt/`.
Reuse the existing pytest suites by selecting the target with
`NNCASE_TEST_TARGETS=pyntt`; do not create a separate PyNTT-only suite unless
the existing runner cannot express the case.

```sh
export PYTHONPATH="$PWD/install/lib:$PWD/install/python:$PWD/tests:$PWD/pyntt:${PYTHONPATH}"
export NNCASE_TEST_TARGETS=pyntt

python - <<'PY'
import nncase
print("pyntt", nncase.check_target("pyntt"))
PY

pytest tests/importer/onnx_/basic/test_identity.py::test_identity \
  --doctest-modules -q -s
```

If `nncase.check_target("pyntt")` is false, first check that
`install/Nncase.Compiler.dll`, `install/Nncase.Modules.NTT.dll`, and
`install/lib/_nncase*.so` came from the same build loop. `Nncase.Modules.NTT.dll`
is a built-in module loaded through `AddNTT()`, and `NNCASE_PLUGIN_PATH` will
not load it as an external plugin.

### Build Native Compiler Components

Linux x86_64 Release:

```sh
conda activate nncase
export CC=gcc-14
export CXX=g++-14

conan install . --build=missing \
  -s build_type=Release \
  -pr:a=toolchains/x86_64-linux.profile.jinja \
  -o "&:runtime=False" \
  -o "&:python=True" \
  -o "&:tests=False"

cmake --preset conan-release
cmake --build build/Release --config Release
cmake --install build/Release --prefix install
```

For macOS arm64, use `toolchains/aarch64-macos.profile.jinja`. For cross builds
or target runtime builds, check the matching workflow and
`toolchains/*.profile.jinja` first.

### Build the C# Compiler and Tools

Install the native components to `install/` first, then run:

```sh
dotnet restore -r linux-x64
dotnet build -c Release --no-restore
dotnet publish src/Nncase.Compiler -c Release --no-restore --sc false -r linux-x64
dotnet publish src/Nncase.Studio -c Release --no-restore --sc false -r linux-x64
```

Use `osx-arm64` as the RID for macOS arm64 and `win-x64` for Windows x64.

### Build Runtime and Run NTT Tests

```sh
conda activate nncase
export CC=gcc-14
export CXX=g++-14

conan install . --build=missing \
  -s build_type=Release \
  -pr:a=toolchains/x86_64-linux.profile.jinja \
  -o "&:runtime=True" \
  -o "&:python=True" \
  -o "&:tests=True"

cmake --preset conan-runtime-release
cmake --build build/Release --config Release
cmake --install build/Release --prefix install
ctest -C Release --test-dir build/Release/ntt/test/ctest --output-on-failure -j4
```

K230 RISC-V Linux runtime builds require the cross toolchain, QEMU loader
arguments, and `-pr:h/-pr:b` profile combination used by CI. Do not assume that
a local x86 build covers cross-target behavior unless those conditions are
reproduced.

### Python and Integration Tests

```sh
conda activate nncase
python -m pip install -r requirements.test.txt

export PYTHONPATH="$PWD/install/lib:$PWD/install/python:$PWD/tests:${PYTHONPATH}"
export LD_LIBRARY_PATH="$PWD/install/lib:${LD_LIBRARY_PATH}"
export NNCASE_COMPILER="$PWD/install/Nncase.Compiler.dll"
export NNCASE_TILING_MAX_SOLUTIONS=1

pytest tests/other/ --doctest-modules
pytest -n 2 tests/importer/onnx_/basic/ --doctest-modules
pytest -n 2 tests/importer/tflite_/basic/ --doctest-modules
pytest -n 2 tests/importer/ncnn_/basic/ --doctest-modules
```

On macOS, use `DYLD_LIBRARY_PATH`. On Windows, add `install/bin` to `PATH` and
adjust `PYTHONPATH` as shown in the workflow.

### Python Wheel

Python package builds are driven by `compiler-python-release.yml` and the
`cibuildwheel` settings in `pyproject.toml`. Before building the wheel, build
the `Nncase.Compiler` publish artifact for the target RID through the same
workflow path and ensure it is present under `install/`:

```sh
conda activate nncase
python -m pip install cibuildwheel
python -m cibuildwheel --output-dir wheelhouse
```

## Formatting and Static Conventions

Run formatters according to the affected scope before submitting changes:

```sh
export CLANG_FORMAT_LLVM_INSTALL_DIR=/usr
bash tools/clang-format.sh
dotnet format -v diag
python -m pip install autopep8
autopep8 tests python -r --in-place
```

C++ follows the repository `.clang-format`. C# follows `.editorconfig`,
StyleCop settings, and `dotnet format`. Python tests and bindings use
autopep8. Do not hand-format code in a way that fights these tools.

## Coding Guidelines

- Be formal: follow the existing architecture, naming, error handling, and test
  patterns. Add new behavior in the correct module instead of placing temporary
  logic at a call site.
- Generalize: implementations should cover the relevant class of shapes, dtypes,
  targets, layouts, or platforms. Avoid hard-coded branches for a single case.
- Be safe: validate external inputs, model metadata, buffer sizes, shape ranges,
  dtypes, and target capabilities. Avoid undefined behavior, out-of-bounds
  access, dangling lifetimes, and silent precision loss.
- Fail fast: report unmet preconditions early. Error messages should include the
  context needed to find the root cause. Do not swallow exceptions, return fake
  success, or hide errors behind defaults.
- Do not workaround: do not bypass problems by skipping passes, weakening
  assertions, disabling tests, hard-coding special models, silently falling
  back, or changing CI thresholds.
- Fix root causes: when something fails, identify the real cause and fix it at
  the lowest reasonable owning layer. If the fix crosses layers, update the
  contract, callers, and tests together.
- Preserve compatibility: changes to public APIs, serialization formats, model
  ABI, runtime ABI, Python package interfaces, and CI artifact layout must have
  a clear reason and migration consideration.
- Close the test loop: for bug fixes, prefer adding the smallest regression test
  that fails before the fix and passes after it. Broaden coverage for shared
  logic, backend behavior, or numerical correctness.

## Issue Handling Principles

1. Reproduce the issue first, including the command, input, environment
   variables, and failure log.
2. Narrow the scope to the concrete layer: import, type/shape inference,
   rewrite, tiling, codegen, runtime, kernel, binding, or test infrastructure.
3. Check whether an equivalent helper, pass, op, evaluator, kernel, or test
   utility already exists.
4. Fix the root cause and keep or strengthen existing assertions. Do not remove
   diagnostics just to make tests pass.
5. Verify with commands that are as close to CI as possible. If full validation
   is not possible, state what was not verified and why.

Generated files, third-party code, build outputs, and download caches should not
be modified as collateral damage. Before finishing, check `git status --short`
and confirm that only files needed for the task are changed.
