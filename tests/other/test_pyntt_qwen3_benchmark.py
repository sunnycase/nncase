# Copyright 2019-2021 Canaan Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from tools.pyntt_qwen3_benchmark import BENCHMARK_NAME
from tools.pyntt_qwen3_benchmark import SCHEMA_VERSION
from tools.pyntt_qwen3_benchmark import StepMeasurement
from tools.pyntt_qwen3_benchmark import _package_contract_json
from tools.pyntt_qwen3_benchmark import build_argument_parser
from tools.pyntt_qwen3_benchmark import drive_token_by_token
from tools.pyntt_qwen3_benchmark import inspect_generated_package
from tools.pyntt_qwen3_benchmark import main
from tools.pyntt_qwen3_benchmark import parse_annotations
from tools.pyntt_qwen3_benchmark import parse_scenario
from tools.pyntt_qwen3_benchmark import percentile
from tools.pyntt_qwen3_benchmark import prepare_isolated_triton_cache
from tools.pyntt_qwen3_benchmark import render_comparison_svg
from tools.pyntt_qwen3_benchmark import run_benchmark
from tools.pyntt_qwen3_benchmark import summarize_ms


def test_latency_summary_uses_linear_percentiles():
    samples = [4.0, 1.0, 3.0, 2.0]

    assert percentile(samples, 50) == pytest.approx(2.5)
    assert percentile(samples, 90) == pytest.approx(3.7)
    summary = summarize_ms(samples)
    assert summary["count"] == 4
    assert summary["mean_ms"] == pytest.approx(2.5)
    assert summary["p50_ms"] == pytest.approx(2.5)
    assert summary["p90_ms"] == pytest.approx(3.7)


def test_scenario_and_annotation_parsing_are_strict():
    scenario = parse_scenario("prefill:20:3")
    assert scenario.prompt_tokens == 20
    assert scenario.output_tokens == 3
    assert scenario.model_calls == 22

    assert parse_annotations(
        ["template_revision=2", 'policy="explicit"', "tag=nightly"]
    ) == {
        "policy": "explicit",
        "tag": "nightly",
        "template_revision": 2,
    }
    with pytest.raises(ValueError, match="positive"):
        parse_scenario("bad:0:3")
    with pytest.raises(ValueError, match="duplicate"):
        parse_annotations(["n=1", "n=2"])


def test_token_driver_prefills_and_decodes_one_token_per_call():
    scheduled_contexts = []
    invocations = []
    predictions = iter([100, 101, 102])

    def schedule_one():
        context = len(scheduled_contexts)
        scheduled_contexts.append(context)
        return {"context": context, "query_len": 1}

    def invoke_one(token, kv_cache, needs_prediction):
        invocations.append((token, kv_cache, needs_prediction))
        predicted = next(predictions) if needs_prediction else None
        return StepMeasurement(1.0, 2.0, predicted)

    result = drive_token_by_token([10, 11, 12], 3, schedule_one, invoke_one)

    assert [item[0] for item in invocations] == [10, 11, 12, 100, 101]
    assert [item[2] for item in invocations] == [False, False, True, True, True]
    assert scheduled_contexts == [0, 1, 2, 3, 4]
    assert result["generated_token_ids"] == [100, 101, 102]
    assert result["prefill_cuda_ms"] == [1.0, 1.0, 1.0]
    assert result["decode_cuda_ms"] == [1.0, 1.0]
    assert result["total_model_cuda_ms"] == pytest.approx(5.0)


def _strict_launch():
    return {
        "meta": {
            "shared_data_pool_bytes": 0,
            "shared_data_pool_alignment_bytes": 8,
        },
        "host_tensor_descriptors": [],
        "sharding": {
            "strategy": "replicated",
            "placement_axis": "b",
            "tensor_axis": 0,
            "extent": "1",
            "hierarchy": [1],
            "hierarchy_levels": "b",
            "global_shape": [],
        },
    }


def _render_kernel(symbol="main_prim"):
    return {
        "metadata": {
            "name": symbol,
            "op_kind": "entry",
            "inputs": ["input_ids", "kv_cache"],
            "outputs": ["logits"],
            "attrs": {
                "target_worker_width": 32,
                "target_threads_per_block": 256,
                "register_file_capacity_units": 65280,
                "register_file_allocation_granularity_units": 256,
                "registers_per_thread_limit": 255,
                "shared_memory_capacity_bytes": 101376,
            },
            "launch": _strict_launch(),
        },
        "helpers": [],
        "device_functions": [],
        "body_source": "pass",
    }


def _render_codegen_manifest(manifest):
    repo_root = Path(__file__).resolve().parents[2]
    package_root = str(repo_root / "pyntt")
    if package_root not in sys.path:
        sys.path.insert(0, package_root)
    from pyntt.codegen.render import render_manifest

    return render_manifest(manifest)


def _write_codegen_manifest(path, manifest):
    (path / "kernel_params.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    (path / "generated_kernels.py").write_text(
        _render_codegen_manifest(manifest), encoding="utf-8"
    )


def _write_generated_package(path, input_shape=(1,), layer_ids=("0",)):
    path.mkdir()
    metadata = {
        "pyntt_spec_version": 4,
        "target_kind": "pyntt",
        "target_machine": "cuda_rtx5060",
        "backend": "triton",
        "functions": [
            {
                "name": "main_prim",
                "is_entry": True,
                "inputs": [
                    {
                        "name": "input_ids",
                        "dtype": "int64",
                        "shape": list(input_shape),
                    },
                    {"name": "kvCache", "dtype": "object", "shape": []},
                ],
                "generated_kernels": [
                    {"attrs": {"layer_id": layer_id}} for layer_id in layer_ids
                ],
            }
        ],
    }
    (path / "metadata.json").write_text(
        "\ufeff" + json.dumps(metadata), encoding="utf-8"
    )
    manifest = {
        "pyntt_codegen_manifest_version": 8,
        "target_kind": "pyntt",
        "backend": "triton",
        "functions": [
            {
                "id": 0,
                "name": "main_prim",
                "module_kind": "pyntt",
                "is_entry": True,
                "render_kernels": [_render_kernel()],
            }
        ],
    }
    _write_codegen_manifest(path, manifest)
    (path / "__init__.py").write_text("# __init__.py\n", encoding="utf-8")
    (path / "model.py").write_text("# model.py\n", encoding="utf-8")
    (path / "assets").mkdir()
    (path / "assets" / "module_rdata.bin").write_bytes(b"weights")


def test_generated_package_contract_and_fingerprint(tmp_path):
    generated_dir = tmp_path / "generated"
    _write_generated_package(generated_dir)

    first = inspect_generated_package(generated_dir)
    second = inspect_generated_package(generated_dir)

    assert first.entry_function == "main_prim"
    assert first.input_dtype == "int64"
    assert first.input_shape == (1,)
    assert first.detected_layer_ids == ("0",)
    assert first.code_manifest_sha256 == second.code_manifest_sha256
    assert first.source_file_sha256 == second.source_file_sha256
    assert first.asset_file_bytes == {"assets/module_rdata.bin": 7}
    assert first.asset_file_sha256 == second.asset_file_sha256
    assert len(first.asset_file_sha256["assets/module_rdata.bin"]) == 64
    assert first.manifest_summary["pyntt_codegen_manifest_version"] == 8

    report_contract = _package_contract_json(first, generated_dir)
    assert report_contract["manifest"]["pyntt_codegen_manifest_version"] == 8
    assert "pipeline_gate" not in report_contract


@pytest.mark.parametrize(
    "input_shape,layer_ids,message",
    [
        ((20,), ("0",), "static one-token"),
        ((1,), ("0", "1"), "one-layer"),
        ((1,), (), "one-layer"),
    ],
)
def test_generated_package_rejects_incompatible_contract(
    tmp_path, input_shape, layer_ids, message
):
    generated_dir = tmp_path / "generated"
    _write_generated_package(generated_dir, input_shape, layer_ids)

    with pytest.raises(ValueError, match=message):
        inspect_generated_package(generated_dir)


def test_generated_package_requires_manifest_v8(tmp_path):
    generated_dir = tmp_path / "generated"
    _write_generated_package(generated_dir)
    manifest_path = generated_dir / "kernel_params.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["pyntt_codegen_manifest_version"] = 7
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="expected 8"):
        inspect_generated_package(generated_dir)


@pytest.mark.parametrize(
    "removed_field", ("pipeline_executions", "shared_arena", "microkernels")
)
def test_generated_package_rejects_removed_compiler_scheduling_fields(
    tmp_path, removed_field
):
    generated_dir = tmp_path / "generated"
    _write_generated_package(generated_dir)
    manifest_path = generated_dir / "kernel_params.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["functions"][0]["render_kernels"][0][removed_field] = []
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="unexpected fields"):
        inspect_generated_package(generated_dir)


def test_run_parser_has_no_compiler_pipeline_mode(tmp_path):
    parser = build_argument_parser(tmp_path)
    args = parser.parse_args(
        [
            "run",
            "--generated-dir",
            str(tmp_path / "generated"),
            "--output-json",
            str(tmp_path / "result.json"),
        ]
    )

    assert not hasattr(args, "allow_ordinary")
    assert "allow-ordinary" not in parser.format_help()


def test_isolated_triton_cache_rejects_preexisting_artifacts(tmp_path):
    cache_dir = tmp_path / "triton-cache"
    cache_dir.mkdir()
    (cache_dir / "stale.ptx").write_text("stale", encoding="utf-8")

    with pytest.raises(ValueError, match="must be empty"):
        prepare_isolated_triton_cache(cache_dir)


def test_benchmark_rejects_triton_imported_before_cache_isolation(
    tmp_path, monkeypatch
):
    parser = build_argument_parser(tmp_path)
    args = parser.parse_args(
        [
            "run",
            "--generated-dir",
            str(tmp_path / "generated"),
            "--output-json",
            str(tmp_path / "result.json"),
        ]
    )
    monkeypatch.setitem(sys.modules, "triton", object())

    with pytest.raises(RuntimeError, match="before importing Triton"):
        run_benchmark(args, tmp_path)


def _summary(p50, p90):
    return {
        "count": 10,
        "mean_ms": p50,
        "stdev_ms": 0.1,
        "min_ms": p50 - 0.1,
        "p50_ms": p50,
        "p90_ms": p90,
        "p99_ms": p90 + 0.1,
        "max_ms": p90 + 0.2,
    }


def _report(label, scale):
    scenarios = []
    for name, prompt_tokens, output_tokens, latency in (
        ("decode_1x3", 1, 3, 3.0),
        ("prefill_20x3", 20, 3, 12.0),
    ):
        scenarios.append(
            {
                "name": name,
                "key": f"prompt={prompt_tokens};output={output_tokens}",
                "prompt_tokens": prompt_tokens,
                "output_tokens": output_tokens,
                "model_calls_per_request": prompt_tokens + output_tokens - 1,
                "latency_ms": {
                    "total_model_cuda": _summary(
                        latency * scale, latency * scale * 1.1
                    ),
                    "time_to_first_token_cuda": _summary(
                        latency * scale * 0.8, latency * scale * 0.9
                    ),
                    "decode_token_cuda": _summary(
                        latency * scale * 0.1, latency * scale * 0.12
                    ),
                },
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark": BENCHMARK_NAME,
        "label": label,
        "hardware": {
            "name": "Test GPU <0>",
            "compute_capability": [9, 0],
            "driver_version": "999.1",
        },
        "measurement": {"timing": "cuda-events", "fixed_output_tokens": True},
        "run_config": {
            "warmup_iterations_per_scenario": 5,
            "timed_iterations_per_scenario": 20,
            "device_index": 0,
            "model_config_sha256": "model-config",
            "prompt_index": 0,
            "prompt_text_sha256": "prompt-hash",
        },
        "model_package": {
            "code_manifest_sha256": "abcdef0123456789",
            "input_ids": {
                "name": "input_ids",
                "dtype": "int64",
                "shape": [1],
                "tokens_per_call": 1,
            },
            "detected_layer_ids": ["0"],
            "manifest": {
                "target_kind": "pyntt",
                "target_machine": "cuda_rtx5060",
                "backend": "triton",
            },
            "asset_file_sha256": {"assets/module_rdata.bin": "asset-hash"},
        },
        "paged_attention_config": {"block_size": 16, "model_layers": 1},
        "scenarios": scenarios,
    }


def test_svg_is_deterministic_and_compares_matching_scenarios():
    baseline = _report("n=1 & baseline", 1.0)
    candidate = _report("n=2 candidate", 0.8)

    first = render_comparison_svg(candidate, baseline)
    second = render_comparison_svg(candidate, baseline)

    assert first == second
    assert "candidate p50 Δ -20.0%" in first
    assert "n=1 &amp; baseline" in first
    assert "Test GPU &lt;0&gt;" in first
    ET.fromstring(first)


def test_svg_rejects_non_matching_baseline():
    baseline = _report("baseline", 1.0)
    candidate = _report("candidate", 0.8)
    candidate["scenarios"].pop()

    with pytest.raises(ValueError, match="scenario keys differ"):
        render_comparison_svg(candidate, baseline)


@pytest.mark.parametrize(
    "path,new_value,field",
    [
        (("hardware", "driver_version"), "different", "hardware.driver_version"),
        (
            ("run_config", "prompt_text_sha256"),
            "different",
            "run_config.prompt_text_sha256",
        ),
        (
            ("model_package", "asset_file_sha256"),
            {"assets/module_rdata.bin": "different"},
            "model_package.asset_file_sha256",
        ),
        (
            ("model_package", "manifest", "target_machine"),
            "different",
            "model_package.manifest.target_machine",
        ),
        (("paged_attention_config", "block_size"), 32, "paged_attention_config"),
    ],
)
def test_svg_rejects_non_comparable_ab_reports(path, new_value, field):
    baseline = _report("baseline", 1.0)
    candidate = _report("candidate", 0.8)
    target = candidate
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = new_value

    with pytest.raises(ValueError, match=field):
        render_comparison_svg(candidate, baseline)


def test_svg_subcommand_writes_deterministic_artifact(tmp_path):
    baseline_json = tmp_path / "baseline.json"
    candidate_json = tmp_path / "candidate.json"
    output_svg = tmp_path / "comparison.svg"
    baseline_json.write_text(json.dumps(_report("n=1", 1.0)), encoding="utf-8")
    candidate_json.write_text(json.dumps(_report("n=2", 0.8)), encoding="utf-8")

    arguments = [
        "svg",
        "--baseline-json",
        str(baseline_json),
        "--candidate-json",
        str(candidate_json),
        "--output-svg",
        str(output_svg),
    ]
    assert main(arguments) == 0
    first = output_svg.read_bytes()
    assert main(arguments) == 0
    assert output_svg.read_bytes() == first
