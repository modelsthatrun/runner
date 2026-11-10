"""The profile's aggregation, on a made-up trace. No GPU needed."""

import pytest

from mtr_measure import profile, record


def call(ts: float, dur: float) -> dict:
    return {"ph": "X", "cat": "user_annotation", "name": profile.CALL, "ts": ts, "dur": dur}


def kernel(name: str, ts: float, dur: float, cat: str = "kernel") -> dict:
    return {"ph": "X", "cat": cat, "name": name, "ts": ts, "dur": dur}


def trace(offset: float) -> list[dict]:
    """One call of 1000 µs: a GEMM queued straight behind a copy, a gap the CPU caused, then two elementwise kernels."""
    o = offset
    return [
        call(o, 1000),
        kernel("Memcpy HtoD (Pageable -> Device)", o + 10, 40, "gpu_memcpy"),
        kernel("gemm", o + 51, 200),  # starts 1 µs after the copy ended: the copy was GPU-bound
        kernel("elementwise", o + 400, 50),  # 149 µs gap: the gemm wasn't holding anything up
        kernel("elementwise", o + 451, 50),  # the last one: the closing synchronize waited for it
        {"ph": "X", "cat": "cpu_op", "name": "aten::add", "ts": o + 390, "dur": 5},
    ]


def test_aggregate_medians_and_bound_time():
    events = trace(0) + trace(5000) + trace(10000)
    agg = profile.aggregate(events, skip=1)
    assert agg["calls"] == 2
    by = {k["name"]: k for k in agg["kernels"]}
    assert by["gemm"]["ms"] == pytest.approx(0.2)
    assert by["gemm"]["bound_ms"] == 0
    assert by["Memcpy HtoD (Pageable -> Device)"]["kind"] == "memcpy"
    assert by["Memcpy HtoD (Pageable -> Device)"]["bound_ms"] == pytest.approx(0.04)
    assert by["elementwise"]["count"] == 2
    assert by["elementwise"]["ms"] == pytest.approx(0.1)
    assert by["elementwise"]["bound_ms"] == pytest.approx(0.1)
    # Busy is the union of the intervals: 40 + 200 + 50 + 50 µs.
    assert agg["gpu_busy_ms"] == pytest.approx(0.34)
    assert agg["profiled_wall_ms"] == pytest.approx(1.0)
    assert agg["kernels"][0]["name"] == "gemm"


def test_overlapping_kernels_count_once_in_busy_time():
    events = [call(0, 100), kernel("a", 0, 50), kernel("b", 10, 50)]
    agg = profile.aggregate(events, skip=0)
    assert agg["gpu_busy_ms"] == pytest.approx(0.06)
    assert {k["name"]: k["bound_ms"] for k in agg["kernels"]} == {"a": pytest.approx(0.05), "b": pytest.approx(0.05)}


def test_no_calls_is_an_error():
    with pytest.raises(profile.MeasureError):
        profile.aggregate([kernel("a", 0, 1)], skip=0)


def test_record_passes_its_schema():
    worker = {
        "precision": "fp16", "input": "640x640", "plain_ms": [10.0, 10.2, 9.9], "calls": 20, "profiled_wall_ms": 11.0,
        "gpu_busy_ms": 3.2, "kernels": [{"name": "gemm", "kind": "kernel", "count": 4, "ms": 2.0, "bound_ms": 1.0}],
        "env": {"cuda": "13.0", "torch": "2.14.1", "cudnn": "91002", "python": "3.12.3", "package": "x==1", "arch_list": ["sm_80", "sm_90"],
                "flags": {"matmul_allow_tf32": False, "cudnn_allow_tf32": True, "float32_matmul_precision": "highest", "cudnn_benchmark": False}},
    }
    rec = profile.build(
        model="x", size="s", weights=[{"file": "w.pt", "bytes": 1, "sha256": "a" * 64}], worker=worker, gpu_name="NVIDIA GeForce RTX 4090",
        driver="580.173.02", facts={"cpu": "c", "gpu_memory_mib": 24564, "compute_capability": "8.9", "os": "Linux", "power_limit_w": 450.0, "power_default_w": 450.0},
        provenance={"tool": {"name": "mtr", "version": "0.1.0", "commit": None, "dirty": None}, "adapter_sha256": "b" * 64, "lock_sha256": None, "input_sha256": ["c" * 64]},
        profiled="2026-11-06T16:00:00+09:00",
    )
    assert record.validate(rec, "profile-record") == []
    assert rec["profile"]["wall_ms"]["median"] == 10.0


def fake_adapter(**over):
    import types

    a = types.SimpleNamespace(
        SIZES=["s"], PRECISIONS=["fp16"], PACKAGE="pytest", INPUT={"s": "640x640"},
        load=lambda size, precision: None, predict=lambda m, x: x, weight_files=lambda size: [],
    )
    for k, v in over.items():
        setattr(a, k, v)
    return a


def test_phase_hooks_come_all_together_or_from_the_package():
    from mtr_measure import adapter as ad

    plain = fake_adapter()
    ad.check_interface(plain)
    assert ad.phase_source(plain) is None
    split = fake_adapter(preprocess=lambda m, x: x, forward=lambda m, i: i, postprocess=lambda m, r, x: r)
    ad.check_interface(split)
    assert ad.phase_source(split) == "adapter"
    # Fewer than three is no split: forward() alone is an adapter's own helper, as in several models here.
    helper = fake_adapter(forward=lambda m, x: x)
    ad.check_interface(helper)
    assert ad.phase_source(helper) is None and ad.phase_hooks(helper) == ["forward"]
    timed = fake_adapter(phase_times=lambda m, out: {"preprocess": 1.0, "forward": 2, "postprocess": 0.5})
    ad.check_interface(timed)
    assert ad.phase_source(timed) == "package"
    assert ad.package_phases(timed, None, None) == {"preprocess": 1.0, "forward": 2.0, "postprocess": 0.5}
    with pytest.raises(ad.AdapterError, match="either"):
        ad.check_interface(fake_adapter(preprocess=lambda m, x: x, forward=lambda m, i: i, postprocess=lambda m, r, x: r,
                                        phase_times=lambda m, out: {}))
    with pytest.raises(ad.AdapterError, match="phase_times"):
        ad.package_phases(fake_adapter(phase_times=lambda m, out: {"preprocess": 1.0, "inference": 2.0}), None, None)


def test_phase_calls_split_each_call():
    calls = []
    a = fake_adapter(preprocess=lambda m, x: calls.append("pre") or x, forward=lambda m, i: calls.append("fwd") or i,
                     postprocess=lambda m, r, x: calls.append("post") or r)
    syncs = []
    ph = profile.phase_calls(a, None, 1, lambda: syncs.append(1))
    assert calls == ["pre", "fwd", "post"] * profile.PHASED
    assert len(syncs) == 4 * profile.PHASED  # before the call and after each phase
    assert ph["source"] == "adapter" and len(ph["call"]) == profile.PHASED
    s = profile.summarize_phases({"source": "package", "preprocess": [1.0, 3.0, 2.0], "forward": [5.0, 5.0, 6.0],
                                  "postprocess": [0.5, 0.5, 0.7], "call": [9.0, 9.5, 10.0]})
    assert s == {"source": "package", "calls": 3, "preprocess_ms": 2.0, "forward_ms": 5.0, "postprocess_ms": 0.5, "call_ms": 9.5}
    assert profile.phase_calls(fake_adapter(), None, 1, lambda: None) is None


def test_record_with_phases_passes_its_schema():
    worker = {
        "precision": "fp16", "input": "640x640", "plain_ms": [10.0, 10.2, 9.9], "calls": 20, "profiled_wall_ms": 11.0,
        "gpu_busy_ms": 3.2, "kernels": [{"name": "gemm", "kind": "kernel", "count": 4, "ms": 2.0, "bound_ms": 1.0}],
        "phases": {"source": "adapter", "preprocess": [4.0, 4.2], "forward": [3.0, 3.1], "postprocess": [1.0, 1.1], "call": [8.5, 8.7]},
        "env": {"cuda": "13.0", "torch": "2.14.1", "cudnn": "91002", "python": "3.12.3", "package": "x==1", "arch_list": ["sm_89"],
                "flags": {"matmul_allow_tf32": False, "cudnn_allow_tf32": True, "float32_matmul_precision": "highest", "cudnn_benchmark": False}},
    }
    rec = profile.build(
        model="x", size="s", weights=[{"file": "w.pt", "bytes": 1, "sha256": "a" * 64}], worker=worker, gpu_name="NVIDIA GeForce RTX 4090",
        driver="580.173.02", facts={"cpu": "c", "gpu_memory_mib": 24564, "compute_capability": "8.9", "os": "Linux", "power_limit_w": 450.0, "power_default_w": 450.0},
        provenance={"tool": {"name": "mtr", "version": "0.1.0", "commit": None, "dirty": None}, "adapter_sha256": "b" * 64, "lock_sha256": None, "input_sha256": ["c" * 64]},
        profiled="2026-11-10T16:00:00+09:00",
    )
    assert record.validate(rec, "profile-record") == []
    assert rec["profile"]["phases"] == {"source": "adapter", "calls": 2, "preprocess_ms": 4.1, "forward_ms": 3.05, "postprocess_ms": 1.05, "call_ms": 8.6}
