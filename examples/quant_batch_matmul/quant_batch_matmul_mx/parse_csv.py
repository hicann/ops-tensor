#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

"""Parse CSV test cases and execute all cases sequentially.

Workflow per case:
    1. gen_data_mx.py – generate input/golden data
    2. kernel binary – run NPU kernel
    3. verify_result_mx.py – compare NPU output vs CPU golden

Results are written to <csv>_result.csv in the same directory.
"""

import csv
import os
import subprocess
import sys


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(SCRIPT_DIR, "quant_batch_matmul_mx.csv")
RESULT_CSV_PATH = os.path.join(SCRIPT_DIR, "quant_batch_matmul_mx_result.csv")
SCRIPTS_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "scripts")
BUILD_DIR = os.path.join(SCRIPT_DIR, "build")


def _log(tag, msg):
    print(f"[{tag}] {msg}")


def _run(cmd, cwd=None, timeout=300):
    """Run a subprocess, return (returncode, stdout+stderr)."""
    _log("CMD", " ".join(cmd))
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = (result.stdout or "") + (result.stderr or "")
        return result.returncode, output.strip()
    except subprocess.TimeoutExpired:
        return -1, "TIMEOUT"
    except Exception as exc:
        return -1, str(exc)


# ---------------------------------------------------------------------------
# CSV parsing
# ---------------------------------------------------------------------------


def parse_csv(csv_path):
    """Read CSV and return a list of case dicts."""
    cases = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cases.append(
                {
                    "casename": row["casename"].strip(),
                    "m": int(row["m"]),
                    "k": int(row["k"]),
                    "n": int(row["n"]),
                    "bias": int(row["bias"]),
                    "a_dtype": row["a_dtype"].strip(),
                    "b_dtype": row["b_dtype"].strip(),
                    "c_dtype": row["c_dtype"].strip(),
                    "transA": row["transA"].strip().lower() == "true",
                    "transB": row["transB"].strip().lower() == "true",
                    "base_m": int(row["base_m"]),
                    "base_n": int(row["base_n"]),
                    "base_k": int(row["base_k"]),
                    "tile_k_l1": int(row["tile_k_l1"]),
                    "scale_k_l1": int(row["scale_k_l1"]),
                    "l1_buffers": int(row["l1_buffers"]),
                    "db_l0c": int(row["db_l0c"]),
                    "a_full_load": row["a_full_load"].strip().lower() == "true",
                    "format": row["format"].strip().strip('"'),
                }
            )
    return cases


# ---------------------------------------------------------------------------
# Per-case execution
# ---------------------------------------------------------------------------


def _build_gen_cmd(case):
    """Build the gen_data_mx.py command for one case."""
    cmd = [
        "python3",
        os.path.join(SCRIPTS_DIR, "gen_data_mx.py"),
        "--m",
        str(case["m"]),
        "--k",
        str(case["k"]),
        "--n",
        str(case["n"]),
        "--bias",
        str(case["bias"]),
        "--a-dtype",
        case["a_dtype"],
        "--b-dtype",
        case["b_dtype"],
        "--c-dtype",
        case["c_dtype"],
        "--format",
        case["format"],
    ]
    if case["transA"]:
        cmd.append("--trans-a")
    if case["transB"]:
        cmd.append("--trans-b")
    cmd.extend(["--output-dir", os.path.join(SCRIPTS_DIR, "input")])
    return cmd


def _build_run_cmd(case):
    """Build the kernel executable command for one case."""
    exec_path = os.environ.get("PARSE_CSV_EXECUTABLE")
    if not exec_path:
        exec_path = os.path.join(
            BUILD_DIR,
            "quant_batch_matmul",
            "quant_batch_matmul_mx",
            "quant_batch_matmul_mx",
        )
    exec_path = os.path.abspath(exec_path)
    if not os.path.isfile(exec_path):
        return None, f"Executable not found: {exec_path}"
    cmd = [
        exec_path,
        str(case["m"]),
        str(case["k"]),
        str(case["n"]),
        str(case["bias"]),
        case["a_dtype"],
        case["b_dtype"],
        case["c_dtype"],
        str(case["transA"]).lower(),
        str(case["transB"]).lower(),
        case["format"],
        str(case["base_m"]),
        str(case["base_n"]),
        str(case["base_k"]),
        str(case["tile_k_l1"]),
        str(case["scale_k_l1"]),
        str(case["l1_buffers"]),
        str(case["db_l0c"]),
        str(case["a_full_load"]).lower(),
    ]
    return cmd, ""


def _build_verify_cmd(case):
    """Build the verify_result_mx.py command for one case."""
    return [
        "python3",
        os.path.join(SCRIPTS_DIR, "verify_result_mx.py"),
        os.path.join(SCRIPTS_DIR, "input", "golden_c.bin"),
        os.path.join(SCRIPTS_DIR, "output", "npu_out.bin"),
        "--dtype",
        case["c_dtype"],
    ]


def run_case(case):
    """Execute one test case through gen_data → kernel → verify.

    Returns (status, stage, message).
    """
    rc, out = _run(_build_gen_cmd(case), cwd=SCRIPTS_DIR)
    if rc != 0:
        return "FAIL", "gen_data", out

    run_cmd, err = _build_run_cmd(case)
    if run_cmd is None:
        return "FAIL", "kernel", err
    rc, out = _run(run_cmd, cwd=SCRIPTS_DIR)
    if rc != 0:
        return "FAIL", "kernel", out

    rc, out = _run(_build_verify_cmd(case), cwd=SCRIPTS_DIR)
    if rc != 0:
        return "FAIL", "verify", out

    return "PASS", "verify", ""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    global CSV_PATH, RESULT_CSV_PATH

    if len(sys.argv) >= 4:
        executable_path = sys.argv[1]
        csv_arg = sys.argv[2]
        result_arg = sys.argv[3]
        CSV_PATH = csv_arg
        RESULT_CSV_PATH = result_arg
        os.environ["PARSE_CSV_EXECUTABLE"] = executable_path
    elif len(sys.argv) == 1:
        pass
    else:
        print("Usage: parse_csv.py [executable csv_file result_file]")
        print("  No args: use default CSV and auto-discover executable")
        sys.exit(1)

    if not os.path.isfile(CSV_PATH):
        print(f"Error: CSV file not found at {CSV_PATH}")
        sys.exit(1)

    cases = parse_csv(CSV_PATH)
    if not cases:
        print("No test cases found in CSV")
        sys.exit(0)

    print(f"Found {len(cases)} test case(s)")
    print("=" * 70)

    results = []
    pass_count = 0
    fail_count = 0

    for i, case in enumerate(cases, 1):
        name = case["casename"]
        print(f"\n[{i}/{len(cases)}] Running: {name}")
        print(
            f"  Shape: {case['m']} x {case['k']} x {case['n']}, "
            f"a={case['a_dtype']} b={case['b_dtype']} c={case['c_dtype']}"
            f" transA={case['transA']} transB={case['transB']}"
        )

        status, stage, message = run_case(case)

        if status == "PASS":
            pass_count += 1
            _log("PASS", name)
        else:
            fail_count += 1
            _log("FAIL", f"{name} (stage={stage})")
            if message:
                preview = message[:500]
                if len(message) > 500:
                    preview += "..."
                print(f"  Error: {preview}")

        results.append(
            {
                "casename": name,
                "status": status,
                "stage": stage,
                "message": message,
            }
        )

    # -- Write result CSV ----------------------------------------------------
    with open(RESULT_CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["casename", "status", "stage", "message"]
        )
        writer.writeheader()
        writer.writerows(results)

    # -- Summary -------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  Summary")
    print("=" * 70)
    for r in results:
        marker = "PASS" if r["status"] == "PASS" else "FAIL"
        print(f"  [{marker}] {r['casename']}")
    print("-" * 70)
    print(f"  Total: {len(results)}  |  PASS: {pass_count}  |  FAIL: {fail_count}")
    print(f"  Results written to: {RESULT_CSV_PATH}")
    print("=" * 70)

    if fail_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
