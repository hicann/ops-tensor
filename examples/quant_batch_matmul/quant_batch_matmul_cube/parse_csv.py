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

"""Parse and execute quant_batch_matmul_cube CSV cases.

Workflow per case:
    1. gen_data_cube.py – generate input and golden data
    2. kernel binary – run the NPU kernel
    3. verify_result_cube.py – compare the NPU output with the golden data

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
SCRIPTS_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "scripts")
CSV_PATH = os.path.join(SCRIPT_DIR, "quant_batch_matmul_cube.csv")
RESULT_CSV_PATH = os.path.join(SCRIPT_DIR, "quant_batch_matmul_cube_result.csv")
DATA_ROOT = os.path.join(SCRIPT_DIR, "data")
BUILD_DIR = os.path.join(SCRIPT_DIR, "build")

CSV_FIELDS = (
    "batch",
    "M",
    "K",
    "N",
    "AType",
    "BType",
    "CType",
    "bias",
    "biasType",
    "transA",
    "transB",
    "x1quantmode",
    "x2quantmode",
    "x2ScaleType",
    "baseM",
    "baseN",
    "baseK",
    "kL1",
)


def _log(tag, message):
    print(f"[{tag}] {message}")


def _run(command, cwd=None, timeout=300):
    """Run a subprocess and return (returncode, stdout+stderr)."""
    _log("CMD", " ".join(command))
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        output = (result.stdout or "") + (result.stderr or "")
        return result.returncode, output.strip()
    except subprocess.TimeoutExpired:
        return -1, "TIMEOUT"
    except Exception as error:  # noqa: BLE001
        return -1, str(error)


def _parse_bool(value):
    normalized = value.strip().lower()
    if normalized in ("true", "1"):
        return True
    if normalized in ("false", "0"):
        return False
    raise ValueError(f"invalid boolean value: {value}")


# ---------------------------------------------------------------------------
# CSV parsing
# ---------------------------------------------------------------------------


def _parse_case(row):
    missing = [
        key for key in ("casename",) + CSV_FIELDS if key not in row or row[key] == ""
    ]
    if missing:
        raise ValueError("missing CSV fields: " + ", ".join(missing))

    case = {
        "casename": row["casename"].strip(),
        "batch": int(row["batch"]),
        "m": int(row["M"]),
        "k": int(row["K"]),
        "n": int(row["N"]),
        "a_type": row["AType"].strip(),
        "b_type": row["BType"].strip(),
        "c_type": row["CType"].strip(),
        "bias": int(row["bias"]),
        "bias_type": row["biasType"].strip(),
        "trans_a": _parse_bool(row["transA"]),
        "trans_b": _parse_bool(row["transB"]),
        "x1_quant_mode": row["x1quantmode"].strip(),
        "x2_quant_mode": row["x2quantmode"].strip(),
        "x2_scale_type": row["x2ScaleType"].strip(),
        "base_m": int(row["baseM"]),
        "base_n": int(row["baseN"]),
        "base_k": int(row["baseK"]),
        "k_l1": int(row["kL1"]),
    }
    if not (
        0 < case["base_m"] <= 256
        and 0 < case["base_n"] <= 256
        and 0 < case["base_k"] <= 128
    ):
        raise ValueError("baseM/baseK/baseN maximum is 256*128*256")
    return case


def parse_csv(csv_path):
    """Read a CSV file and return normalized case dictionaries."""
    cases = []
    with open(csv_path, "r", newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            cases.append(_parse_case(row))
    return cases


# ---------------------------------------------------------------------------
# Per-case execution
# ---------------------------------------------------------------------------


def _case_dir(case):
    return os.path.join(DATA_ROOT, case["casename"])


def _build_gen_cmd(case):
    return [
        "python3",
        os.path.join(SCRIPTS_DIR, "gen_data_cube.py"),
        "--batch",
        str(case["batch"]),
        "--m",
        str(case["m"]),
        "--k",
        str(case["k"]),
        "--n",
        str(case["n"]),
        "--bias",
        str(case["bias"]),
        "--a-type",
        case["a_type"],
        "--b-type",
        case["b_type"],
        "--c-type",
        case["c_type"],
        "--bias-type",
        case["bias_type"],
        "--trans-a",
        str(case["trans_a"]).lower(),
        "--trans-b",
        str(case["trans_b"]).lower(),
        "--x1-quant-mode",
        case["x1_quant_mode"],
        "--x2-quant-mode",
        case["x2_quant_mode"],
        "--x2-scale-type",
        case["x2_scale_type"],
        "--output-dir",
        _case_dir(case),
    ]


def _build_run_cmd(case):
    executable = os.environ.get("PARSE_CSV_EXECUTABLE")
    if not executable:
        executable = os.path.join(
            BUILD_DIR,
            "quant_batch_matmul",
            "quant_batch_matmul_cube",
            "quant_batch_matmul_cube",
        )
    executable = os.path.abspath(executable)
    if not os.path.isfile(executable):
        return None, f"Executable not found: {executable}"

    values = [
        case["batch"],
        case["m"],
        case["k"],
        case["n"],
        case["a_type"],
        case["b_type"],
        case["c_type"],
        case["bias"],
        case["bias_type"],
        str(case["trans_a"]).lower(),
        str(case["trans_b"]).lower(),
        case["x1_quant_mode"],
        case["x2_quant_mode"],
        case["x2_scale_type"],
        case["base_m"],
        case["base_n"],
        case["base_k"],
        case["k_l1"],
    ]
    return [executable] + [str(value) for value in values] + [_case_dir(case)], ""


def _build_verify_cmd(case):
    case_dir = _case_dir(case)
    return [
        "python3",
        os.path.join(SCRIPTS_DIR, "verify_result_cube.py"),
        os.path.join(case_dir, "golden_c.bin"),
        os.path.join(case_dir, "npu_out.bin"),
        "--batch",
        str(case["batch"]),
        "--m",
        str(case["m"]),
        "--n",
        str(case["n"]),
        "--dtype",
        case["c_type"],
    ]


def run_case(case):
    """Execute one case through gen_data, kernel, and verification."""
    return_code, output = _run(_build_gen_cmd(case), cwd=SCRIPTS_DIR)
    if return_code != 0:
        return "FAIL", "gen_data", output

    run_command, error = _build_run_cmd(case)
    if run_command is None:
        return "FAIL", "kernel", error
    return_code, output = _run(run_command, cwd=SCRIPT_DIR)
    if return_code != 0:
        return "FAIL", "kernel", output

    return_code, output = _run(_build_verify_cmd(case), cwd=SCRIPT_DIR)
    if return_code != 0:
        return "FAIL", "verify", output

    return "PASS", "verify", ""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    global CSV_PATH, RESULT_CSV_PATH

    if len(sys.argv) >= 4:
        os.environ["PARSE_CSV_EXECUTABLE"] = sys.argv[1]
        CSV_PATH = sys.argv[2]
        RESULT_CSV_PATH = sys.argv[3]
    elif len(sys.argv) != 1:
        print("Usage: parse_csv.py [executable csv_file result_file]")
        print("  No args: use default CSV and auto-discover executable")
        return 1

    if not os.path.isfile(CSV_PATH):
        print(f"Error: CSV file not found at {CSV_PATH}")
        return 1

    try:
        cases = parse_csv(CSV_PATH)
    except (KeyError, TypeError, ValueError) as error:
        print(f"Error: invalid CSV: {error}")
        return 1

    if not cases:
        print("No test cases found in CSV")
        return 0

    print(f"Found {len(cases)} test case(s)")
    print("=" * 70)

    results = []
    pass_count = 0
    fail_count = 0

    for index, case in enumerate(cases, 1):
        name = case["casename"]
        print(f"\n[{index}/{len(cases)}] Running: {name}")
        print(
            f"  Shape: batch={case['batch']}, "
            f"{case['m']} x {case['k']} x {case['n']}, "
            f"a={case['a_type']} b={case['b_type']} c={case['c_type']} "
            f"transA={case['trans_a']} transB={case['trans_b']}"
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

    with open(RESULT_CSV_PATH, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["casename", "status", "stage", "message"]
        )
        writer.writeheader()
        writer.writerows(results)

    print("\n" + "=" * 70)
    print("  Summary")
    print("=" * 70)
    for result in results:
        print(f"  [{result['status']}] {result['casename']}")
    print("-" * 70)
    print(f"  Total: {len(results)}  |  PASS: {pass_count}  |  FAIL: {fail_count}")
    print(f"  Results written to: {RESULT_CSV_PATH}")
    print("=" * 70)

    return 1 if fail_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
