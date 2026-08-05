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
    1. gen_data.py - generate input and golden data
    2. kernel binary - run the NPU kernel
    3. verify_result.py - compare the NPU output with the golden data

Results are written to <csv>_result.csv in the same directory.
"""

import csv
import os
import subprocess
import sys


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(SCRIPT_DIR, "weight_quant_batch_matmul_mx.csv")
RESULT_CSV_PATH = os.path.join(SCRIPT_DIR, "weight_quant_batch_matmul_mx_result.csv")
SCRIPTS_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "scripts")
DATA_ROOT = os.path.join(SCRIPT_DIR, "data")

TILING_KEYS = (
    "base_m",
    "base_n",
    "base_k",
    "tile_k_l1",
    "scale_k_l1",
    "k_bub",
    "n_bub",
    "l1_buffers",
    "block_num",
)


def _log(tag, message):
    print(f"[{tag}] {message}")


def _run(command, timeout=300):
    _log("CMD", " ".join(command))
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout, check=False
        )
        output = (result.stdout or "") + (result.stderr or "")
        return result.returncode, output.strip()
    except subprocess.TimeoutExpired:
        return -1, "TIMEOUT"
    except Exception as error:
        return -1, str(error)


def parse_csv(csv_path):
    cases = []
    with open(csv_path, "r", newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            case = {
                "casename": row["casename"].strip(),
                "m": int(row["m"]),
                "k": int(row["k"]),
                "n": int(row["n"]),
                "bias": int(row["bias"]),
                "layout": row["layout"].strip().lower(),
            }
            for key in TILING_KEYS:
                case[key] = int(row[key])
            cases.append(case)
    return cases


def run_case(case, executable):
    case_dir = os.path.join(DATA_ROOT, case["casename"])

    gen_command = [
        sys.executable,
        os.path.join(SCRIPTS_DIR, "gen_data.py"),
        "--m",
        str(case["m"]),
        "--k",
        str(case["k"]),
        "--n",
        str(case["n"]),
        "--bias",
        str(case["bias"]),
        "--layout",
        case["layout"],
        "--output-dir",
        case_dir,
    ]
    return_code, output = _run(gen_command)
    if return_code != 0:
        return "FAIL", "gen_data", output

    run_command = [
        executable,
        str(case["m"]),
        str(case["k"]),
        str(case["n"]),
        str(case["bias"]),
        case["layout"],
    ]
    run_command.extend(str(case[key]) for key in TILING_KEYS)
    run_command.append(case_dir)
    return_code, output = _run(run_command)
    if return_code != 0:
        return "FAIL", "kernel", output

    verify_command = [
        sys.executable,
        os.path.join(SCRIPTS_DIR, "verify_result.py"),
        os.path.join(case_dir, "golden_c.bin"),
        os.path.join(case_dir, "npu_out.bin"),
    ]
    return_code, output = _run(verify_command)
    if return_code != 0:
        return "FAIL", "verify", output

    return "PASS", "verify", ""


def main():
    global CSV_PATH, RESULT_CSV_PATH

    executable_path = os.path.join(
        SCRIPT_DIR,
        "build",
        "quant_batch_matmul_mx",
        "weight_quant_batch_matmul_mx",
        "weight_quant_batch_matmul_mx",
    )
    if len(sys.argv) >= 4:
        executable_path = sys.argv[1]
        CSV_PATH = sys.argv[2]
        RESULT_CSV_PATH = sys.argv[3]
    elif len(sys.argv) != 1:
        print("Usage: parse_csv.py [executable csv_file result_file]")
        return 1

    if not os.path.isfile(CSV_PATH):
        print(f"Error: CSV file not found at {CSV_PATH}")
        return 1
    if not os.path.isfile(executable_path):
        print(f"Error: executable not found at {executable_path}")
        return 1

    cases = parse_csv(CSV_PATH)
    if not cases:
        print(f"No test cases found in {CSV_PATH}")
        return 0

    print(f"Found {len(cases)} test case(s) in {CSV_PATH}")
    print("=" * 70)

    results = []
    pass_count = 0
    fail_count = 0

    for index, case in enumerate(cases, 1):
        name = case["casename"]
        print(f"\n[{index}/{len(cases)}] Running: {name}")
        print(
            f"  Shape: {case['m']} x {case['k']} x {case['n']}, "
            f"layout={case['layout']}, bias={case['bias']}"
        )

        status, stage, message = run_case(case, executable_path)
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
            {"casename": name, "status": status, "stage": stage, "message": message}
        )

    with open(RESULT_CSV_PATH, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=("casename", "status", "stage", "message")
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
