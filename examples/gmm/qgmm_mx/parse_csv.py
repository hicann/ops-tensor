#!/usr/bin/env python3

# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ============================================================================

import csv
import os
import shutil
import subprocess
import sys

SUPPORTED_DTYPES = ("mxfp4_e2m1", "mxfp4_e1m2", "mxfp8_e4m3", "mxfp8_e5m2")
SUPPORTED_GROUP_LISTS = ("length", "offset", "sparse")


def _build_command(executable, case):
    trans_a = case["transA"].strip().lower() == "true"
    trans_b = case["transB"].strip().lower() == "true"
    with_bias = case["bias"].strip().lower() == "true"
    group_list_type = case["group_list_type"].strip().lower()
    base_k = int(case["base_k"])
    tile_k_l1 = int(case["tile_k_l1"])
    scale_k_l1 = int(case["scale_k_l1"])
    l1_buffers = int(case["l1_buffers"])
    db_l0c = int(case["db_l0c"])
    a_full_load = case["a_full_load"].strip().lower() == "true"
    layout_pair = [
        item.strip() for item in case["format"].strip().strip('"()').lower().split(",")
    ]
    expected_b = layout_pair[-1]
    kernel_layout_b = (
        ("zn" if trans_b else "nz")
        if expected_b == "nz"
        else ("dn" if trans_b else "nd")
    )
    supported = (
        case["dtype"].strip() in SUPPORTED_DTYPES
        and len(layout_pair) == 2
        and layout_pair == ["nd", expected_b]
        and expected_b in ("nd", "nz")
        and case["weight_mode"].strip() in ("single", "multi")
        and (not with_bias or case["dtype"].strip().startswith("mxfp4"))
        and (case["dtype"].strip() != "mxfp4_e1m2" or layout_pair[1] in ("nz", "zn"))
        and group_list_type in SUPPORTED_GROUP_LISTS
        and base_k > 0
        and tile_k_l1 > 0
        and scale_k_l1 > 0
        and l1_buffers in (2, 3)
        and db_l0c in (1, 2)
        and all(int(case[dim]) > 0 for dim in ("e", "m", "n", "k"))
        and (
            not trans_a
            or (
                case["dtype"].strip() in ("mxfp8_e4m3", "mxfp8_e5m2")
                and layout_pair == ["nd", "nd"]
                and case["weight_mode"].strip() == "single"
                and not with_bias
                and group_list_type in ("length", "offset")
            )
        )
    )
    if not supported:
        return None
    return [
        executable,
        case["dtype"].strip(),
        kernel_layout_b,
        case["weight_mode"].strip(),
        *(case[dim].strip() for dim in ("e", "m", "n", "k")),
        str(trans_a).lower(),
        str(with_bias).lower(),
        group_list_type,
        str(base_k),
        str(tile_k_l1),
        str(scale_k_l1),
        str(l1_buffers),
        str(db_l0c),
        str(a_full_load).lower(),
    ]


def _run(command):
    return subprocess.run(command, check=False, capture_output=True, text=True)


def _print_details(completed):
    details = "\n".join(
        part.strip() for part in (completed.stdout, completed.stderr) if part.strip()
    )
    if details:
        print(details)
    return details


def _generation_command(scripts_dir, command, case_dir, case):
    generation = [
        sys.executable,
        os.path.join(scripts_dir, "gen_data.py"),
        "--dtype",
        command[1],
        "--weight-format",
        command[2],
        "--e",
        command[4],
        "--m",
        command[5],
        "--n",
        command[6],
        "--k",
        command[7],
        "--group-list-type",
        command[10],
        "--group-list",
        (case.get("group_list") or "").strip(),
        "--output-dir",
        case_dir,
    ]
    if command[8] == "true":
        generation.append("--trans-a")
    if command[3] == "multi":
        generation.append("--multi-tensor")
    if command[9] == "true":
        generation.append("--with-bias")
    return generation


def _failure(case_name, stage, message):
    print(f"[FAIL] {case_name}: {message}")
    return {"casename": case_name, "status": "FAIL", "stage": stage, "message": message}


def _run_case(executable, scripts_dir, output_root, case):
    case_name = case["casename"].strip()
    print(f"[RUN] {case_name}")
    command = _build_command(executable, case)
    if command is None:
        return _failure(case_name, "validate", "case is not supported")
    case_dir = os.path.join(output_root, case_name)
    shutil.rmtree(case_dir, ignore_errors=True)
    os.makedirs(case_dir)
    completed = _run(_generation_command(scripts_dir, command, case_dir, case))
    details = _print_details(completed)
    if completed.returncode != 0:
        return _failure(case_name, "gen_data", details or "data generation failed")
    output_path = os.path.join(case_dir, "npu_out.bin")
    completed = _run([*command, case_dir, output_path])
    details = _print_details(completed)
    if completed.returncode != 0:
        return _failure(case_name, "kernel", details or "kernel execution failed")
    completed = _run(
        [
            sys.executable,
            os.path.join(scripts_dir, "verify_result.py"),
            os.path.join(case_dir, "golden_c.bin"),
            output_path,
            "--groups",
            command[4],
            "--m",
            command[5],
            "--n",
            command[6],
        ]
    )
    details = _print_details(completed)
    if completed.returncode != 0:
        return _failure(case_name, "verify", details or "golden comparison failed")
    print(f"[PASS] {case_name}")
    return {"casename": case_name, "status": "PASS", "stage": "verify", "message": ""}


def _write_results(result_path, results):
    with open(result_path, "w", newline="", encoding="utf-8") as result_file:
        writer = csv.DictWriter(
            result_file, fieldnames=["casename", "status", "stage", "message"]
        )
        writer.writeheader()
        writer.writerows(results)


def main():
    if len(sys.argv) != 4:
        print("Usage: parse_csv.py executable csv_file result_file")
        return 1
    executable, csv_path, result_path = sys.argv[1:]
    script_dir = os.path.dirname(os.path.abspath(__file__))
    scripts_dir = os.path.join(os.path.dirname(script_dir), "scripts")
    output_root = os.path.join(script_dir, "output")
    os.makedirs(output_root, exist_ok=True)
    with open(csv_path, "r", newline="", encoding="utf-8") as csv_file:
        results = [
            _run_case(executable, scripts_dir, output_root, case)
            for case in csv.DictReader(csv_file)
        ]
    _write_results(result_path, results)
    return 1 if any(result["status"] == "FAIL" for result in results) else 0


if __name__ == "__main__":
    sys.exit(main())
