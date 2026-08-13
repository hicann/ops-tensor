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
SUPPORTED_LAYOUT_B = ("nd", "dn", "nz", "zn")
TILING_FIELDS = (
    "groupNum",
    "m",
    "n",
    "k",
    "baseM",
    "baseN",
    "baseK",
    "kAL1",
    "kBL1",
    "scaleKAL1",
    "scaleKBL1",
    "isBias",
    "dbL0C",
    "l1BufferStage",
    "groupType",
    "groupListType",
    "singleW",
)


def _build_command(executable, case):
    values = {field: int(case[field]) for field in TILING_FIELDS}
    dtype = case["dtype"].strip()
    layout_a = case["layoutA"].strip().lower()
    layout_b = case["layoutB"].strip().lower()
    a_full_load = int(case["aFullLoad"])
    k_grouped = values["groupType"] == 2
    supported = (
        dtype in SUPPORTED_DTYPES
        and layout_a in ("nd", "dn")
        and layout_b in SUPPORTED_LAYOUT_B
        and values["groupType"] in (0, 2)
        and ((layout_a == "dn") == k_grouped)
        and values["groupListType"] in (0, 1, 2)
        and values["isBias"] in (0, 1)
        and values["singleW"] in (0, 1)
        and values["dbL0C"] in (1, 2)
        and values["l1BufferStage"] in (2, 3)
        and a_full_load in (0, 1)
        and all(values[field] > 0 for field in TILING_FIELDS[:11])
        and (values["isBias"] == 0 or dtype.startswith("mxfp4"))
        and (dtype != "mxfp4_e1m2" or layout_b in ("nz", "zn"))
        and bool(case["groupList"].strip())
        and (
            not k_grouped
            or (
                dtype in ("mxfp8_e4m3", "mxfp8_e5m2")
                and layout_b == "nd"
                and values["singleW"] == 1
                and values["isBias"] == 0
                and values["groupListType"] in (0, 1)
            )
        )
    )
    if not supported:
        return None
    return [
        executable,
        *(case[field].strip() for field in TILING_FIELDS),
        dtype,
        layout_a,
        layout_b,
        str(a_full_load),
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


def _generation_command(scripts_dir, case_dir, case):
    generation = [
        sys.executable,
        os.path.join(scripts_dir, "gen_data.py"),
        "--dtype",
        case["dtype"].strip(),
        "--layout-b",
        case["layoutB"].strip().lower(),
        "--group-num",
        case["groupNum"].strip(),
        "--m",
        case["m"].strip(),
        "--n",
        case["n"].strip(),
        "--k",
        case["k"].strip(),
        "--group-list-type",
        case["groupListType"].strip(),
        "--group-type",
        case["groupType"].strip(),
        "--single-w",
        case["singleW"].strip(),
        "--is-bias",
        case["isBias"].strip(),
        "--group-list",
        (case.get("groupList") or "").strip(),
        "--output-dir",
        case_dir,
    ]
    return generation


def _failure(case_name, stage, message):
    print(f"[FAIL] {case_name}: {message}")
    return {"caseName": case_name, "status": "FAIL", "stage": stage, "message": message}


def _run_case(executable, scripts_dir, output_root, case):
    case_name = case["caseName"].strip()
    print(f"[RUN] {case_name}")
    command = _build_command(executable, case)
    if command is None:
        return _failure(case_name, "validate", "case is not supported")
    case_dir = os.path.join(output_root, case_name)
    shutil.rmtree(case_dir, ignore_errors=True)
    os.makedirs(case_dir)
    completed = _run(_generation_command(scripts_dir, case_dir, case))
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
            case["groupNum"].strip(),
            "--m",
            case["m"].strip(),
            "--n",
            case["n"].strip(),
        ]
    )
    details = _print_details(completed)
    if completed.returncode != 0:
        return _failure(case_name, "verify", details or "golden comparison failed")
    print(f"[PASS] {case_name}")
    return {"caseName": case_name, "status": "PASS", "stage": "verify", "message": ""}


def _write_results(result_path, results):
    with open(result_path, "w", newline="", encoding="utf-8") as result_file:
        writer = csv.DictWriter(
            result_file, fieldnames=["caseName", "status", "stage", "message"]
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
