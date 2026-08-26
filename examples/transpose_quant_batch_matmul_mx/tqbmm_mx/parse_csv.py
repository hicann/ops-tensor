#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# ----------------------------------------------------------------------------

"""Parse CSV test cases and execute transpose quantized batch matmul MX cases."""

import csv
import os
import subprocess
import sys


DEFAULT_PARAMS = {
    "bias": "0",
    "a_dtype": "fp8_e4m3",
    "b_dtype": "fp8_e4m3",
    "c_dtype": "bfloat16",
    "transA": "false",
    "transB": "false",
    "format": "(ND,ND)",
    "base_m": "128",
    "base_n": "256",
    "base_k": "64",
    "tile_k_l1": "64",
    "scale_k_l1": "64",
    "l1_buffers": "2",
    "db_l0c": "1",
    "a_full_load": "false",
}


def parse_csv(csv_path):
    cases = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cases.append(row)
    return cases


def build_exec_args(case):
    m, n, k = case["M"], case["N"], case["K"]
    base_m = case.get("base_m", DEFAULT_PARAMS["base_m"])
    base_n = case.get("base_n", DEFAULT_PARAMS["base_n"])
    base_k = case.get("base_k", DEFAULT_PARAMS["base_k"])
    return [
        m,
        n,
        k,
        DEFAULT_PARAMS["bias"],
        DEFAULT_PARAMS["a_dtype"],
        DEFAULT_PARAMS["b_dtype"],
        DEFAULT_PARAMS["c_dtype"],
        DEFAULT_PARAMS["transA"],
        DEFAULT_PARAMS["transB"],
        DEFAULT_PARAMS["format"],
        base_m,
        base_n,
        base_k,
        DEFAULT_PARAMS["tile_k_l1"],
        DEFAULT_PARAMS["scale_k_l1"],
        DEFAULT_PARAMS["l1_buffers"],
        DEFAULT_PARAMS["db_l0c"],
        DEFAULT_PARAMS["a_full_load"],
    ]


def main():
    if len(sys.argv) < 2:
        script_name = os.path.basename(sys.argv[0]) if sys.argv else "parse_csv.py"
        print(f"Usage: {script_name} <executable> [csv_file] [result_file]")
        sys.exit(1)

    executable = sys.argv[1]
    csv_file = sys.argv[2] if len(sys.argv) > 2 else "tqbmm_mx.csv"
    result_file = (
        sys.argv[3] if len(sys.argv) > 3 else csv_file.replace(".csv", "_result.csv")
    )

    cases = parse_csv(csv_file)
    results = []

    script_dir = os.path.dirname(os.path.abspath(__file__))
    gen_data = os.path.join(script_dir, "..", "scripts", "gen_data.py")
    verify_result = os.path.join(script_dir, "..", "scripts", "verify_result.py")
    tqbmm_dir = os.path.dirname(executable)

    for idx, case in enumerate(cases):
        case_result = {"index": idx, "status": "PASS", "message": ""}
        try:
            subprocess.run(
                [
                    sys.executable,
                    gen_data,
                    case["M"],
                    case["N"],
                    case["K"],
                    case["Batch"],
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            exec_args = build_exec_args(case)
            subprocess.run(
                [executable] + exec_args,
                check=True,
                capture_output=True,
                text=True,
                cwd=tqbmm_dir,
            )
            subprocess.run(
                [
                    sys.executable,
                    verify_result,
                    case["M"],
                    case["N"],
                    case["K"],
                    case["Batch"],
                    "--output",
                    "output/npu_out.bin",
                ],
                check=True,
                capture_output=True,
                text=True,
                cwd=tqbmm_dir,
            )
        except subprocess.CalledProcessError as e:
            case_result["status"] = "FAIL"
            case_result["message"] = str(e)
        results.append(case_result)

    with open(result_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["index", "status", "message"])
        writer.writeheader()
        writer.writerows(results)

    failed = sum(1 for r in results if r["status"] != "PASS")
    print(f"Total: {len(results)}, Pass: {len(results) - failed}, Fail: {failed}")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
