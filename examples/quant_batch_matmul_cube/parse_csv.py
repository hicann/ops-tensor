#!/usr/bin/env python3
# coding=utf-8

# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Generate, run, and verify every quant_batch_matmul_cube CSV case."""

import argparse
import csv
import os
import subprocess
import sys


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


def run(command):
    print("[CMD]", " ".join(command))
    return subprocess.run(command, check=False).returncode


def validate_row(row):
    missing = [
        key for key in ("casename",) + CSV_FIELDS if key not in row or row[key] == ""
    ]
    if missing:
        raise ValueError("missing CSV fields: " + ", ".join(missing))
    base_m, base_n, base_k = int(row["baseM"]), int(row["baseN"]), int(row["baseK"])
    if not (0 < base_m <= 256 and 0 < base_n <= 256 and 0 < base_k <= 128):
        raise ValueError("baseM/baseK/baseN maximum is 256*128*256")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("executable")
    parser.add_argument("csv_path")
    parser.add_argument("result_path")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_root = os.path.join(script_dir, "data")
    os.makedirs(data_root, exist_ok=True)
    results = []

    with open(args.csv_path, newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            status, stage, message = "PASS", "verify", ""
            try:
                validate_row(row)
            except ValueError as error:
                status, stage, message = "FAIL", "csv", str(error)
                print(f"[{status}] {row.get('casename', '<unnamed>')}: {message}")
                results.append(
                    {
                        "casename": row.get("casename", ""),
                        "status": status,
                        "stage": stage,
                        "message": message,
                    }
                )
                continue

            case_dir = os.path.join(data_root, row["casename"])
            generate = [
                sys.executable,
                os.path.join(script_dir, "gen_data.py"),
                "--batch",
                row["batch"],
                "--m",
                row["M"],
                "--k",
                row["K"],
                "--n",
                row["N"],
                "--bias",
                row["bias"],
                "--a-type",
                row["AType"],
                "--b-type",
                row["BType"],
                "--c-type",
                row["CType"],
                "--bias-type",
                row["biasType"],
                "--trans-a",
                row["transA"],
                "--trans-b",
                row["transB"],
                "--x1-quant-mode",
                row["x1quantmode"],
                "--x2-quant-mode",
                row["x2quantmode"],
                "--x2-scale-type",
                row["x2ScaleType"],
                "--output-dir",
                case_dir,
            ]
            if run(generate) != 0:
                status, stage, message = "FAIL", "gen_data", "data generation failed"
            else:
                launch = (
                    [args.executable] + [row[key] for key in CSV_FIELDS] + [case_dir]
                )
                if run(launch) != 0:
                    status, stage, message = "FAIL", "kernel", "kernel execution failed"
                else:
                    verify = [
                        sys.executable,
                        os.path.join(script_dir, "verify_result.py"),
                        os.path.join(case_dir, "golden_c.bin"),
                        os.path.join(case_dir, "npu_out.bin"),
                        "--batch",
                        row["batch"],
                        "--m",
                        row["M"],
                        "--n",
                        row["N"],
                        "--dtype",
                        row["CType"],
                    ]
                    if run(verify) != 0:
                        status, stage, message = (
                            "FAIL",
                            "verify",
                            "golden comparison failed",
                        )
            print(f"[{status}] {row['casename']}")
            results.append(
                {
                    "casename": row["casename"],
                    "status": status,
                    "stage": stage,
                    "message": message,
                }
            )

    with open(args.result_path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=("casename", "status", "stage", "message")
        )
        writer.writeheader()
        writer.writerows(results)
    return 1 if any(item["status"] != "PASS" for item in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
