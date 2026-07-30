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

"""Run each MX example case: generate data, launch the NPU kernel, verify output."""

import argparse
import csv
import os
import subprocess
import sys


def run(command, cwd=None):
    print("[CMD]", " ".join(command))
    return subprocess.run(command, cwd=cwd, check=False).returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("executable")
    parser.add_argument("csv_path")
    parser.add_argument("result_path")
    args = parser.parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    scripts_dir = os.path.join(os.path.dirname(script_dir), "scripts")
    data_root = os.path.join(script_dir, "data")
    os.makedirs(data_root, exist_ok=True)
    results = []

    with open(args.csv_path, newline="") as stream:
        for row in csv.DictReader(stream):
            case_dir = os.path.join(data_root, row["casename"])
            generate = [
                sys.executable, os.path.join(scripts_dir, "gen_data.py"),
                "--m", row["m"], "--k", row["k"], "--n", row["n"], "--bias", row["bias"],
                "--layout", row["layout"], "--output-dir", case_dir,
            ]
            status = "PASS"
            stage = "verify"
            message = ""
            if run(generate) != 0:
                status, stage, message = "FAIL", "gen_data", "data generation failed"
            else:
                launch = [args.executable, row["m"], row["k"], row["n"], row["bias"], row["layout"]]
                tiling_keys = (
                    "base_m", "base_n", "base_k", "tile_k_l1", "scale_k_l1", "k_bub", "n_bub",
                    "l1_buffers", "block_num")
                for key in tiling_keys:
                    launch.append(row[key])
                launch.append(case_dir)
                if run(launch) != 0:
                    status, stage, message = "FAIL", "kernel", "kernel execution failed"
                else:
                    verify = [sys.executable, os.path.join(scripts_dir, "verify_result.py"),
                              os.path.join(case_dir, "golden_c.bin"), os.path.join(case_dir, "npu_out.bin")]
                    if run(verify) != 0:
                        status, stage, message = "FAIL", "verify", "golden comparison failed"
            print(f"[{status}] {row['casename']}")
            results.append({"casename": row["casename"], "status": status, "stage": stage, "message": message})

    with open(args.result_path, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("casename", "status", "stage", "message"))
        writer.writeheader()
        writer.writerows(results)
    if any(item["status"] != "PASS" for item in results):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
