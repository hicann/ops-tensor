#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# ----------------------------------------------------------------------------

"""Parse CSV test cases and execute all cases sequentially."""

import csv
import os
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(SCRIPT_DIR, "testcase.csv")
RESULT_CSV_PATH = os.path.join(SCRIPT_DIR, "testcase_result.csv")
SCRIPTS_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "scripts")
BUILD_DIR = os.path.join(SCRIPT_DIR, "build")


def _log(tag, msg):
    print(f"[{tag}] {msg}")


def _run(cmd, cwd=None, timeout=300):
    _log("CMD", " ".join(cmd))
    try:
        result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
        output = (result.stdout or "") + (result.stderr or "")
        return result.returncode, output.strip()
    except subprocess.TimeoutExpired:
        return -1, "TIMEOUT"
    except Exception as exc:
        return -1, str(exc)


def parse_csv(csv_path):
    cases = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cases.append({
                "casename": row["casename"].strip(),
                "m": int(row["m"]),
                "k": int(row["k"]),
                "n": int(row["n"]),
                "batch": int(row["batch"]),
                "batchA": int(row["batchA"]),
                "batchB": int(row["batchB"]),
                "bias": int(row["bias"]),
                "dtype": row["dtype"].strip(),
                "transA": row["transA"].strip().lower() == "true",
                "transB": row["transB"].strip().lower() == "true",
                "hf32": row["hf32"].strip().lower() == "true",
            })
    return cases


def run_case(case):
    m = case["m"]
    k = case["k"]
    n = case["n"]
    batch = case["batch"]
    batch_a = case["batchA"]
    batch_b = case["batchB"]
    trans_a = str(case["transA"]).lower()
    trans_b = str(case["transB"]).lower()
    dtype = case["dtype"]
    bias = str(case["bias"])

    gen_cmd = [
        "python3", os.path.join(SCRIPTS_DIR, "gen_data_batch.py"),
        str(m), str(k), str(n), str(batch_a), str(batch_b), str(batch),
        trans_a, trans_b, dtype, bias,
    ]
    rc, out = _run(gen_cmd, cwd=SCRIPTS_DIR)
    if rc != 0:
        return "FAIL", "gen_data", out

    exec_path = os.environ.get("PARSE_CSV_EXECUTABLE")
    if not exec_path:
        exec_path = os.path.join(BUILD_DIR, "mat_mul", "mat_mul_bmm_broadcast", "mat_mul_bmm_broadcast")
    if not os.path.isfile(exec_path):
        return "FAIL", "kernel", f"Executable not found: {exec_path}"

    hf32 = str(case["hf32"]).lower()

    run_cmd = [
        exec_path, str(m), str(k), str(n), str(batch), str(batch_a), str(batch_b),
        trans_a, trans_b, dtype, hf32, bias,
    ]
    rc, out = _run(run_cmd, cwd=SCRIPTS_DIR)
    if rc != 0:
        return "FAIL", "kernel", out

    verify_cmd = [
        "python3", os.path.join(SCRIPTS_DIR, "verify_result_batch.py"),
        str(m), str(n), str(batch), dtype,
    ]
    if case.get("hf32"):
        verify_cmd.append("--hf32")
    rc, out = _run(verify_cmd, cwd=SCRIPTS_DIR)
    if rc != 0:
        return "FAIL", "verify", out

    return "PASS", "verify", ""


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
        sys.exit(1)

    if not os.path.isfile(CSV_PATH):
        print(f"Error: testcase.csv not found at {CSV_PATH}")
        sys.exit(1)

    cases = parse_csv(CSV_PATH)
    if not cases:
        print("No test cases found in testcase.csv")
        sys.exit(0)

    print(f"Found {len(cases)} test case(s) in testcase.csv")
    print("=" * 70)

    results = []
    pass_count = 0
    fail_count = 0

    for i, case in enumerate(cases, 1):
        name = case["casename"]
        print(f"\n[{i}/{len(cases)}] Running: {name}")
        print(f"  Shape: {case['m']} x {case['k']} x {case['n']}, "
              f"batch={case['batch']}, batchA={case['batchA']}, batchB={case['batchB']}, "
              f"dtype={case['dtype']}")

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

        results.append({"casename": name, "status": status, "stage": stage, "message": message})

    with open(RESULT_CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["casename", "status", "stage", "message"])
        writer.writeheader()
        writer.writerows(results)

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
