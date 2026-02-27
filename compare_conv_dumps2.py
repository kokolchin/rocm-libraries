#!/usr/bin/env python3

# compare_conv_dumps2.py

#

# Compare CTest and GTest convolution config dumps.

#

# Usage:

#   python3 compare_conv_dumps2.py \

#     --ctest-dir /path/to/ctest_conv_dumps \

#     --gtest-dir /path/to/gtest_conv_dumps

#

# Expected filename patterns:

#   ctest_<name>--all.txt

#   ctest_<name>--all--limit1.txt

#   gtest_<name>--full.txt

#   gtest_<name>--smoke.txt
 
import argparse

import os

import re

import shlex

import sys

from collections import defaultdict
 
CANON_KEYS = [

    "input",

    "weights",

    "pads_strides_dilations",

    "trans_output_pads",

    "group_count",

    "cmode",

    "pmode",

    "in_layout",

    "fil_layout",

    "out_layout",

    "deterministic",

    "tensor_vect",

    "vector_length",

    "output_type",

    "int8_vectorize",

    "disable_forward",

    "disable_backward_data",

    "disable_backward_weights",

]
 
LIST_KEYS = {

    "input",

    "weights",

    "pads_strides_dilations",

    "trans_output_pads",

}
 
BOOL_KEYS = {

    "deterministic",

    "int8_vectorize",

    "disable_forward",

    "disable_backward_data",

    "disable_backward_weights",

}
 
FLAG_KEYS = {

    "input",

    "weights",

    "pads_strides_dilations",

    "trans_output_pads",

    "group_count",

    "cmode",

    "pmode",

    "in_layout",

    "fil_layout",

    "out_layout",

    "deterministic",

    "tensor_vect",

    "vector_length",

    "output_type",

    "int8_vectorize",

    "disable-forward",

    "disable-backward-data",

    "disable-backward-weights",

}
 
FILE_RE = re.compile(r"^(ctest|gtest)_(.+?)--(.+)\.txt$")
 
 
def normalize_key(k: str) -> str:

    return k.strip().lower().replace("-", "_")
 
 
def normalize_bool(v: str) -> str:

    t = v.strip().lower()

    if t in {"1", "true", "on", "yes"}:

        return "1"

    if t in {"0", "false", "off", "no"}:

        return "0"

    return t
 
 
def normalize_list(v: str) -> str:

    s = v.strip()

    s = s.replace("[", " ").replace("]", " ").replace("(", " ").replace(")", " ")

    parts = [p for p in re.split(r"[,\s]+", s) if p]

    return "x".join(parts)
 
 
def normalize_value(k: str, v: str) -> str:

    if k in LIST_KEYS:

        return normalize_list(v)

    if k in BOOL_KEYS:

        return normalize_bool(v)

    return v.strip()
 
 
def parse_gtest_cfg_line(line: str):

    # Format example:

    # GTEST_CFG|input=1,16,24,24|weights=16,16,7,7|pads_strides_dilations=3,3,1,1,1,1|...

    if "GTEST_CFG|" not in line:

        return None

    payload = line.split("GTEST_CFG|", 1)[1].strip()

    items = [x for x in payload.split("|") if x]

    out = {}

    for item in items:

        if "=" not in item:

            continue

        k, v = item.split("=", 1)

        k = normalize_key(k)

        if k in CANON_KEYS:

            out[k] = normalize_value(k, v)

    return out if out else None
 
 
def parse_kv_style_line(line: str):

    # Parse key=value occurrences in generic log lines.

    # Keeps only known keys.

    out = {}

    for m in re.finditer(r"([A-Za-z_][A-Za-z0-9_-]*)\s*=\s*([^\s|]+)", line):

        k = normalize_key(m.group(1))

        v = m.group(2)

        if k in CANON_KEYS:

            out[k] = normalize_value(k, v)

    return out if out else None
 
 
def parse_cli_style_line(line: str):

    # Parse lines containing --flags (CTest command-like lines).

    if "--" not in line:

        return None
 
    try:

        toks = shlex.split(line)

    except Exception:

        return None
 
    out = {}

    i = 0

    n = len(toks)
 
    while i < n:

        tok = toks[i]

        if not tok.startswith("--"):

            i += 1

            continue
 
        raw = tok[2:]

        if raw not in FLAG_KEYS:

            i += 1

            continue
 
        # Boolean switches with no value.

        if raw in {"disable-forward", "disable-backward-data", "disable-backward-weights"}:

            out[normalize_key(raw)] = "1"

            i += 1

            continue
 
        # Consume all subsequent non-flag tokens as value.

        j = i + 1

        vals = []

        while j < n and not toks[j].startswith("--"):

            vals.append(toks[j])

            j += 1
 
        if vals:

            k = normalize_key(raw)

            v = " ".join(vals)

            if k in CANON_KEYS:

                out[k] = normalize_value(k, v)
 
        i = j
 
    return out if out else None
 
 
def cfg_to_tuple(cfg: dict):

    return tuple((k, cfg.get(k, "")) for k in CANON_KEYS if k in cfg)
 
 
def parse_dump_file(path: str):

    configs = []

    with open(path, "r", encoding="utf-8", errors="replace") as f:

        for line in f:

            line = line.rstrip("\n")
 
            cfg = parse_gtest_cfg_line(line)

            if cfg:

                configs.append(cfg)

                continue
 
            cfg = parse_cli_style_line(line)

            if cfg:

                configs.append(cfg)

                continue
 
            cfg = parse_kv_style_line(line)

            if cfg:

                configs.append(cfg)

                continue
 
    # Deduplicate

    uniq = []

    seen = set()

    for c in configs:

        t = cfg_to_tuple(c)

        if t and t not in seen:

            seen.add(t)

            uniq.append(c)

    return uniq
 
 
def normalize_variant(framework: str, variant: str):

    v = variant.lower()

    if framework == "gtest":

        if v == "full":

            return "full"

        if v == "smoke":

            return "smoke"

    if framework == "ctest":

        if v == "all":

            return "full"

        if v in {"all--limit1", "all-limit1", "limit1"}:

            return "smoke"

    return None
 
 
def load_index(dump_dir: str):

    idx = {}

    for name in os.listdir(dump_dir):

        m = FILE_RE.match(name)

        if not m:

            continue

        framework, test_name, variant = m.groups()

        nv = normalize_variant(framework, variant)

        if nv is None:

            continue

        key = (framework, test_name, nv)

        idx[key] = os.path.join(dump_dir, name)

    return idx
 
 
def tuple_set(cfgs):

    return {cfg_to_tuple(c) for c in cfgs if cfg_to_tuple(c)}
 
 
def format_cfg_tuple(t):

    parts = [f"{k}={v}" for k, v in t if v != ""]

    return ", ".join(parts)
 
 
def compare_pair(name, variant, ctest_file, gtest_file):

    c_cfg = parse_dump_file(ctest_file)

    g_cfg = parse_dump_file(gtest_file)
 
    c_set = tuple_set(c_cfg)

    g_set = tuple_set(g_cfg)
 
    only_c = sorted(c_set - g_set)

    only_g = sorted(g_set - c_set)
 
    ok = not only_c and not only_g

    print(f"\n[{name} | {variant}]")

    print(f"  ctest: {len(c_set)} unique configs")

    print(f"  gtest: {len(g_set)} unique configs")

    if ok:

        print("  status: OK (exact match)")

        return True
 
    print("  status: MISMATCH")

    print(f"  only in ctest: {len(only_c)}")

    print(f"  only in gtest: {len(only_g)}")
 
    max_show = 5

    if only_c:

        print("  examples only in ctest:")

        for t in only_c[:max_show]:

            print(f"    - {format_cfg_tuple(t)}")

    if only_g:

        print("  examples only in gtest:")

        for t in only_g[:max_show]:

            print(f"    - {format_cfg_tuple(t)}")

    return False
 
 
def main():

    ap = argparse.ArgumentParser()

    ap.add_argument("--ctest-dir", required=True)

    ap.add_argument("--gtest-dir", required=True)

    args = ap.parse_args()
 
    c_idx = load_index(args.ctest_dir)

    g_idx = load_index(args.gtest_dir)
 
    # Compare intersection of names+variants available in both dirs.

    c_keys = {(k[1], k[2]) for k in c_idx.keys() if k[0] == "ctest"}

    g_keys = {(k[1], k[2]) for k in g_idx.keys() if k[0] == "gtest"}

    common = sorted(c_keys & g_keys)
 
    if not common:

        print("No comparable dump pairs found.")

        print("Check naming: ctest_<name>--all(.txt)/--all--limit1.txt and gtest_<name>--full.txt/--smoke.txt")

        return 2
 
    all_ok = True

    for name, variant in common:

        c_file = c_idx[("ctest", name, variant)]

        g_file = g_idx[("gtest", name, variant)]

        ok = compare_pair(name, variant, c_file, g_file)

        all_ok = all_ok and ok
 
    # Report missing pairs as warning.

    only_ctest = sorted(c_keys - g_keys)

    only_gtest = sorted(g_keys - c_keys)

    if only_ctest:

        print("\nWarning: present only in ctest dumps:")

        for nv in only_ctest:

            print(f"  - {nv[0]} | {nv[1]}")

    if only_gtest:

        print("\nWarning: present only in gtest dumps:")

        for nv in only_gtest:

            print(f"  - {nv[0]} | {nv[1]}")
 
    return 0 if all_ok else 1
 
 
if __name__ == "__main__":

    sys.exit(main())
 
