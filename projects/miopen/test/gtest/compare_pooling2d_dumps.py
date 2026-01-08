#!/usr/bin/env python3
"""
Compare configuration dumps from ctest and gtest variants of pooling2d tests.

Both ctest and gtest now use the same space-separated format:
"1 19 1024 2048 2 2 0 0 1 1 0 0 0" (13 space-separated values)
Format: input_dims[4] lens[2] pads[2] strides[2] index_type mode wsidx

Supports comparing all three datasets:
- Dataset 0: Standard pooling (pooling2d)
- Dataset 1: Asymmetric pooling (pooling2d_asymmetric)
- Dataset 2: Wide window pooling (pooling2d_wide)
"""

import sys
import os
from collections import defaultdict

def parse_config_line(line):
    """Parse space-separated format: 1 19 1024 2048 2 2 0 0 1 1 0 0 0"""
    # Skip comments and empty lines
    line = line.strip()
    if not line or line.startswith('#'):
        return None
    
    parts = line.split()
    if len(parts) != 13:
        return None
    
    try:
        return {
            'input_dims': (int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])),
            'lens': (int(parts[4]), int(parts[5])),
            'pads': (int(parts[6]), int(parts[7])),
            'strides': (int(parts[8]), int(parts[9])),
            'index_type': int(parts[10]),
            'mode': int(parts[11]),
            'wsidx': int(parts[12])
        }
    except (ValueError, IndexError):
        return None

def normalize_config(config):
    """Create a normalized tuple for comparison"""
    return (
        config['input_dims'],
        config['lens'],
        config['pads'],
        config['strides'],
        config['index_type'],
        config['mode'],
        config['wsidx']
    )

def load_configs(filename):
    """Load configurations from a file (both ctest and gtest use same format now)"""
    configs = []
    if not os.path.exists(filename):
        return set()
    
    with open(filename, 'r') as f:
        for line_num, line in enumerate(f, 1):
            config = parse_config_line(line)
            if config:
                configs.append(normalize_config(config))
            elif line.strip() and not line.strip().startswith('#'):
                print(f"Warning: Could not parse line {line_num} in {filename}: {line[:80]}")
    
    return set(configs)  # Use set for comparison

def compare_files(ctest_file, gtest_file, dataset_name=""):
    """Compare two configuration files"""
    if dataset_name:
        print(f"\n{'='*60}")
        print(f"DATASET: {dataset_name}")
        print(f"{'='*60}")
    
    print(f"\nLoading ctest configs from: {ctest_file}")
    ctest_configs = load_configs(ctest_file)
    print(f"  Loaded {len(ctest_configs)} unique configurations")
    
    print(f"\nLoading gtest configs from: {gtest_file}")
    gtest_configs = load_configs(gtest_file)
    print(f"  Loaded {len(gtest_configs)} unique configurations")
    
    # Find differences
    only_in_ctest = ctest_configs - gtest_configs
    only_in_gtest = gtest_configs - ctest_configs
    in_both = ctest_configs & gtest_configs
    
    print(f"\nConfigurations in both: {len(in_both)}")
    print(f"Configurations only in ctest: {len(only_in_ctest)}")
    print(f"Configurations only in gtest: {len(only_in_gtest)}")
    
    if len(only_in_ctest) == 0 and len(only_in_gtest) == 0:
        print("\n✓ SUCCESS: Configurations are IDENTICAL!")
        print(f"  Total: {len(ctest_configs)} configurations match")
        return True
    else:
        print("\n✗ DIFFERENCES FOUND:")
        
        if only_in_ctest:
            print(f"\n  {len(only_in_ctest)} configurations only in ctest:")
            for i, config in enumerate(sorted(only_in_ctest)[:12], 1):
                print(f"    {i}. {config}")
            if len(only_in_ctest) > 12:
                print(f"    ... and {len(only_in_ctest) - 12} more")
        
        if only_in_gtest:
            print(f"\n  {len(only_in_gtest)} configurations only in gtest:")
            for i, config in enumerate(sorted(only_in_gtest)[:12], 1):
                print(f"    {i}. {config}")
            if len(only_in_gtest) > 12:
                print(f"    ... and {len(only_in_gtest) - 12} more")
        
        # Statistics
        print("\n" + "-"*60)
        print("STATISTICS")
        print("-"*60)
        
        # Count by input shape
        ctest_shapes = defaultdict(int)
        for config in ctest_configs:
            ctest_shapes[config[0]] += 1
        
        gtest_shapes = defaultdict(int)
        for config in gtest_configs:
            gtest_shapes[config[0]] += 1
        
        print("\nConfigurations per input shape:")
        all_shapes = set(ctest_shapes.keys()) | set(gtest_shapes.keys())
        for shape in sorted(all_shapes):
            ctest_count = ctest_shapes.get(shape, 0)
            gtest_count = gtest_shapes.get(shape, 0)
            match = "✓" if ctest_count == gtest_count else "✗"
            print(f"  {match} {shape}: ctest={ctest_count}, gtest={gtest_count}")
        
        return False

def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python3 compare_pooling2d_dumps.py <mode> [args...]")
        print()
        print("Modes:")
        print("  single <ctest_file> <gtest_file>")
        print("    Compare two specific files")
        print()
        print("  all [directory]")
        print("    Compare all three datasets (default: current directory)")
        print("    Looks for:")
        print("      - pooling2d_ctest_configs.txt / pooling2d_gtest_configs.txt")
        print("      - pooling2d_asymmetric_ctest_configs.txt / pooling2d_asymmetric_gtest_configs.txt")
        print("      - pooling2d_wide_ctest_configs.txt / pooling2d_wide_gtest_configs.txt")
        print()
        print("Examples:")
        print("  python3 compare_pooling2d_dumps.py single pooling2d_ctest_configs.txt pooling2d_gtest_configs.txt")
        print("  python3 compare_pooling2d_dumps.py all")
        print("  python3 compare_pooling2d_dumps.py all /path/to/configs/")
        sys.exit(1)
    
    mode = sys.argv[1]
    
    if mode == "single":
        if len(sys.argv) < 4:
            print("Error: 'single' mode requires two file arguments")
            print("Usage: python3 compare_pooling2d_dumps.py single <ctest_file> <gtest_file>")
            sys.exit(1)
        
        ctest_file = sys.argv[2]
        gtest_file = sys.argv[3]
        compare_files(ctest_file, gtest_file)
        
    elif mode == "all":
        # Determine directory
        if len(sys.argv) >= 3:
            directory = sys.argv[2]
        else:
            directory = "."
        
        # Define dataset files
        datasets = [
            ("Dataset 0 (Standard)", 
             os.path.join(directory, "pooling2d_ctest_configs.txt"),
             os.path.join(directory, "pooling2d_gtest_configs.txt")),
            ("Dataset 1 (Asymmetric)",
             os.path.join(directory, "pooling2d_asymmetric_ctest_configs.txt"),
             os.path.join(directory, "pooling2d_asymmetric_gtest_configs.txt")),
            ("Dataset 2 (Wide Window)",
             os.path.join(directory, "pooling2d_wide_ctest_configs.txt"),
             os.path.join(directory, "pooling2d_wide_gtest_configs.txt"))
        ]
        
        print("="*60)
        print("COMPARING ALL DATASETS")
        print("="*60)
        
        all_match = True
        for dataset_name, ctest_file, gtest_file in datasets:
            if not os.path.exists(ctest_file) and not os.path.exists(gtest_file):
                print(f"\nSkipping {dataset_name}: files not found")
                continue
            
            if not compare_files(ctest_file, gtest_file, dataset_name):
                all_match = False
        
        print("\n" + "="*60)
        if all_match:
            print("✓ ALL DATASETS MATCH!")
        else:
            print("✗ SOME DATASETS HAVE DIFFERENCES")
        print("="*60)
        
    else:
        print(f"Error: Unknown mode '{mode}'")
        print("Use 'single' or 'all'")
        sys.exit(1)

if __name__ == '__main__':
    main()

