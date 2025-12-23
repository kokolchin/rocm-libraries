#!/usr/bin/env python3
"""
Compare configuration dumps from ctest and gtest variants of pooling2d tests.
Handles both formats:
- CTest format: "input_dims: [1,19,1024,2048] lens: [2,2] pads: [0,0] strides: [1,1] index_type: 0 mode: 0 wsidx: 0"
- GTest format: "1 19 1024 2048 2 2 0 0 1 1 0 0 0" (13 space-separated values)
"""

import sys
import re
from collections import defaultdict

def parse_ctest_line(line):
    """Parse ctest format: input_dims: [1,19,1024,2048] lens: [2,2] ..."""
    config = {}
    
    # Parse input_dims: [1,19,1024,2048]
    match = re.search(r'input_dims:\s*\[([^\]]+)\]', line)
    if match:
        config['input_dims'] = tuple(int(x.strip()) for x in match.group(1).split(','))
    
    # Parse lens: [2,2]
    match = re.search(r'lens:\s*\[([^\]]+)\]', line)
    if match:
        config['lens'] = tuple(int(x.strip()) for x in match.group(1).split(','))
    
    # Parse pads: [0,0]
    match = re.search(r'pads:\s*\[([^\]]+)\]', line)
    if match:
        config['pads'] = tuple(int(x.strip()) for x in match.group(1).split(','))
    
    # Parse strides: [1,1]
    match = re.search(r'strides:\s*\[([^\]]+)\]', line)
    if match:
        config['strides'] = tuple(int(x.strip()) for x in match.group(1).split(','))
    
    # Parse index_type: 0
    match = re.search(r'index_type:\s*(\d+)', line)
    if match:
        config['index_type'] = int(match.group(1))
    
    # Parse mode: 0
    match = re.search(r'mode:\s*(\d+)', line)
    if match:
        config['mode'] = int(match.group(1))
    
    # Parse wsidx: 0
    match = re.search(r'wsidx:\s*(\d+)', line)
    if match:
        config['wsidx'] = int(match.group(1))
    
    return config if len(config) == 7 else None

def parse_gtest_line(line):
    """Parse gtest format: 1 19 1024 2048 2 2 0 0 1 1 0 0 0"""
    # Skip comments
    if line.strip().startswith('#'):
        return None
    
    parts = line.strip().split()
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

def load_configs(filename, is_gtest=False):
    """Load configurations from a file"""
    configs = []
    with open(filename, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            if is_gtest:
                config = parse_gtest_line(line)
            else:
                config = parse_ctest_line(line)
            
            if config:
                configs.append(normalize_config(config))
            else:
                print(f"Warning: Could not parse line {line_num} in {filename}: {line[:80]}")
    
    return set(configs)  # Use set for comparison

def main():
    if len(sys.argv) < 3:
        print("Usage: python3 compare_pooling2d_dumps.py <ctest_file> <gtest_file>")
        print("  ctest_file: Output from ctest (pooling2d_ctest_configs.txt)")
        print("  gtest_file: Output from gtest (pooling2d_gtest_configs.txt)")
        print()
        print("Example:")
        print("  python3 compare_pooling2d_dumps.py pooling2d_asymmetric_ctest_configs.txt pooling2d_gtest_configs.txt")
        sys.exit(1)
    
    ctest_file = sys.argv[1]
    gtest_file = sys.argv[2]
    
    print(f"Loading ctest configs from: {ctest_file}")
    ctest_configs = load_configs(ctest_file, is_gtest=False)
    print(f"  Loaded {len(ctest_configs)} unique configurations")
    
    print(f"\nLoading gtest configs from: {gtest_file}")
    gtest_configs = load_configs(gtest_file, is_gtest=True)
    print(f"  Loaded {len(gtest_configs)} unique configurations")
    
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)
    
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
    else:
        print("\n✗ DIFFERENCES FOUND:")
        
        if only_in_ctest:
            print(f"\n  {len(only_in_ctest)} configurations only in ctest:")
            for i, config in enumerate(sorted(only_in_ctest)[:10], 1):
                print(f"    {i}. {config}")
            if len(only_in_ctest) > 10:
                print(f"    ... and {len(only_in_ctest) - 10} more")
        
        if only_in_gtest:
            print(f"\n  {len(only_in_gtest)} configurations only in gtest:")
            for i, config in enumerate(sorted(only_in_gtest)[:10], 1):
                print(f"    {i}. {config}")
            if len(only_in_gtest) > 10:
                print(f"    ... and {len(only_in_gtest) - 10} more")
    
    # Statistics
    print("\n" + "="*60)
    print("STATISTICS")
    print("="*60)
    
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

if __name__ == '__main__':
    main()

