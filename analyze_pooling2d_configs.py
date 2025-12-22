#!/usr/bin/env python3
"""
Script to analyze pooling2d_gtest_configs.txt and extract statistics
about test case configurations.
"""

import sys
from collections import defaultdict

def parse_config_file(filename):
    """Parse the config file and return test cases and statistics."""
    test_cases = []
    
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip comments and empty lines
            if not line or line.startswith('#'):
                continue
            
            # Parse: input_dims[4] lens[2] pads[2] strides[2] index_type mode wsidx
            # Format: N C H W lens_H lens_W pad_H pad_W stride_H stride_W index_type mode wsidx
            # Total: 13 values
            parts = line.split()
            if len(parts) != 13:
                print(f"Warning: Skipping malformed line (expected 13 values, got {len(parts)}): {line}")
                continue
            
            try:
                input_dims = [int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])]
                lens = [int(parts[4]), int(parts[5])]
                pads = [int(parts[6]), int(parts[7])]
                strides = [int(parts[8]), int(parts[9])]
                index_type = int(parts[10])
                mode = int(parts[11])
                wsidx = int(parts[12])
                
                test_cases.append({
                    'input_dims': input_dims,
                    'lens': lens,
                    'pads': pads,
                    'strides': strides,
                    'index_type': index_type,
                    'mode': mode,
                    'wsidx': wsidx
                })
            except (ValueError, IndexError) as e:
                print(f"Warning: Error parsing line '{line}': {e}")
                continue
    
    return test_cases

def analyze_test_cases(test_cases):
    """Analyze test cases and print statistics."""
    print(f"\n{'='*60}")
    print(f"Total test cases: {len(test_cases)}")
    print(f"{'='*60}\n")
    
    # Index type mapping
    index_type_names = {
        0: 'Uint8',
        1: 'Uint16',
        2: 'Uint32',
        3: 'Uint64'
    }
    
    # Mode mapping
    mode_names = {
        0: 'Max',
        1: 'Average',
        2: 'AverageInclusive'
    }
    
    # Count by index type
    print("Breakdown by Index Type:")
    print("-" * 60)
    index_type_counts = defaultdict(int)
    for tc in test_cases:
        index_type_counts[tc['index_type']] += 1
    for idx_type in sorted(index_type_counts.keys()):
        name = index_type_names.get(idx_type, f'Unknown({idx_type})')
        count = index_type_counts[idx_type]
        print(f"  {name}: {count} cases")
    print()
    
    # Count by mode
    print("Breakdown by Mode:")
    print("-" * 60)
    mode_counts = defaultdict(int)
    for tc in test_cases:
        mode_counts[tc['mode']] += 1
    for mode in sorted(mode_counts.keys()):
        name = mode_names.get(mode, f'Unknown({mode})')
        count = mode_counts[mode]
        print(f"  {name}: {count} cases")
    print()
    
    # Count by wsidx
    print("Breakdown by wsidx:")
    print("-" * 60)
    wsidx_counts = defaultdict(int)
    for tc in test_cases:
        wsidx_counts[tc['wsidx']] += 1
    for wsidx in sorted(wsidx_counts.keys()):
        count = wsidx_counts[wsidx]
        print(f"  wsidx={wsidx}: {count} cases")
    print()
    
    # Count by index_type + mode
    print("Breakdown by Index Type + Mode:")
    print("-" * 60)
    combo_counts = defaultdict(int)
    for tc in test_cases:
        idx_name = index_type_names.get(tc['index_type'], f'Unknown({tc["index_type"]})')
        mode_name = mode_names.get(tc['mode'], f'Unknown({tc["mode"]})')
        combo_counts[(idx_name, mode_name)] += 1
    for (idx_name, mode_name) in sorted(combo_counts.keys()):
        count = combo_counts[(idx_name, mode_name)]
        print(f"  {idx_name} + {mode_name}: {count} cases")
    print()
    
    # Count by index_type + wsidx
    print("Breakdown by Index Type + wsidx:")
    print("-" * 60)
    combo_counts = defaultdict(int)
    for tc in test_cases:
        idx_name = index_type_names.get(tc['index_type'], f'Unknown({tc["index_type"]})')
        combo_counts[(idx_name, tc['wsidx'])] += 1
    for (idx_name, wsidx) in sorted(combo_counts.keys()):
        count = combo_counts[(idx_name, wsidx)]
        print(f"  {idx_name} + wsidx={wsidx}: {count} cases")
    print()
    
    # Count by mode + wsidx
    print("Breakdown by Mode + wsidx:")
    print("-" * 60)
    combo_counts = defaultdict(int)
    for tc in test_cases:
        mode_name = mode_names.get(tc['mode'], f'Unknown({tc["mode"]})')
        combo_counts[(mode_name, tc['wsidx'])] += 1
    for (mode_name, wsidx) in sorted(combo_counts.keys()):
        count = combo_counts[(mode_name, wsidx)]
        print(f"  {mode_name} + wsidx={wsidx}: {count} cases")
    print()
    
    # Count unique input shapes
    print("Unique input shapes:")
    print("-" * 60)
    input_shapes = set()
    for tc in test_cases:
        input_shapes.add(tuple(tc['input_dims']))
    print(f"  {len(input_shapes)} unique input shapes")
    for shape in sorted(input_shapes):
        print(f"    {shape}")
    print()
    
    # Count by input shape
    print("Breakdown by Input Shape:")
    print("-" * 60)
    shape_counts = defaultdict(int)
    for tc in test_cases:
        shape_counts[tuple(tc['input_dims'])] += 1
    for shape in sorted(shape_counts.keys()):
        count = shape_counts[shape]
        print(f"  {shape}: {count} cases")
    print()

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 analyze_pooling2d_configs.py <config_file>")
        print("Example: python3 analyze_pooling2d_configs.py pooling2d_gtest_configs.txt")
        sys.exit(1)
    
    filename = sys.argv[1]
    
    try:
        test_cases = parse_config_file(filename)
        if not test_cases:
            print(f"Error: No test cases found in {filename}")
            sys.exit(1)
        
        analyze_test_cases(test_cases)
        
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()

