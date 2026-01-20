# BN 3D PerAct Test Conversion - Reviewer Response

## Summary
This document addresses the reviewer's concerns about timing and TheRock exclusion for the `bn_3d_peract_test` conversion from ctest to gtest.

## 1. Timing Comparison

### Methodology
To compare timing between the old ctest and new gtest versions:

1. **Restore old ctest version** (if needed):
   ```bash
   git checkout 6ad849d0033^  # Before removal commit
   # Build old ctest version
   ```

2. **Build new gtest version**:
   ```bash
   git checkout feature/gtest-bn-3d-peract
   cd build
   cmake .. && make test_bn_3d_peract_test
   ```

3. **Run timing comparison script**:
   ```bash
   ./compare_bn_3d_peract_timing.sh [iterations]
   ```

### Expected Results
The script will:
- Run each version multiple times (default: 5 iterations)
- Calculate average execution time
- Compare the results
- Report if new gtest is significantly slower (>10%) or faster (>10%)

### Test Details
- **Old ctest**: `test_bn_3d_peract_test`
- **New gtest**: `test_bn_3d_peract_test` with filter `Smoke/GPU_Bn3dPerAct_FP32.*`
- **Test cases**: 5 test cases (ForwardTraining, ForwardInferenceRecalc, ForwardInferenceUseEstimated, BackwardRecalc, BackwardUseSaved)

## 2. TheRock Exclusion

### Current Status
The old ctest (`test_bn_3d_peract_test`) was **not explicitly excluded** from TheRock in the original CMakeLists.txt. However, since ctests were not run on TheRock by default, the test would not have been executed there.

### Action Required
To ensure the new gtest is excluded from TheRock (matching the old behavior), we need to add it to `SKIP_TESTS` in `projects/miopen/test/gtest/CMakeLists.txt`:

```cmake
# Exclude bn_3d_peract_test from TheRock (ctests were not run on TheRock)
if(MIOPEN_BUILD_CK)  # TheRock builds with MIOPEN_BUILD_CK=OFF
    list(APPEND SKIP_TESTS bn_3d_peract_test.cpp)
endif()
```

**Note**: The exact condition for TheRock detection may need to be verified. TheRock typically has `MIOPEN_BUILD_CK=OFF` (as seen in line 240 of gtest/CMakeLists.txt).

### Alternative Approach
If TheRock uses a different mechanism (e.g., negative filter), we can use:
```cmake
if(MIOPEN_BUILD_CK)  # TheRock condition
    add_gtest_negative_filter("*GPU_Bn3dPerAct*")
endif()
```

## 3. Next Steps

1. **Run timing comparison** on a representative machine
2. **Post results** to the PR with:
   - Old ctest average time
   - New gtest average time
   - Comparison (faster/slower/similar)
   - Number of iterations run
3. **Add TheRock exclusion** if timing is acceptable
4. **Verify exclusion** works correctly

## 4. Questions for Reviewer

1. What is the exact condition to detect TheRock builds? (Is it `MIOPEN_BUILD_CK=OFF`?)
2. Should we use `SKIP_TESTS` or `add_gtest_negative_filter` for exclusion?
3. What is considered "significantly slower" - is 10% threshold acceptable?
