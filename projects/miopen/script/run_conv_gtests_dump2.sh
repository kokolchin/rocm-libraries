BUILD=/data/build-conv-dump-gdump-clean

OUT="$BUILD/ctest_conv_dumps"

BIN="$BUILD/bin"

mkdir -p "$OUT"
 
for t in conv2d conv2d_bias conv2d_find2 find_2_conv conv3d conv3d_bias conv3d_find2 immed_conv2d immed_conv3d; do

  [ -x "$BIN/test_$t" ] || { echo "[WARN] missing $BIN/test_$t"; continue; }
 
  if [[ "$t" == "conv2d_find2" ]]; then

    echo "[WARN] Skipping slow test_$t --all"

  else

    echo "[INFO] start test_$t --all"

    MIOPEN_DUMP_CONFIGS=1 "$BIN/test_$t" --verbose --all > "$OUT/ctest_${t}--all.txt" 2>&1

  fi
 
  echo "[INFO] start test_$t --all --limit 1"

  MIOPEN_DUMP_CONFIGS=1 "$BIN/test_$t" --verbose --all --limit 1 > "$OUT/ctest_${t}--all--limit1.txt" 2>&1

done
 
