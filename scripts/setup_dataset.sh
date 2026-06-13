#!/bin/bash
# ============================================================
# SG-VO Dataset Setup Script
# Run this AFTER the KITTI odometry zip finishes downloading.
# It will extract, organize intrinsics, and verify the layout.
# ============================================================

set -e

REPO_DIR=/home/areeba/ICRAMaxxing/SG-VO
DATA_DIR=$REPO_DIR/data
KITTI_ZIP=$DATA_DIR/data_odometry_color.zip
SEQUENCES_DIR=$DATA_DIR/kitti_odom/sequences
INTRINSICS_DIR=$SEQUENCES_DIR/kitti_odom256_intrinsics

echo "======================================"
echo " SG-VO Dataset Setup"
echo "======================================"

# --- Step 1: Check zip exists ---
if [ ! -f "$KITTI_ZIP" ]; then
    echo "[ERROR] KITTI zip not found at: $KITTI_ZIP"
    exit 1
fi
echo "[OK] Found KITTI zip: $(du -sh $KITTI_ZIP | cut -f1)"

# --- Step 2: Extract the zip ---
echo ""
echo "[1/3] Extracting KITTI odometry dataset (this will take several minutes)..."
unzip -q "$KITTI_ZIP" -d "$DATA_DIR/kitti_odom_raw" && echo "[OK] Extraction complete"

# --- Step 3: Reorganize to expected structure ---
# Expected: data/kitti_odom/sequences/00/image_2/*.png
# Zip produces: dataset/sequences/00/image_2/*.png
echo ""
echo "[2/3] Reorganizing directory structure..."

RAW_SEQ="$DATA_DIR/kitti_odom_raw/dataset/sequences"
if [ ! -d "$RAW_SEQ" ]; then
    # Try alternate zip structure
    RAW_SEQ=$(find "$DATA_DIR/kitti_odom_raw" -name "sequences" -type d | head -1)
fi

if [ -z "$RAW_SEQ" ]; then
    echo "[ERROR] Could not find sequences/ directory after extraction."
    exit 1
fi

# Move sequences to the right place
for seq in $(ls "$RAW_SEQ"); do
    if [ -d "$RAW_SEQ/$seq" ]; then
        mv "$RAW_SEQ/$seq" "$SEQUENCES_DIR/"
        echo "  Moved sequence: $seq"
    fi
done
rm -rf "$DATA_DIR/kitti_odom_raw"
echo "[OK] Sequences moved to $SEQUENCES_DIR"

# --- Step 4: Copy intrinsics cam.txt into each sequence's image_2/ folder ---
echo ""
echo "[3/3] Installing camera intrinsics (cam.txt) into each sequence..."

# Mapping: cam_XX.txt → sequences/XX/image_2/cam.txt
for i in $(seq -w 00 10); do
    CAM_SRC="$INTRINSICS_DIR/cam_${i}.txt"
    SEQ_IMG_DIR="$SEQUENCES_DIR/${i}/image_2"

    if [ -f "$CAM_SRC" ] && [ -d "$SEQ_IMG_DIR" ]; then
        cp "$CAM_SRC" "$SEQ_IMG_DIR/cam.txt"
        echo "  [OK] sequences/${i}/image_2/cam.txt"
    elif [ ! -f "$CAM_SRC" ]; then
        echo "  [WARN] No intrinsics for sequence ${i} (cam_${i}.txt missing)"
    elif [ ! -d "$SEQ_IMG_DIR" ]; then
        echo "  [WARN] image_2/ directory missing for sequence ${i}"
    fi
done

# --- Verify final structure ---
echo ""
echo "======================================"
echo " Verification"
echo "======================================"
for i in 00 09 10; do
    SEQ_DIR="$SEQUENCES_DIR/$i"
    IMG_COUNT=$(ls "$SEQ_DIR/image_2/"*.png 2>/dev/null | wc -l)
    CAM_EXISTS=$( [ -f "$SEQ_DIR/image_2/cam.txt" ] && echo "YES" || echo "MISSING" )
    echo "  Seq $i → images: $IMG_COUNT, cam.txt: $CAM_EXISTS"
done

echo ""
echo "======================================"
echo " Setup Complete!"
echo "======================================"
echo ""
echo "You can now run:"
echo "  cd $REPO_DIR"
echo "  conda activate orbsfm"
echo ""
echo "  # Evaluation (offline):"
echo "  bash scripts/test_kitti_vo.sh"
echo ""
echo "  # Evaluation (online adaptation):"
echo "  bash scripts/test_kitti_vo_online.sh"
echo ""
echo "  # Training (requires kitti_256 dataset):"
echo "  bash scripts/train.sh"
