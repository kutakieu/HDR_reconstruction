#!/bin/bash

# This script is for generating hdr image from ldr image and rendering the scene with the generated hdr image.
# usage: ./pred_and_vis.sh ./test_data/test_img.jpg ./results
#   outputs:
#       ./results/test_img.hdr
#       ./results/rendered_test_img.png
#       ./results/final_test_img.png

INPUT_LDR_IMG_FILE=$1
SAVE_DIR=$2

FILENAME_EXT=$(basename $INPUT_LDR_IMG_FILE)
FILENAME_ONLY="${FILENAME_EXT%.*}"
HDR_FILENAME="$FILENAME_ONLY.hdr"
HDR_SAVE_PATH="$SAVE_DIR/$HDR_FILENAME"
RENDERED_RESULT_PATH="$SAVE_DIR/rendered_$FILENAME_ONLY.png"
FINAL_RESULT_PATH="$SAVE_DIR/final_$FILENAME_ONLY.png"

# generate hdr image
python predict.py --i $INPUT_LDR_IMG_FILE --save_dir $SAVE_DIR

# render scene with the generated hdr image
blender --background --python render.py $HDR_SAVE_PATH

# concatenate ldr and rendered image
python -c "
import cv2
import numpy as np
ldr = cv2.resize(cv2.imread('$INPUT_LDR_IMG_FILE'), (1024, 512))
rendered = cv2.imread('$RENDERED_RESULT_PATH')
concat = np.concatenate((ldr, rendered), axis=0)
cv2.imwrite('$FINAL_RESULT_PATH', concat)
"
