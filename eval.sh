#!/bin/bash
export CUDA_VISIBLE_DEVICES=7

MOT_PATH=/ssd/ssd1/lyz/MOT17
MOT17POI_PATH=/ssd/ssd1/lyz/POI_feat
OUTPUT_DIR=/home/lyz/TRMatcher_output/tmp
VID_SET=evaltrain
SCORE_THRE=0.9

# run inference
OUTPUT_FILE=${OUTPUT_DIR}/`basename ${OUTPUT_DIR}`_${VID_SET}.json

if [ ! -f ${OUTPUT_FILE} ]; then
echo "Test results does not exist, start running inference"
python main.py \
  --batch_size 32 \
  --mot_path ${MOT_PATH} \
  --mot17poi_path ${MOT17POI_PATH} \
  --dataset_file mot17poi \
  --testset ${VID_SET} \
  --output_dir ${OUTPUT_DIR} \
  --resume ${OUTPUT_DIR}/checkpoint9999.pth \
  --position_embedding repeat \
  --hidden_dim 256 \
  --tube_len 2 \
  --min_vis 0.1 \
  --filtvis 0.3 \
  --enc_layers 2 \
  --dec_layers 2 \
  --no_cat_conf \
  --eval
else
echo "Test results found"
echo $OUTPUT_FILE
fi

# run matching
SCORE_THRE_100=`echo "${SCORE_THRE}*100" | bc`
INT_SCORE=${SCORE_THRE_100%.*}
MATHCHING_OUTPUT_DIR=${OUTPUT_DIR}/res_${INT_SCORE}
echo "Start matching"
 
python test_matcher.py \
    --output_file ${OUTPUT_FILE} \
    --output_dir ${MATHCHING_OUTPUT_DIR} \
    --score_thre ${SCORE_THRE} \
    --vis

# run evaluation
if [ "$VID_SET" = "evaltrain" ];then
    python -m motmetrics.apps.evaluateTracking ${MOT_PATH}/train ${MATHCHING_OUTPUT_DIR} seqmap.txt
fi
