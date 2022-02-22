#!/usr/bin/env bash


# overall parameters
IMAGES_DIR=../
NUM_CLASSES=51
CKPT_FILE=./checkpoints/resnet18_places365.pth.tar
GPU=0
BATCH_SIZE=256
DECAY=1e-4
LR=0.01
EPOCHS=10
MODEL=offline

for ORDER in class_instance instance
do

	if [ "${ORDER}" == "instance" ]; then
	    STEP=30000
	else
	    STEP=10
	fi

	for SEED in 10 20 30
	do
		echo "Stream-51; model ${MODEL}; order ${ORDER}; seed ${SEED};"
		EXPT_NAME=stream51_${MODEL}_experiment_${ORDER}_seed${SEED}
		SAVE_DIR=./results/${EXPT_NAME}

		CUDA_VISIBLE_DEVICES=${GPU} python -u offline_experiments.py \
		--dataset stream51 \
		--step ${STEP} \
		--images_dir ${IMAGES_DIR} \
		--expt_name ${EXPT_NAME} \
		--save_dir ${SAVE_DIR} \
		--order ${ORDER} \
		--ckpt_file ${CKPT_FILE} \
		--batch_size ${BATCH_SIZE} \
		--init_lr ${LR} \
		--weight_decay ${DECAY} \
		--num_epochs ${EPOCHS} \
		--num_classes ${NUM_CLASSES} \
		--seed ${SEED} | tee logs/${EXPT_NAME}.log
	done
done
