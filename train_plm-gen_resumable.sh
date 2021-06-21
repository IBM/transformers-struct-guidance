#!/bin/bash
#
#SBATCH --ntasks=1
#SBATCH --time=168:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --output=logs/log-plm-gen_%j.txt

#source activate /om/user/pqian/envs/py37
source activate plm

STRUCTURED_MASK=false
FREQUENT_VALID=false

POSITIONAL=()
while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    -d|--data)
    CORPUS="$2"
    shift # past argument
    shift # past value
    ;;
    -s|--seed)
    SEED="$2"
    shift # past argument
    shift # past value
    ;;
    -v|--valid_every)
    FREQUENT_VALID=true
    shift # past argument
    ;;
    --structured)
    STRUCTURED_MASK=true
    shift # past argument
    ;;
    *)    # unknown option
    POSITIONAL+=("$1") # save it in an array for later
    shift # past argument
    ;;
esac
done
set -- "${POSITIONAL[@]}" # restore positional parameters


BATCH_SIZE=5
EPOCHS=50
REPORT=5000
SAMPLE_EVERY=50000

if [[ "${FREQUENT_VALID}" == true ]]; then
    VALID_EVERY=100000
else
    VALID_EVERY=-1
fi

if [[ "${STRUCTURED_MASK}" == true ]]; then
    # using attention mask
    MODEL_PATH="model_params/xplm-mask_${CORPUS}_rand-init_${SEED}_${BATCH_SIZE}.params"
    LOG_FILE="train_logs/plm-mask_${CORPUS}_${SEED}_${BATCH_SIZE}.txt"
    if [ -f ${MODEL_PATH} ]; then
        python src/plm-gen.py --train_data data/${CORPUS}/oracle_gen/train_gen.oracle --dev_data data/${CORPUS}/oracle_gen/dev_gen.oracle --lr 1e-5 --epochs ${EPOCHS} --do_train --batch_size ${BATCH_SIZE} --random_init --add_structured_mask --buffer_head 0 --stack_head 1 --report ${REPORT} --sample_every ${SAMPLE_EVERY} --valid_every ${VALID_EVERY} --model_path ${MODEL_PATH} --restore_from ${MODEL_PATH} >> ${LOG_FILE} 2>&1
    else
        python src/plm-gen.py --train_data data/${CORPUS}/oracle_gen/train_gen.oracle --dev_data data/${CORPUS}/oracle_gen/dev_gen.oracle --lr 1e-5 --epochs ${EPOCHS} --seed ${SEED} --do_train --batch_size ${BATCH_SIZE} --random_init --add_structured_mask --buffer_head 0 --stack_head 1 --report ${REPORT} --sample_every ${SAMPLE_EVERY} --valid_every ${VALID_EVERY} --model_path ${MODEL_PATH} >> ${LOG_FILE} 2>&1
    fi
else
    # not using the attention mask
    MODEL_PATH="model_params/xplm_${CORPUS}_rand-init_${SEED}_${BATCH_SIZE}.params"
    LOG_FILE="train_logs/plm_${CORPUS}_${SEED}_${BATCH_SIZE}.txt"
    if [ -f ${MODEL_PATH} ]; then
        python src/plm-gen.py --train_data data/${CORPUS}/oracle_gen/train_gen.oracle --dev_data data/${CORPUS}/oracle_gen/dev_gen.oracle --lr 1e-5 --epochs ${EPOCHS} --do_train --batch_size ${BATCH_SIZE} --random_init --report ${REPORT} --sample_every ${SAMPLE_EVERY} --valid_every ${VALID_EVERY} --model_path ${MODEL_PATH} --restore_from ${MODEL_PATH} >> ${LOG_FILE} 2>&1
    else
        python src/plm-gen.py --train_data data/${CORPUS}/oracle_gen/train_gen.oracle --dev_data data/${CORPUS}/oracle_gen/dev_gen.oracle --lr 1e-5 --epochs ${EPOCHS} --seed ${SEED} --do_train --batch_size ${BATCH_SIZE} --random_init --report ${REPORT} --sample_every ${SAMPLE_EVERY} --valid_every ${VALID_EVERY} --model_path ${MODEL_PATH} >> ${LOG_FILE} 2>&1
    fi
fi
