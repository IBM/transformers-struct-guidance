# Structural Guidance for Transformer Language Models

ACL2021 paper code

This repository includes inplementation for Parsing-as-Language-Modelling and structural scaffolding for Transformer language models.

## Environment

The code is based on Python3. You can install the different modules with
```
bash scripts/download_and_patch_transformers.sh
pip install -r requirements.txt
python -c "import nltk;nltk.download('punkt')"
```

The Huggingface transformers is updated indirectly through a patch. If you
modifiy the code, to commit changes run

```
bash scripts/generate_patch.sh
```

and then just commit this patch


## Prepare parsing oracle files 

```
python src/get_oracle.py --gen --fpath train.txt > train_gen.oracle
python src/get_oracle.py --gen --fpath dev.txt > dev_gen.oracle
python src/get_oracle.py --gen --fpath test.txt > test_gen.oracle
```

## Vanilla Language Models (LM)

The script `src/lm.py` implements a vanilla Transformer language model. Below are the commands for model training and evaluation, as well as commands to compute word-level surprisals from a trained model.

```
# Model training
python src/lm.py --train_data train.txt --dev_data dev.txt --lr 1e-5 --epochs ${EPOCHS} --seed ${SEED} --do_train --random_init --batch_size ${BATCH_SIZE} --report ${REPORT} --sample_every ${SAMPLE_EVERY} --model_path ${MODEL_PATH}

# Compute word-level perplexity
python src/lm.py --restore_from ${MODEL_PATH} --test_data test.txt --do_test

# Estimate word surprisals
python src/lm.py --restore_from ${MODEL_PATH} --do_eval --fpath ${TEST_SUITE_PATH} --pretokenized > ${OUTPUT_PATH}
```

## Scaffoled Language Models (ScLM)

The script `src/lm-sc.py` implements Transformer language model with structural prediction as an auxilliary task, referred as ScLM in short. The commanline variable, ${SCAFFOLD_TYPE}, can be set as `past` or `next`, which corresponds to `ScLM-past` or `ScLM-next` respectively in the paper.

```
# Model training  
python src/lm-sc.py --train_data train_gen.oracle --dev_data dev_gen.oracle --lr 1e-5 --epochs ${EPOCHS} --seed ${SEED} --do_train --random_init --batch_size ${BATCH_SIZE} --report ${REPORT} --sample_every ${SAMPLE_EVERY} --alpha 0.5 --scaffold_type ${SCAFFOLD_TYPE} --model_path ${MODEL_PATH}

# Compute word-level perplexity
python src/plm-gen.py --restore_from model_params/xplm_bllip-lg_rand-init_1101_5.params --test_data data/bllip-md/oracle_gen/test_gen.oracle --do_test

# Estimate word surprisals
python src/lm-sc.py --restore_from ${MODEL_PATH} --do_eval --fpath ${TEST_SUITE_PATH} --pretokenized > ${OUTPUT_PATH}
```

## Parsing as Language Modelling (PLM/PLM-mask)

The script `src/plm-gen.py` implements the idea of generative parsing as language modelling, a probabilistic model of top-down parsing action sequence, parameterized by Transformer architecture.

```
# Estimate word surprisals with PLM
python src/plm-gen.py --restore_from ${MODEL_PATH} --do_eval --beam_size 100 --word_beam_size 10 --fast_track_size 5 --pretokenized --fpath ${TEST_SUITE_PATH} > ${OUTPUT_PATH} 2>${EVAL_LOG_PATH}
# Estimate word surprisals with PLM-mask
python src/plm-gen.py --restore_from ${MODEL_PATH} --add_structured_mask --buffer_head 0 --stack_head 1 --do_eval --beam_size 100 --word_beam_size 10 --fast_track_size 5 --pretokenized --fpath ${TEST_    SUITE_PATH} > ${OUTPUT_PATH} 2>>${EVAL_LOG_PATH}
```.

## Estimate word surprisals

LM:
```
python src/lm.py --restore_from ${MODEL_PATH} --do_eval --fpath ${TEST_SUITE_PATH} --pretokenized > ${OUTPUT_PATH}
```

ScLM:
```
python src/lm-sc.py --restore_from ${MODEL_PATH} --do_eval --fpath ${TEST_SUITE_PATH} --pretokenized > ${OUTPUT_PATH}
```

PLM:
```
python src/plm-gen.py --restore_from ${MODEL_PATH} --do_eval --beam_size 100 --word_beam_size 10 --fast_track_size 5 --pretokenized --fpath ${TEST_SUITE_PATH} > ${OUTPUT_PATH} 2>${EVAL_LOG_PATH}
```

PLM-mask:
```
python src/plm-gen.py --restore_from ${MODEL_PATH} --add_structured_mask --buffer_head 0 --stack_head 1 --do_eval --beam_size 100 --word_beam_size 10 --fast_track_size 5 --pretokenized --fpath ${TEST_SUITE_PATH} > ${OUTPUT_PATH} 2>>${EVAL_LOG_PATH}
```
