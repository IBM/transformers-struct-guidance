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
