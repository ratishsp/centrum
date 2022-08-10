
# Centrum 
This repo contains code for [Multi-Document Summarization with Centroid-Based Pretraining](https://arxiv.org/abs/2208.01006) (Ratish Puduppully and Mark Steedman.

## Citation
```
@misc{https://doi.org/10.48550/arxiv.2208.01006,
  doi = {10.48550/ARXIV.2208.01006},
  url = {https://arxiv.org/abs/2208.01006},
  author = {Puduppully, Ratish and Steedman, Mark},
  title = {Multi-Document Summarization with Centroid-Based Pretraining},
  publisher = {arXiv},
  year = {2022}
}
```
## Requirements

Create conda environment:
```bash
conda create -n centrum_env python=3.7
conda activate centrum_env
conda install pytorch cudatoolkit=11 -c pytorch
```
All dependencies can be installed via:
```bash
pip install -r requirements.txt
```

## Code Details
The scripts to create the preprocessed NewSHead dataset are located in `utils` directory

## Model
The models are uploaded to HuggingFace. The pretrained checkpoint is available at [Centrum](https://huggingface.co/ratishsp/Centrum) and the checkpoint finetuned on MultiNews is available at [Centrum-multinews](https://huggingface.co/ratishsp/Centrum-multinews).

## Attention window of 512
```python
from transformers import AutoConfig
config = AutoConfig.from_pretrained('allenai/led-base-16384')
config.attention_window = 512
config.save_pretrained("/home/user/hf/led-base-16384")
```
## Command for Pretraining 
```bash
NUM_GPU=4
MODEL_NAME=allenai/led-base-16384
LOADER=/home/user/centrum/script/newshead_dataset_loader
LOCAL_MODEL_NAME=/home/user/hf/led-base-16384
OUTPUT_DIR=/home/user/centrum_model
ACCUM_COUNT=4
LEARNING_RATE=0.00003
SAVE_TOTAL=5
PATIENCE=80
python -m torch.distributed.launch --nproc_per_node=${NUM_GPU} run_centrum.py \
    --model_name_or_path ${MODEL_NAME} \
    --do_train \
    --do_eval \
    --dataset_name ${LOADER} \
    --config_name ${LOCAL_MODEL_NAME} \
    --output_dir ${OUTPUT_DIR} \
    --per_device_train_batch_size=1 \
    --per_device_eval_batch_size=4 \
    --overwrite_output_dir \
    --logging_dir logs/centrum_logs \
    --gradient_accumulation_steps ${ACCUM_COUNT} \
    --evaluation_strategy steps \
    --overwrite_output_dir \
    --eval_steps 500 \
    --learning_rate ${LEARNING_RATE} \
    --logging_first_step \
    --logging_steps 100 \
    --report_to tensorboard \
    --save_total_limit ${SAVE_TOTAL} \
    --save_steps 500 \
    --warmup_steps 10000 \
    --max_steps 100000 \
    --label_smoothing_factor 0.1 \
    --gradient_checkpointing \
    --val_max_target_length 1024 \
    --max_target_length 1024 \
    --max_source_length 4096 \
    --fp16 \
    --load_best_model_at_end \
    --greater_is_better False \
    --metric_for_best_model loss \
    --is_pretrain \
    --resize_position_embeddings True \
    --early_stopping_patience ${PATIENCE}
```

## Command for inference on multi-news
```bash
MODEL_NAME=/home/user/centrum_model
DATASET_NAME=multi_news
OUTPUT_DIR=/home/user/centrum_multi_news_output
TARGET_LENGTH=256 # 256 in zero-shot; can be 1024 in fully-supervised setting
MODE=test
python -m torch.distributed.launch --nproc_per_node=4 run_centrum.py \
    --model_name_or_path ${MODEL_NAME} \
    --dataset_name ${DATASET_NAME} \
    --do_predict \
    --predict_with_generate \
    --logging_dir logs/multi_news_inference \
    --output_dir ${OUTPUT_DIR} \
    --overwrite_output_dir \
    --max_target_length ${TARGET_LENGTH} \
    --max_source_length 4096 \
    --num_beams 5 \
    --prediction_mode ${MODE} \
    --per_device_eval_batch_size 2
```

## Command for finetuning on multi-news
```bash
MODEL_NAME_FULL_PATH=/home/user/centrum_model
OUTPUT_DIR=/home/user/centrum_multi_news_model
LEARNING_RATE=0.00003
SAVE_TOTAL=5
PATIENCE=20
python -m torch.distributed.launch --nproc_per_node=4 run_centrum.py \
    --model_name_or_path ${MODEL_NAME_FULL_PATH} \
    --do_train \
    --do_eval \
    --predict_with_generate \
    --dataset_name multi_news \
    --output_dir ${OUTPUT_DIR} \
    --per_device_train_batch_size=1 \
    --per_device_eval_batch_size=4 \
    --overwrite_output_dir \
    --logging_dir logs/centrum_multi_news_train \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy steps \
    --overwrite_output_dir \
    --eval_steps 500 \
    --learning_rate ${LEARNING_RATE} \
    --logging_first_step \
    --logging_steps 100 \
    --report_to tensorboard \
    --save_total_limit 5 \
    --save_steps 500 \
    --warmup_steps 2500 \
    --max_steps 25000 \
    --label_smoothing_factor 0.1 \
    --gradient_checkpointing \
    --val_max_target_length 1024 \
    --max_target_length 1024 \
    --max_source_length 4096 \
    --fp16 \
    --greater_is_better True \
    --metric_for_best_model rougeL \
    --load_best_model_at_end \
    --num_beams 5 \
    --prediction_mode validation \
    --early_stopping_patience ${PATIENCE} \
    --eval_delay 5000 \
    --generation_max_length 1024
```
## Acknowledgements
Some code for preprocessing is based on the [Primera repo](https://github.com/allenai/PRIMER). The script run_centrum.py is based on the [run_summarization.py](https://github.com/huggingface/transformers/blob/master/examples/pytorch/summarization/run_summarization.py) on HuggingFace Transformers.

