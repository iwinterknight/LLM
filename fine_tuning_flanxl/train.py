model_id = "google/flan-t5-xxl"


def train(tokenized_dataset, projectdir, config, max_target_length):
    print("Reached Training...")

    data_path = config["paths"]["data_path"]
    save_dataset_path = os.path.join(projectdir, data_path, "finetuning_data")

    # deepspeed --num_gpus=4 scripts/run_seq2seq_deepspeed.py \
    # --model_id $model_id \
    # --dataset_path $save_dataset_path \
    # --epochs 5 \
    # --per_device_train_batch_size 8 \
    # --per_device_eval_batch_size 8 \
    # --generation_max_length $max_target_length \
    # --lr 1e-4 \
    # --deepspeed configs/ds_flan_t5_z3_config_bf16.json 
    pass