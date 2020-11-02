export TRAIN_FILE=/home_local/saareliad/data/wikitext-2-raw/wiki.train.raw
export TEST_FILE=/home_local/saareliad/data/wikitext-2-raw/wiki.test.raw

export CUDA_VISABLE_DEVICES="0,1,2,3"
export OMP_NUM_THREADS=5

output_lowercase_noclip_untied(){
python -m torch.distributed.launch --nproc_per_node 4 run_language_modeling.py \
    --output_dir=output_lowercase_noclip_untied \
    --model_type=gpt2 \
    --model_name_or_path=gpt2 \
    --do_train \
    --train_data_file=$TRAIN_FILE \
    --do_eval \
    --num_train_epochs 3 \
    --per_gpu_train_batch_size 1 \
    --per_gpu_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --eval_data_file=$TEST_FILE \
    --warmup_steps 0 \
    --logging_step 592 \
    --save_steps 0 \
    --evaluate_during_training \
    --do_lower_case \
    --max_grad_norm 0 \
    --untied \
    --overwrite_cache \
    --overwrite_output_dir \
    --evaluate_every_epoch
}

output_lowercase_noclip_tied(){
python -m torch.distributed.launch --nproc_per_node 4 run_language_modeling.py \
    --output_dir=output_lowercase_noclip_tied \
    --model_type=gpt2 \
    --model_name_or_path=gpt2 \
    --do_train \
    --train_data_file=$TRAIN_FILE \
    --do_eval \
    --num_train_epochs 3 \
    --per_gpu_train_batch_size 1 \
    --per_gpu_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --eval_data_file=$TEST_FILE \
    --warmup_steps 0 \
    --logging_step 592 \
    --save_steps 0 \
    --evaluate_during_training \
    --do_lower_case \
    --max_grad_norm 0 \
    --overwrite_cache \
    --overwrite_output_dir \
    --evaluate_every_epoch
}

output_lowercase_clip_tied(){
python -m torch.distributed.launch --nproc_per_node 4 run_language_modeling.py \
    --output_dir=output_lowercase_clip_tied \
    --model_type=gpt2 \
    --model_name_or_path=gpt2 \
    --do_train \
    --train_data_file=$TRAIN_FILE \
    --do_eval \
    --num_train_epochs 3 \
    --per_gpu_train_batch_size 1 \
    --per_gpu_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --eval_data_file=$TEST_FILE \
    --warmup_steps 0 \
    --logging_step 592 \
    --save_steps 0 \
    --evaluate_during_training \
    --do_lower_case \
    --max_grad_norm 1 \
    --overwrite_cache \
    --overwrite_output_dir \
    --evaluate_every_epoch
}

output_lowercase_clip_untied(){
python -m torch.distributed.launch --nproc_per_node 4 run_language_modeling.py \
    --output_dir=output_lowercase_clip_untied \
    --model_type=gpt2 \
    --model_name_or_path=gpt2 \
    --do_train \
    --train_data_file=$TRAIN_FILE \
    --do_eval \
    --num_train_epochs 3 \
    --per_gpu_train_batch_size 1 \
    --per_gpu_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --eval_data_file=$TEST_FILE \
    --warmup_steps 0 \
    --logging_step 592 \
    --save_steps 0 \
    --evaluate_during_training \
    --do_lower_case \
    --max_grad_norm 1 \
    --untied \
    --overwrite_cache \
    --overwrite_output_dir \
    --evaluate_every_epoch
}



output_noclip_untied(){
python -m torch.distributed.launch --nproc_per_node 4 run_language_modeling.py \
    --output_dir=output_noclip_untied \
    --model_type=gpt2 \
    --model_name_or_path=gpt2 \
    --do_train \
    --train_data_file=$TRAIN_FILE \
    --do_eval \
    --num_train_epochs 3 \
    --per_gpu_train_batch_size 1 \
    --per_gpu_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --eval_data_file=$TEST_FILE \
    --warmup_steps 0 \
    --logging_step 592 \
    --save_steps 0 \
    --evaluate_during_training \
    --max_grad_norm 0 \
    --untied \
    --overwrite_cache \
    --overwrite_output_dir \
    --evaluate_every_epoch
}

output_noclip_tied(){
python -m torch.distributed.launch --nproc_per_node 4 run_language_modeling.py \
    --output_dir=output_noclip_tied \
    --model_type=gpt2 \
    --model_name_or_path=gpt2 \
    --do_train \
    --train_data_file=$TRAIN_FILE \
    --do_eval \
    --num_train_epochs 3 \
    --per_gpu_train_batch_size 1 \
    --per_gpu_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --eval_data_file=$TEST_FILE \
    --warmup_steps 0 \
    --logging_step 592 \
    --save_steps 0 \
    --evaluate_during_training \
    --max_grad_norm 0 \
    --overwrite_cache \
    --overwrite_output_dir \
    --evaluate_every_epoch
}

output_clip_untied(){
python -m torch.distributed.launch --nproc_per_node 4 run_language_modeling.py \
    --output_dir=output_clip_untied \
    --model_type=gpt2 \
    --model_name_or_path=gpt2 \
    --do_train \
    --train_data_file=$TRAIN_FILE \
    --do_eval \
    --num_train_epochs 3 \
    --per_gpu_train_batch_size 1 \
    --per_gpu_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --eval_data_file=$TEST_FILE \
    --warmup_steps 0 \
    --logging_step 592 \
    --save_steps 0 \
    --evaluate_during_training \
    --max_grad_norm 1 \
    --untied \
    --overwrite_cache \
    --overwrite_output_dir \
    --evaluate_every_epoch
}

output_clip_tied(){
python -m torch.distributed.launch --nproc_per_node 4 run_language_modeling.py \
    --output_dir=output_clip_untied \
    --model_type=gpt2 \
    --model_name_or_path=gpt2 \
    --do_train \
    --train_data_file=$TRAIN_FILE \
    --do_eval \
    --num_train_epochs 3 \
    --per_gpu_train_batch_size 1 \
    --per_gpu_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --eval_data_file=$TEST_FILE \
    --warmup_steps 0 \
    --logging_step 592 \
    --save_steps 0 \
    --evaluate_during_training \
    --max_grad_norm 1 \
    --overwrite_cache \
    --overwrite_output_dir \
    --evaluate_every_epoch
}


lowercase(){
output_lowercase_noclip_untied
output_lowercase_noclip_tied
output_lowercase_clip_untied
output_lowercase_clip_tied

}

normalcase(){
output_noclip_untied
output_noclip_tied
output_clip_untied
output_clip_tied
}


# normalcase
lowercase
