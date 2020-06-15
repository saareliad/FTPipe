export SQUAD_DIR=/home_local/saareliad/data/squad2/
export OMP_NUM_THREADS=10
# MODEL="deepset/bert-large-uncased-whole-word-masking-squad2" # its not finetuned...
MODEL="bert-large-uncased-whole-word-masking"
function eval(){
python -m torch.distributed.launch --nproc_per_node=8 run_squad.py \
    --model_type bert \
    --model_name_or_path ${MODEL} \
    --do_eval \
    --do_lower_case \
    --train_file $SQUAD_DIR/train-v2.0.json \
    --predict_file $SQUAD_DIR/dev-v2.0.json \
    --learning_rate 3e-5 \
    --num_train_epochs 2 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --output_dir ./wwm_uncased_finetuned_squad2/ \
    --per_gpu_eval_batch_size=16 
}

function train(){
python -m torch.distributed.launch --nproc_per_node=8 run_squad.py \
    --model_type bert \
    --model_name_or_path ${MODEL} \
    --do_eval \
    --do_train \
    --do_lower_case \
    --train_file $SQUAD_DIR/train-v2.0.json \
    --predict_file $SQUAD_DIR/dev-v2.0.json \
    --learning_rate 3e-5 \
    --num_train_epochs 2 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --output_dir ./wwm_uncased_finetuned_squad2/ \
    --per_gpu_train_batch_size=3  \
    --per_gpu_eval_batch_size=3 
}

train
