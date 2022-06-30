GPU=$1

# This should be the path to the project directory.
PROJECT=/home/zilong/Projects/LASER-release
cd $PROJECT

for sz in 1 3 5 7
do
    for sd in 1 2 3 4 5 6
    do
        echo ====FUNSD sz: $sz, sd: $sd====

        # TRAIN

        DATASET=${PROJECT}/data/funsd
        FEW_SHOT_INFO=${PROJECT}/data_utils/funsd_few_shot_info.json

        TRAIN_OUTPUT=${PROJECT}/outputs/funsd/funsd-sz${sz}-sd${sd}/train_result
        TEST_OUTPUT=${PROJECT}/outputs/funsd/funsd-sz${sz}-sd${sd}/decode_result

        if [ ! -f "${TRAIN_OUTPUT}/ckpt-last/pytorch_model.bin" ]; then
            echo ==TRAIN==   
            CUDA_VISIBLE_DEVICES=$GPU python run_seq2seq.py \
            --train_file                    ${DATASET}/train-text-s2s.jsons \
            --few_shot_info                 $FEW_SHOT_INFO \
            --few_shot_size                 $sz \
            --few_shot_seed                 $sd \
            --meta_file                     ${DATASET}/meta.json \
            --output_dir                    $TRAIN_OUTPUT \
            \
            --num_training_epochs           200 \
            --num_warmup_rate               0. \
            \
            --model_type                    layoutlm \
            --model_name_or_path            ${PROJECT}/weights/layoutreader/ckpt/pytorch_model.bin \
            --tokenizer_name                ${PROJECT}/weights/special_bert_base_uncased_tokenizer \
            --config_name                   ${PROJECT}/weights/layoutreader/ckpt/config.json \
            --do_lower_case                 \
            --fp16                          \
            --fp16_opt_level                O2 \
            --max_source_seq_length         513 \
            --max_target_seq_length         511 \
            --per_gpu_train_batch_size      8 \
            --gradient_accumulation_steps   1 \
            --learning_rate                 5e-5 \
            --label_smoothing               0.1 \
            --num_training_steps            -1 \
            --logging_steps                 100 \
            --save_steps                    -1 \
            --num_warmup_steps              -1 \
            --save_last_step
        fi

        if [ ! -f "${TEST_OUTPUT}/results.jsons" ]; then
            echo ==DECODE==   
            CUDA_VISIBLE_DEVICES=$GPU python decode_seq2seq.py \
            --beam_size                     5 \
            --model_type                    layoutlm \
            --model_path                    ${TRAIN_OUTPUT}/ckpt-last \
            --meta_file                     ${DATASET}/meta.json \
            --tokenizer_name                ${PROJECT}/weights/special_bert_base_uncased_tokenizer \
            --max_seq_length                1024 \
            --fp16                          \
            --input_file                    ${DATASET}/test-text-s2s.jsons \
            --output_folder                 $TEST_OUTPUT \
            --do_lower_case                 \
            --max_tgt_length                511 \
            --batch_size                    10 \
            --length_penalty                0 \
            --forbid_duplicate_ngrams       \
            --mode                          s2s \
            --forbid_ignore_word            .
        fi

        if [ ! -f "${TEST_OUTPUT}/evaluation.json" ]; then
            echo ==EVAL==   
            python eval_seq2seq.py          \
            -result_folder                 $TEST_OUTPUT \
            -dataset_folder                $DATASET
        fi
    
    done
done
