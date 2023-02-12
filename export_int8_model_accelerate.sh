# 注意这个脚本是失败品！由于量化时的batch_size是1，所以在多机运行时只需要将模型分发到各个机器上即可，不需要将数据分发到各个机器上，也就不需要用到accelerater了。
MODEL_NAME=facebook/opt-1.3b

accelerate-launch examples/export_int8_model_distributed.py \
       --model-name ${MODEL_NAME} \
       --act-scales act_scales/${MODEL_NAME}.pt \
       --device-map auto 