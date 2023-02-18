MODEL_NAME=facebook/opt-1.3b

python examples/export_int8_model.py \
       --model-name ${MODEL_NAME} \
       --act-scales act_scales/${MODEL_NAME}.pt \
       --device-map auto \
       --export-FT