import os
import torch
from pprint import pprint
from argparse import ArgumentParser
from transformers import AutoTokenizer, AutoModelForCausalLM
from smoothquant.smooth import smooth_lm
from smoothquant.fake_quant import W8A8Linear
from lm_for_evaluate import LMClass
from lm_evaluation.lm_eval import evaluator
from transformers.models.opt.modeling_opt import OPTAttention, OPTDecoderLayer, OPTForCausalLM

# if want to use modified OPT, use `from pruned_kv_opt import OPTForCausalLM` instead

def quantize_model(model, weight_quant='per_tensor', act_quant='per_tensor', quantize_bmm_input=True):
    for name, m in model.model.named_modules():
        if isinstance(m, OPTDecoderLayer):
            m.fc1 = W8A8Linear.from_float(m.fc1, weight_quant=weight_quant, act_quant=act_quant)
            m.fc2 = W8A8Linear.from_float(m.fc2, weight_quant=weight_quant, act_quant=act_quant)
        elif isinstance(m, OPTAttention):
            # Her we simulate quantizing BMM inputs by quantizing the output of q_proj, k_proj, v_proj
            m.q_proj = W8A8Linear.from_float(
                m.q_proj, weight_quant=weight_quant, act_quant=act_quant, quantize_output=quantize_bmm_input)
            m.k_proj = W8A8Linear.from_float(
                m.k_proj, weight_quant=weight_quant, act_quant=act_quant, quantize_output=quantize_bmm_input)
            m.v_proj = W8A8Linear.from_float(
                m.v_proj, weight_quant=weight_quant, act_quant=act_quant, quantize_output=quantize_bmm_input)
            m.out_proj = W8A8Linear.from_float(m.out_proj, weight_quant=weight_quant, act_quant=act_quant)
    return model


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model-name', type=str, default='facebook/opt-6.7b')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--task', type=str, default='lambada_openai')
    parser.add_argument('--smooth', type=float, default=0.5)
    args = parser.parse_args()
    print(args)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    if not(os.path.exists('results')):
        os.mkdir(f'results')

    # fp16
    model_fp16 = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float16, device_map='auto')
    model_fp16.eval()
    lm = LMClass(model_fp16, tokenizer, args.batch_size)
    acc_fp16 = evaluator.simple_evaluate(lm, tasks = args.task)
    pprint(f'Original model (fp16) accuracy: {acc_fp16}')
    # write the result to a file
    args.model_name = args.model_name.split('/')[-1]
    with open(f'results/{args.model_name}/{args.task}.txt', 'w') as f:
        f.write(f'Original model (fp16) accuracy: {acc_fp16}\n')
    del lm

    # int8
    model_w8a8 = quantize_model(model_fp16)
    model_w8a8.eval()
    lm = LMClass(model_w8a8, tokenizer, args.batch_size)
    acc_w8a8 = evaluator.simple_evaluate(lm, tasks = args.task)
    pprint(f'Naive W8A8 quantized model accuracy: {acc_w8a8}')
    # write the result to a file
    with open(f'results/{args.model_name}/{args.task}.txt', 'a') as f:
        f.write(f'Naive W8A8 quantized model accuracy: {acc_w8a8}\n')
    del lm
    del model_fp16
    del model_w8a8

    # int8 with smooth
    model = OPTForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float16, device_map='auto')
    act_scales = torch.load(f'../act_scales/{args.model_name}.pt')
    smooth_lm(model, act_scales, args.smooth)
    model_smoothquant_w8a8 = quantize_model(model)
    model_smoothquant_w8a8.eval()
    lm = LMClass(model_smoothquant_w8a8, tokenizer, args.batch_size)
    acc_smoothquant_w8a8 = evaluator.simple_evaluate(lm, tasks = args.task)
    print(f'SmoothQuant W8A8 quantized model accuracy: {acc_smoothquant_w8a8}')
    # write the result to a file
    with open(f'results/{args.model_name}/{args.task}.txt', 'a') as f:
        f.write(f'SmoothQuant W8A8 quantized model accuracy: {acc_smoothquant_w8a8}\n')
    del lm