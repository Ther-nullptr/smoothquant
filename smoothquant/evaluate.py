import torch
import tqdm
from datasets import load_dataset
from transformers.models.opt.modeling_opt import OPTAttention, OPTDecoderLayer, OPTForCausalLM
from transformers import GPT2Tokenizer
from smoothquant.smooth import smooth_lm
from smoothquant.fake_quant import W8A8Linear
from lm_evaluation.lm_eval import tasks, evaluator

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


class Evaluator:
    def __init__(self, dataset, tokenizer, device):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.device = device

        # tokenize the dataset
        def tokenize_function(examples):
            example = self.tokenizer(examples['text'])
            return example

        self.dataset = self.dataset.map(tokenize_function, batched=True)
        self.dataset.set_format(type='torch', columns=['input_ids'])

    @torch.no_grad()
    def evaluate(self, model):
        model.eval()
        # The task is to predict the last word of the input.
        total, hit = 0, 0
        for batch in self.dataset:
            input_ids = batch['input_ids'].to(self.device).unsqueeze(0)
            label = input_ids[:, -1]
            outputs = model(input_ids)
            last_token_logits = outputs.logits[:, -2, :]
            pred = last_token_logits.argmax(dim=-1)
            total += label.size(0)
            hit += (pred == label).sum().item()
        acc = hit / total
        return acc

if __name__ == '__main__':
    model_name = 'facebook/opt-2.7b'
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    dataset = load_dataset('lambada', split='validation[:1000]')
    evaluator = Evaluator(dataset, tokenizer, 'cuda')

    # fp16
    model_fp16 = OPTForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map='auto')
    acc_fp16 = evaluator.evaluate(model_fp16)
    print(f'Original model (fp16) accuracy: {acc_fp16}')

    # int8
    model_w8a8 = quantize_model(model_fp16)
    acc_w8a8 = evaluator.evaluate(model_w8a8)
    print(f'Naive W8A8 quantized model accuracy: {acc_w8a8}')
    del model_w8a8

    # int8 with smooth
    model = OPTForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map='auto')
    act_scales = torch.load(f'../act_scales/{model_name}.pt')
    smooth_lm(model, act_scales, 0.5)
    model_smoothquant_w8a8 = quantize_model(model)
    acc_smoothquant_w8a8 = evaluator.evaluate(model_smoothquant_w8a8)
    print(f'SmoothQuant W8A8 quantized model accuracy: {acc_smoothquant_w8a8}')
    del model_smoothquant_w8a8