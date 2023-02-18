import torch
from transformers.models.opt.modeling_opt import OPTAttention, OPTDecoderLayer, OPTForCausalLM
from transformers import GPT2Tokenizer
from smoothquant.smooth import smooth_lm
from smoothquant.fake_quant import W8A8Linear
import time

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

def record_gpu_memory(prefix):
    print('-' * 80)
    for i in range(torch.cuda.device_count()):
        memory_allocated = torch.cuda.memory_allocated(i) / 1024 ** 2
        max_memory_allocated = torch.cuda.max_memory_allocated(i) / 1024 ** 2
        memory_reserved = torch.cuda.memory_reserved(i) / 1024 ** 2
        max_memory_reserved = torch.cuda.max_memory_reserved(i) / 1024 ** 2
        print(f'({prefix}) memory use in device {i} (MiB): {memory_allocated}, max (MiB): {max_memory_allocated}, reserved (MiB): {memory_reserved}, max reserved (MiB): {max_memory_reserved}')
    print('-' * 80)

if __name__ == '__main__':
    model_path = '/home/yujin/projects/smoothquant/int8_models/naivequant_6.7b_fake'
    prompt = 'I love you forever, and '
    print('load model')
    model = OPTForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map='auto')
    print('load tokenizer')
    tokenizer = GPT2Tokenizer.from_pretrained('facebook/opt-2.7b')
    model = quantize_model(model)
    inputs = tokenizer(prompt, return_tensors="pt")
    # calculate the time (with seconds precision)
    start = time.time()
    generate_ids = model.generate(inputs.input_ids.cuda(), max_length=50)
    end = time.time()
    print('time: ', end - start)
    record_gpu_memory('after int8 inference')
    sentence = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    print(sentence)
