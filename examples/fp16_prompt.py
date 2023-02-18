import torch
from transformers.models.opt.modeling_opt import OPTAttention, OPTDecoderLayer, OPTForCausalLM
from transformers import GPT2Tokenizer
import time

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
    model_path = 'facebook/opt-6.7b'
    prompt = 'Hey, are you consciours? Can you talk to me?'
    print('load model')
    model = OPTForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map='auto')
    print('load tokenizer')
    tokenizer = GPT2Tokenizer.from_pretrained('facebook/opt-2.7b')
    # model = quantize_model(model)
    inputs = tokenizer(prompt, return_tensors="pt")
    start = time.time()
    generate_ids = model.generate(inputs.input_ids.cuda(), max_length=50)
    end = time.time()
    print('time: ', end - start)
    record_gpu_memory('after fp16 inference')
    sentence = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    print(sentence)
