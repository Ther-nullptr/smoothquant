import torch
import argparse
import os

from pathlib import Path

from transformers.models.opt.modeling_opt import OPTForCausalLM
from transformers import AutoTokenizer

from datasets import load_dataset

from smoothquant.opt import Int8OPTForCausalLM
from smoothquant.smooth import smooth_lm
from smoothquant.calibration import get_static_decoder_layer_scales

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default='facebook/opt-13b')
    parser.add_argument("--num-samples", type=int, default=512)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--act-scales", type=str,
                        default='act_scales/opt-13b.pt')
    parser.add_argument("--output-path", type=str, default='int8_models')
    parser.add_argument('--dataset-path', type=str, default='dataset/val.jsonl.zst',
                        help='location of the calibration dataset, we use the validation set of the Pile dataset')
    parser.add_argument('--export-FT', default=False, action="store_true")
    parser.add_argument('--device-map', type=str, default="auto", help='device map for model parallelism')
    args = parser.parse_args()

    model = OPTForCausalLM.from_pretrained(
        args.model_name, device_map=args.device_map, torch_dtype=torch.float16)
    print(f"Model num: {sum(p.numel() for p in model.parameters()) / (1024**2):.2f} MiB")
    print(f"Model size: {sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2):.2f} MiB")
    record_gpu_memory('load fp16 model')
    act_scales = torch.load(args.act_scales) #! load per token activation scales
    smooth_lm(model, act_scales, 0.5)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    record_gpu_memory('after smooth')

    if not os.path.exists(args.dataset_path):
        print(f'Cannot find the dataset at {args.dataset_path}')
        print('Please download the Pile dataset and put the validation set at the path')
        print('You can download the validation dataset of the Pile at https://mystic.the-eye.eu/public/AI/pile/val.jsonl.zst')
        raise FileNotFoundError

    dataset = load_dataset('json', data_files=args.dataset_path, split="train")
    dataset = dataset.shuffle(seed=42)
    dataset = dataset.select(range(args.num_samples))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)
    decoder_layer_scales, raw_scales = get_static_decoder_layer_scales(model,
                                                                       tokenizer,
                                                                       dataloader,
                                                                       seq_len=args.seq_len)
    record_gpu_memory('after calibration')
    

    if args.export_FT:
        output_path = Path(args.output_path) / (Path(args.model_name).name + "-smoothquant-smoothed.pt")
        model.save_pretrained(output_path)
        print(f"Saved smoothed model at {output_path}")

        output_path = Path(args.output_path) / (Path(args.model_name).name + "-smoothquant-scales.pt")
        torch.save(raw_scales, output_path)
        print(f"Saved scaling factors at {output_path}")
    else:
        output_path = Path(args.output_path) / (Path(args.model_name).name + "-smoothquant-int.pt")
        int8_model = Int8OPTForCausalLM.from_float(model, decoder_layer_scales)
        record_gpu_memory('after quantize')
        int8_model.save_pretrained(output_path)
        print(f"Saved int8 model at {output_path}")
