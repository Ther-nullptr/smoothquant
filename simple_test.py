import torch
from torchinfo import summary
from transformers.models.opt.modeling_opt import OPTForCausalLM

if __name__ == '__main__':
    model = OPTForCausalLM.from_pretrained(
        'facebook/opt-2.7b', device_map='sequential', torch_dtype=torch.float16)
    # model.eval()
    # define a (1, 512) input tensor with type torch.long
    input_ids = torch.ones(1, 512, dtype=torch.long)
    summary(model, input_data = input_ids, depth=2, col_names=("input_size", "output_size", "num_params", "kernel_size", "mult_adds"), verbose=2, mode='eval')
    