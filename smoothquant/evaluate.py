import os
from pprint import pprint
from argparse import ArgumentParser
from transformers import AutoTokenizer, AutoModelForCausalLM
from lm_for_evaluate import LMClass
from lm_evaluation.lm_eval import evaluator

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model-name', type=str, default='decapoda-research/llama-7b-hf')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--task', type=str, default='lambada_openai')
    args = parser.parse_args()
    print(args)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    if not(os.path.exists('results')):
        os.mkdir(f'results')

    # fp16
    model_fp16 = AutoModelForCausalLM.from_pretrained(args.model_name, device_map='auto')
    model_fp16.eval()
    lm = LMClass(model_fp16, tokenizer, args.batch_size)
    acc_fp16 = evaluator.simple_evaluate(lm, tasks = args.task)
    pprint(f'Original model (fp16) accuracy: {acc_fp16}')

    args.model_name = args.model_name.split('/')[-1]
    # write the result to a file
    with open(f'results/{args.model_name}/{args.task}.txt', 'w') as f:
        f.write(f'Original model (fp16) accuracy: {acc_fp16}\n')
    del lm