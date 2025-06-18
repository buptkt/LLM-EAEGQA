import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from vllm import LLM, SamplingParams
import json
from tqdm import tqdm
import re

# llm = LLM(model="Llama-2-13b-chat-ms", gpu_memory_utilization=0.9)

# # Create a sampling params object.
# sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=512)
llm = None
sampling_params = None

def load_prompt(filepath):
    # formatting = '[INST] <<SYS>>\nYou are a helpful assistant.\n<</SYS>>\n\n [/INST] {}'
    nf = [] # custom_id, prompt
    with open(filepath, 'r') as f:
        for line in f.readlines():
            line = json.loads(line)
            nline = {'custom_id': line['custom_id'], 'messages': line['body']['messages']}
            nf.append(nline)
    return nf

def generate(examples):
    for example in examples[:5]:
        prompt = example['prompt']
        outputs = llm.generate(prompt, sampling_params)
        print(outputs)
        # Print the outputs.
        print("\nGenerated Outputs:\n" + "-" * 60)
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"Prompt:    {prompt!r}")
            print(f"Output:    {generated_text!r}")
            print("-" * 60)

from openai import OpenAI
# Set OpenAI's API key and API base to use vLLM's API server.
# openai_api_key = "EMPTY"
# openai_api_base = "http://localhost:31867/v1"

# client = OpenAI(
#     api_key=openai_api_key,
#     base_url=openai_api_base,
# )
client = None
def chat(examples, split, modelname='qwen2.5'):
    nf = open(f'{split}-{modelname}-ACE05.jsonl', 'w')
    for example in tqdm(examples):
        messages = example['messages']
        chat_response = client.chat.completions.create(
            model=Qwen2.5-72B-Instruct",
            messages=messages,
            temperature=0.7,
            top_p=0.8,
            max_tokens=256,
            extra_body={
                "repetition_penalty": 1.05,
            },
        )
        print("Chat response:", chat_response)
        response = chat_response.choices[0].message.content
        print(response)
        print(example['answer'])
        example['response'] = response
        nf.write(json.dumps(example)+'\n')


def load_inout_prompt(filepath='/data/sdb2/lkt/eeqga/ace/test_ace_in_out_file.jsonl'):
    # formatting = '[INST] <<SYS>>\nYou are a helpful assistant.\n<</SYS>>\n\n [/INST] {}'
    nf = [] # custom_id, prompt
    with open(filepath, 'r') as f:
        for line in f.readlines():
            line = json.loads(line)
            nline = {'messages': [
    {"role": "system", "content": "You are an expert in event argument extraction. Please extract the target arguments based on the following prompts and text content. Your response should follow the format: The event type is <event_type>, the trigger is <trigger_token>, and the trigger span is <trigger_span>, so the *<target_argument_type>* and span is: <target_argument_span> target_argument_token, if no answer the response should be 'None'.\nExample2: The event type is <Movement.Transport>, the trigger is <travelling>, and the trigger span is <7, 7>, so the *<Origin>* and span is: <8, 8> residential area; <9, 9> university\n"},
    {"role": "user", "content": line['input']}
  ], 'answer': line['answer']}
            nf.append(nline)
    return nf

def run():
    # outputs = llm.generate('what is your name? you are', sampling_params)
    # print(outputs)
    # for split in ['test', 'valid', 'train']:
    #     examples = load_prompt(f'/data/sdb2/lkt/eeqga/ace/{split}_convert.json.3request.jsonl.qwen2.5-72b.jsonl')
    #     chat(examples, split)
        
    #     # split_jsonl(split)
    examples = load_inout_prompt()
    chat(examples, 'test', modelname='qwen2.5-72b-int4')


def split_jsonl(split):
    f = open(f'Llama-2-13b-chat-ms-ACE05.jsonl', 'r')
    nf = open(f'Llama-2-13b-chat-ms-ACE05-cleaned.jsonl', 'w')
    text = f.read()
    lines = ['{"custom_id":' + x for x in text.split('{"custom_id":')]
    for line in lines:
        nf.write(line+'\n')



def cal_LLM_res(filepath):
    f = open(filepath, 'r')
    all_pred = collections.OrderedDict()
    all_gold = collections.OrderedDict()
    invalid = 0
    for line in f.readlines():
        line = json.loads(line)
        pre_answer = line['response']
        content = line['messages'][1]['content']
        pattern = r"<context>: ([\s\S]*)\n<question>"
        input_sent = re.findall(pattern, content)[0]
        example_id = input_sent

        # print(input_sent)
        # exit()
        pattern = r'<s> <event type>: ([\s\S]*) \n'
        event_type = re.findall(pattern, content)[0]
        
        pattern = r'\n<argument type>: ([\s\S]*) \n'
        argument_type = re.findall(pattern, content)[0]
        
        event_type_argument_type = event_type + '_' + argument_type
        
        if len(pre_answer.split(": ")) == 2:
            pre_answer = pre_answer.split(": ")[1]
        pre_arg_list = pre_answer.split("; ")
        cur_start = 0
        if 'and span is: None' in pre_answer:
            pre_arg_list = []
        for one_pre_arg in pre_arg_list:
            one_pre_arg = one_pre_arg.strip()
            if one_pre_arg == "":
                continue
            pattern = r"<([0-9]*), ([0-9]*)> (.*)"
                # 使用正则表达式查找匹配项
            matches = re.search(pattern, one_pre_arg)
                # 如果找到匹配项，则提取组
            if matches:
                one_arg_s = matches.group(1).strip()
                one_arg_e = matches.group(2).strip()
                one_pre_arg = matches.group(3).strip()
                if not one_arg_s or not one_arg_e or not one_pre_arg:
                    invalid += 1
                    continue
                try:
                    one_arg_s = int(one_arg_s)
                    one_arg_e = int(one_arg_e)
                except:
                    one_arg_s = 0
                    one_arg_e = 0
                    invalid += 1
                
                start = one_pre_arg.split()[0]
                cnt = len(one_pre_arg.split())
                if '-' in start:
                    start = start.split('-')[0]
                    cnt += 1
            else:
                one_arg_s = input_sent.find(one_pre_arg, cur_start)
                if one_arg_s == -1:
                    continue
                one_arg_e = one_arg_s + len(one_pre_arg) - 1
                cur_start = one_arg_e + 1

                if one_arg_s >= len(input_sent.split()) or one_arg_e >= len(input_sent.split()) or ' '.join(input_sent.split()[one_arg_s: one_arg_e + 1]) != one_pre_arg:
                    continue
            
            if example_id not in all_pred:
                all_pred[example_id] = []
            all_pred[example_id].append(
                [
                    event_type_argument_type,
                    one_arg_s,
                    one_arg_e,
                    one_pre_arg,
                ]
            )
            
        answer = line['answer']
        if example_id not in all_gold:
            all_gold[example_id] = []
        for a in answer:
            all_gold[example_id].append(
                [
                    event_type_argument_type,
                    a[0],
                    a[1],
                    a[2],
                ]
            )
        
    return all_gold, all_pred

import copy
import collections
def cal_metrics(all_gold, all_pred):
    pre_answers = collections.OrderedDict()
    
    gold_arg_n, pred_arg_n, pred_in_gold_n, gold_in_pred_n = 0, 0, 0, 0
    for example_id in all_gold:
        if example_id not in all_pred:
            all_pred[example_id] = []
            pre_answers[example_id] = []
            print("gold example_id not in all_pred")
            
        pred_arg = all_pred[example_id]
        gold_arg = all_gold[example_id]
        # error_file.write(f"pre_answer:{str(pre_answers[example_id])}\npred:{str(pred_arg)}\ngold:{str(gold_arg)}\n\n")
            
        for _ in pred_arg:
            pred_arg_n += 1

        for _ in gold_arg:
            gold_arg_n += 1

        for argument in pred_arg:
            if argument in gold_arg:
                pred_in_gold_n += 1

        for argument in gold_arg:
            if argument in pred_arg:
                gold_in_pred_n += 1


    if pred_arg_n != 0:
        prec_c = 100.0 * pred_in_gold_n / pred_arg_n
    else:
        prec_c = 0
    if gold_arg_n != 0:
        recall_c = 100.0 * gold_in_pred_n / gold_arg_n
    else:
        recall_c = 0
    if prec_c or recall_c:
        f1_c = 2 * prec_c * recall_c / (prec_c + recall_c)
    else:
        f1_c = 0

    # get results identification
    for example_id in all_gold:
        for argument in all_pred[example_id]:
            argument[0] = argument[0].split("_")[0]
        for argument in all_gold[example_id]:
            argument[0] = argument[0].split("_")[0]

    gold_arg_n, pred_arg_n, pred_in_gold_n, gold_in_pred_n = 0, 0, 0, 0
    for example_id in all_gold:
        pred_arg = all_pred[example_id]
        gold_arg = all_gold[example_id]

        for _ in pred_arg:
            pred_arg_n += 1

        for _ in gold_arg:
            gold_arg_n += 1

        for argument in pred_arg:
            if argument in gold_arg:
                pred_in_gold_n += 1

        for argument in gold_arg:
            if argument in pred_arg:
                gold_in_pred_n += 1

    if pred_arg_n != 0:
        prec_i = 100.0 * pred_in_gold_n / pred_arg_n
    else:
        prec_i = 0
    if gold_arg_n != 0:
        recall_i = 100.0 * gold_in_pred_n / gold_arg_n
    else:
        recall_i = 0
    if prec_i or recall_i:
        f1_i = 2 * prec_i * recall_i / (prec_i + recall_i)
    else:
        f1_i = 0

    result = collections.OrderedDict(
        [
            ("prec_c", prec_c),
            ("recall_c", recall_c),
            ("f1_c", f1_c),
            ("prec_i", prec_i),
            ("recall_i", recall_i),
            ("f1_i", f1_i),
        ]
    )     
    return result
if __name__ == '__main__':
    pass
