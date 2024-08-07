import json
from argparse import ArgumentParser
import torch
import os
import json
from tqdm import tqdm
from PIL import Image
import math
import multiprocessing
from multiprocessing import Pool, Queue, Manager
from transformers import AutoModelForCausalLM, AutoTokenizer


# https://github.com/QwenLM/Qwen-VL/blob/master/eval_mm/evaluate_vqa.py
def split_list(lst, n):
    length = len(lst)
    avg = length // n  # 每份的大小
    result = []  # 存储分割后的子列表
    for i in range(n - 1):
        result.append(lst[i * avg:(i + 1) * avg])
    result.append(lst[(n - 1) * avg:])
    return result


def save_json(json_list, save_path):
    with open(save_path, 'w') as file:
        json.dump(json_list, file, indent=4)


def _get_args():
    parser = ArgumentParser()
    parser.add_argument("--image_folder", type=str, default="./EcommerceOCRBench_Images")
    parser.add_argument("--output_folder", type=str, default="./results")
    parser.add_argument("--OCRBench_file", type=str, default="./bench/Specific_EcommerceOCRBench.json")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen-VL")
    parser.add_argument("--save_name", type=str, default="qwenvl")
    parser.add_argument("--num_workers", type=int, default=1)
    args = parser.parse_args()
    return args


AllDataset_score = {"bankCard": 0, "business_license": 0, "ReCTS": 0, "ESTVQA_cn": 0, "general_machine_invoice": 0,
                    "general_pj_ch": 0, "ICDAR13": 0, "ICDAR15": 0, "idCard": 0, "letter_pj_ch": 0, "MTWI2018": 0,
                    "plane_invoice": 0, "plate": 0, "quota_invoice": 0, "RCTW17": 0, "reimbursement_pj_ch": 0,
                    "RiskCtrl_detail": 0, "sales_pj_ch": 0, "SN_main": 0, "taxi_invoice": 0, "train_invoice": 0,
                    "value_added_tax_invoice": 0, "VAT_pj_ch": 0}

num_all = {"bankCard": 0, "business_license": 0, "ReCTS": 0, "ESTVQA_cn": 0, "general_machine_invoice": 0,
           "general_pj_ch": 0, "ICDAR13": 0, "ICDAR15": 0, "idCard": 0, "letter_pj_ch": 0, "MTWI2018": 0,
           "plane_invoice": 0, "plate": 0, "quota_invoice": 0, "RCTW17": 0, "reimbursement_pj_ch": 0,
           "RiskCtrl_detail": 0, "sales_pj_ch": 0, "SN_main": 0, "taxi_invoice": 0, "train_invoice": 0,
           "value_added_tax_invoice": 0, "VAT_pj_ch": 0}


def eval_worker(args, data, eval_id, output_queue):
    print(f"Process {eval_id} start.")
    checkpoint = args.model_path
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint, device_map=f'cuda:{eval_id}', trust_remote_code=True).eval()

    tokenizer = AutoTokenizer.from_pretrained(checkpoint,
                                              trust_remote_code=True)
    tokenizer.padding_side = 'left'
    tokenizer.pad_token_id = tokenizer.eod_id

    for i in tqdm(range(len(data))):
        img_path = os.path.join(args.image_folder, data[i]['image_path'])
        qs = data[i]['question']
        # query = f'<img>{img_path}</img> {qs} Answer: '
        query = f'<img>{img_path}</img>{qs} Answer:'
        input_ids = tokenizer(query, return_tensors='pt', padding='longest')
        attention_mask = input_ids.attention_mask
        input_ids = input_ids.input_ids

        pred = model.generate(
            input_ids=input_ids.to(f'cuda:{eval_id}'),
            attention_mask=attention_mask.to(f'cuda:{eval_id}'),
            do_sample=False,
            num_beams=1,
            max_new_tokens=100,
            min_new_tokens=1,
            length_penalty=1,
            num_return_sequences=1,
            output_hidden_states=True,
            use_cache=True,
            pad_token_id=tokenizer.eod_id,
            eos_token_id=tokenizer.eod_id,
        )
        response = tokenizer.decode(pred[0][input_ids.size(1):].cpu(), skip_special_tokens=True).strip()
        data[i]['predict'] = response
    output_queue.put({eval_id: data})
    print(f"Process {eval_id} has completed.")


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    args = _get_args()
    if os.path.exists(os.path.join(args.output_folder, f"{args.save_name}.json")):
        data_path = os.path.join(args.output_folder, f"{args.save_name}.json")
        print(f"output_path:{data_path} exist! Only generate the results that were not generated in {data_path}.")
    else:
        data_path = args.OCRBench_file

    with open(data_path, "r") as f:
        data = json.load(f)

    data_list = split_list(data, args.num_workers)

    output_queue = Manager().Queue()

    pool = Pool(processes=args.num_workers)
    for i in range(len(data_list)):
        pool.apply_async(eval_worker, args=(args, data_list[i], i, output_queue))
    pool.close()
    pool.join()

    results = {}
    while not output_queue.empty():
        result = output_queue.get()
        results.update(result)
    data = []
    for i in range(len(data_list)):
        data.extend(results[i])

    for i in range(len(data)):
        # data_type = data[i]["type"]
        dataset_name = data[i]["dataset_name"]
        answers = data[i]["answers"]
        if data[i].get('predict', 0) == 0:
            continue
        predict = data[i]['predict']
        data[i]['result'] = 0

        if type(answers) == list:
            for j in range(len(answers)):
                answer = answers[j].lower().strip().replace("\n", " ")
                predict = predict.lower().strip().replace("\n", " ")
                if answer in predict:
                    data[i]['result'] = 1
        else:
            answers = answers.lower().strip().replace("\n", " ")
            predict = predict.lower().strip().replace("\n", " ")
            if answers in predict:
                data[i]['result'] = 1
    save_json(data, os.path.join(args.output_folder, f"{args.save_name}.json"))

    for i in range(len(data)):
        num_all[data[i]['dataset_name']] += 1
        if data[i].get("result", 100) == 100:
            continue
        AllDataset_score[data[i]['dataset_name']] += data[i]['result']
    for key in AllDataset_score.keys():
        print(f"{key}: {AllDataset_score[key] / float(num_all[key])}")
