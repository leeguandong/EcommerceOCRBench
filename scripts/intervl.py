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
from PIL import Image
from transformers import AutoModel, CLIPImageProcessor
from transformers import AutoTokenizer


# https://github.com/OpenGVLab/InternVL

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
    parser.add_argument("--image_folder", type=str, default="./EcommereceOCRBench_Images")
    parser.add_argument("--output_folder", type=str, default="./results")
    parser.add_argument("--OCRBench_file", type=str, default="./bench/Specific_EcommerceOCRBench.json")
    parser.add_argument("--model_path", type=str,
                        default='OpenGVLab/InternVL-Chat-Chinese-V1-1')  # TODO Set the address of your model's weights
    parser.add_argument("--save_name", type=str,
                        default="InternVL-Chat-Chinese-V1-1")  # TODO Set the name of the JSON file you save in the output_folder.
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
    model = AutoModel.from_pretrained(
        checkpoint,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map='cuda').eval()

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    for i in tqdm(range(len(data))):
        img_path = os.path.join(args.image_folder, data[i]['image_path'])
        qs = data[i]['question']
        image = Image.open(img_path).convert('RGB')
        image = image.resize((448, 448))
        image_processor = CLIPImageProcessor.from_pretrained(checkpoint)

        pixel_values = image_processor(images=image, return_tensors='pt').pixel_values
        pixel_values = pixel_values.to(torch.bfloat16).cuda()

        generation_config = dict(
            num_beams=1,
            max_new_tokens=512,
            do_sample=False,
        )
        response = model.chat(tokenizer, pixel_values, qs, generation_config)
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
