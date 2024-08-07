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

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path


# https://github.com/haotian-liu/LLaVA/blob/main/llava/eval/model_vqa_loader.py

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
    parser.add_argument("--model_path", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--model_base", type=str, default=None)
    parser.add_argument("--save_name", type=str, default="llava1_5_7b")
    parser.add_argument("--conv_mode", type=str, default="vicuna_v1")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
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
    device = f"cuda:{eval_id}"
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path=model_path,
                                                                           model_base=args.model_base,
                                                                           model_name=model_name, device=device)
    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(
            f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')
    for i in tqdm(range(len(data))):
        img_path = os.path.join(args.image_folder, data[i]['image_path'])
        qs = data[i]['question']
        qs = qs + "\nAnswer the question using a single word or phrase."
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image = Image.open(img_path).convert('RGB')
        image_tensor = process_images([image], image_processor, model.config)
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0)
        if data[i].get("predict", 0) != 0:
            print(f"{img_path} predict exist, continue.")
            continue

        stop_str = conv_templates[args.conv_mode].sep if conv_templates[
                                                             args.conv_mode].sep_style != SeparatorStyle.TWO else \
        conv_templates[args.conv_mode].sep2
        input_ids = input_ids.to(device=device, non_blocking=True)
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.to(dtype=torch.float16, device=device, non_blocking=True),
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=128,
                use_cache=True)

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()

        data[i]['predict'] = outputs
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
