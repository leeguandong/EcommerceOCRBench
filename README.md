# EcommerceOCRBench

>
探索电商场景的OCR基准测试，电商场景有海量的ocr应用，在原始的应用链路中，总是基于文字检测+文字识别+后处理链路，现在基于多模态大语言模型，在基础的base模型上，可以一步到位取代原始链路做出巨大进步，因此EcommerceOCRBench应运而生。

# News

* ```2024.8.7 ``` 🚀 We realese EcommerceOCRBench, to explore the Capabilities of Large Multimodal Models on ecommerce
  Text.

# Data

| Data                   | Link                                                         | Description         |
|------------------------|--------------------------------------------------------------|---------------------|
| Full Test Json         | [Full Test](./bench/Total_EcommerceOCRBench.json)            | all data，包含大量内部测试数据 |
| EcommerceOCRBench Json | [EcommerceOCRBench](./bench/Specific_EcommerceOCRBench.json) | 开源出来的几个低质量的测评数据集    |

# EcommerceOCRBench

EcommerceOCRBench is a comprehensive evaluation benchmark designed to assess the OCR capabilities of Large Multimodal Models.

All EcommerceOCRBench images can download from baidu 链接: https://pan.baidu.com/s/1ZyEj_Z01_G5QAGiFlGOnaQ?pwd=ch7y 提取码: ch7y

# Evaluation

The test code for evaluating models in the paper can be found in [scripts](./scripts). Before conducting the evaluation,
you need to configure the model weights and environment based on the official code link provided in the scripts.

Example evaluation scripts:

```python

python. / scripts / monkey.py - -image_folder. / OCRBench_Images - -OCRBench_file. / OCRBench / OCRBench.json - -save_name
Monkey_OCRBench - -num_workers
GPU_Nums  # Test on OCRBench
python. / scripts / monkey.py - -image_folder. / OCRBench_Images - -OCRBench_file. / OCRBench / FullTest.json - -save_name
Monkey_FullTest - -num_workers
GPU_Nums  # Full Test

```

# Other Related Multilingual Datasets

| Data                                             | Link                                          | Description                                                                                                          |
|--------------------------------------------------|-----------------------------------------------|----------------------------------------------------------------------------------------------------------------------|
| EST-VQA Dataset (CVPR 2020, English and Chinese) | [Link](https://github.com/xinke-wang/EST-VQA) | On the General Value of Evidence, and Bilingual Scene-Text Visual Question Answering.                                |
| Swahili Dataset (ICDAR 2024)                     | [Link](https://arxiv.org/abs/2405.11437)      | The First Swahili Language Scene Text Detection and Recognition Dataset.                                             |
| Urdu Dataset (ICDAR 2024)                        | [Link](https://arxiv.org/abs/2405.12533)      | Dataset and Benchmark for Urdu Natural Scenes Text Detection, Recognition and Visual Question Answering.             |
| MTVQA (9 languages)                              | [Link](https://arxiv.org/abs/2405.11985)      | MTVQA: Benchmarking Multilingual Text-Centric Visual Question Answering.                                             |
| EVOBC (Oracle Bone Script Evolution Dataset)     | [Link](https://arxiv.org/abs/2401.12467)      | We systematically collected ancient characters from authoritative texts and websites spanning six historical stages. |
| HUST-OBC (Oracle Bone Script Character Dataset)  | [Link](https://arxiv.org/abs/2401.15365)      | For deciphering oracle bone script characters.                                                                       |



