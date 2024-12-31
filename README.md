# DELL Repository
Repository for ACL2024 Findings: [DELL: Generating Reactions and Explanations for LLM-Based Misinformation Detection](https://arxiv.org/abs/2402.10426)

All resources are available on [Google Drive](https://drive.google.com/drive/folders/1nPo6x3AY7Kt1Nb9DUUvBAhpMwad3zhvq?usp=sharing)

# Basic Usage

DELL mainly contains three core parts: (i) **D**iverse Reaction Generation; (ii) **E**xplainable Proxy Tasks; (iii) **LL**M-Based Expert Ensemble.  

You can follow the steps below to reproduce our results or train DELL on your dataset.

## data

This folder includes our employed datasets, dataset split, the generated user-news networks, and the output of each proxy task.

You can download them from [Google Drive](https://drive.google.com/drive/folders/1nPo6x3AY7Kt1Nb9DUUvBAhpMwad3zhvq?usp=sharing).

Choose data_mistral-v3 or data_mistral-v1 and unzip it. V1 is adopted in the original paper. However, we have lost some files due to the sever failure. Thus, we re-generated all files using Mistral-7B-Instruct-v0.3 (data_mistral-v3)

If you want to adopt DELL on your dataset, please put your dataset in the folder of the corresponding task. (TASK1: Fake News Detection; TASK2: Framing Detection; TASK3: Propaganda Tactic Detection.)

**D**iverse Reaction Generation and **E**xplainable Proxy Tasks apply to any social text. You can generate diverse user reaction networks for any social text.

## networks

The code 'main.py' provides the function 'generate(text)', where the input is a social text and the output is the user-news interaction networks.

You can adjust parameters size_limit, rt_ratio, and height_ratio to simulate networks with different characteristics.

You can obtain the networks by running:

```
cd networks
python main.py
```

We also provide the networks we generated in our experiments in the folder `data/networks`.

## proxy

We design 4 content-based proxy tasks:

- Sentiment analysis
- Framing Detection
- Propaganda Tactics Detection
- Knowledge Retrieval

and 2 comment-based proxy tasks:
- Stance Detection
- Response Characterization

You can obtain the output of proxy tasks by running:
```
cd proxy
python content.py
python search_entity_from_wiki.py
python retrieval.py
python stance.py
python response.py
```

We also provide the output we generated in our experiments in the folder `data/proxy`.

## expert
For each proxy task, we train a task-specific expert model by running:
```
cd expert
python train.py --task [task] --dataset_name [dataset_name] --text_augmentation [content-based proxy] --graph_augmentation [comment-based proxy]
```
--text_augmention are shorts for content-based proxy tasks:

[emotion] for sentiment analysis, [framing] for framing detection, [propaganda] for propaganda tactics detection, and [retrieval] for Knowledge Retrieval.

--graph_augmention are shorts for comment-based proxy tasks:

[stance] for stance detection and [relation] for response characterization

Then, you can obtain the predictions of each expert by running:
```
python inference.py
```

We also provide the predictions we generated in our experiments in the folder `expert/results`.


## ensemble
We provide three LLM-based methods for prediction ensemble: vanilla, confidence, and selective.

You can obtain the final predictions by running:
```
cd ensemble
python [vallina/confidence/selective].py
```

We also provide the final predictions we generated in our experiments in the folder `ensemble/results`.

# Citation
If you find our work interesting/helpful, please consider citing DELL:
```
@inproceedings{wan-etal-2024-dell,
    title = "{DELL}: Generating Reactions and Explanations for {LLM}-Based Misinformation Detection",
    author = "Wan, Herun  and
      Feng, Shangbin  and
      Tan, Zhaoxuan  and
      Wang, Heng  and
      Tsvetkov, Yulia  and
      Luo, Minnan",
    editor = "Ku, Lun-Wei  and
      Martins, Andre  and
      Srikumar, Vivek",
    booktitle = "Findings of the Association for Computational Linguistics ACL 2024",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand and virtual meeting",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-acl.155",
    pages = "2637--2667",
    abstract = "Large language models are limited by challenges in factuality and hallucinations to be directly employed off-the-shelf for judging the veracity of news articles, where factual accuracy is paramount. In this work, we propose DELL that identifies three key stages in misinformation detection where LLMs could be incorporated as part of the pipeline: 1) LLMs could generate news reactions to represent diverse perspectives and simulate user-news interaction networks; 2) LLMs could generate explanations for proxy tasks (e.g., sentiment, stance) to enrich the contexts of news articles and produce experts specializing in various aspects of news understanding; 3) LLMs could merge task-specific experts and provide an overall prediction by incorporating the predictions and confidence scores of varying experts. Extensive experiments on seven datasets with three LLMs demonstrate that DELL outperforms state-of-the-art baselines by up to 16.8{\%} in macro f1-score. Further analysis reveals that the generated reactions and explanations are greatly helpful in misinformation detection, while our proposed LLM-guided expert merging helps produce better-calibrated predictions.",
}
```
```
@article{wan2024dell,
  title={DELL: Generating Reactions and Explanations for LLM-Based Misinformation Detection},
  author={Wan, Herun and Feng, Shangbin and Tan, Zhaoxuan and Wang, Heng and Tsvetkov, Yulia and Luo, Minnan},
  journal={arXiv preprint arXiv:2402.10426},
  year={2024}
}
```

# Questions?
Feel free to open issues in this repository! Instead of emails, Github issues are much better at facilitating a conversation between you and our team to address your needs. You can also contact Herun Wan through `wanherun@stu.xjtu.edu.cn`.



# Updating


### 20241231
- We updated our codes, which are adapted to Mistral-7B-Instruct-v0.3.
- We uploaded user networks and the outputs of proxy tasks generated by Mistral-7B-Instruct-v0.3.
- We plan to re-train DELL using the re-generated data.

### 20240530
- We upload all related resources on [Google Drive](https://drive.google.com/drive/folders/1nPo6x3AY7Kt1Nb9DUUvBAhpMwad3zhvq?usp=sharing).
- We uploaded the complete code.
- We provide a simple tutorial on using DELL.


### 20240529
- We upload the network generation code and proxy task code.
- We are uploading the output of the proxy task on Google Drive.

### 20240519
- Our paper has been accepted to the ACL Findings.
- We upload the employed dataset and the generated user-news networks. Now the folder *data* is available.

### before
- We upload the core codes of DELL, but it's missing a lot of details.

