# DELL Repository
Repository for ACL2024 Findings: [DELL: Generating Reactions and Explanations for LLM-Based Misinformation Detection](https://arxiv.org/abs/2402.10426)

All resources are available on [Google Drive](https://drive.google.com/drive/folders/1nPo6x3AY7Kt1Nb9DUUvBAhpMwad3zhvq?usp=sharing)

# Basic Usage

DELL mainly contains three core parts: (i) **D**iverse Reaction Generation; (ii) **E**xplainable Proxy Tasks; (iii) **LL**M-Based Expert Ensemble.  

You can follow the steps below to reproduce our results or train DELL on your dataset.

## data

This folder includes our employed datasets, dataset split, the generated user-news networks, and the output of each proxy task.

You can download them from [Google Drive](https://drive.google.com/drive/folders/1nPo6x3AY7Kt1Nb9DUUvBAhpMwad3zhvq?usp=sharing).

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

