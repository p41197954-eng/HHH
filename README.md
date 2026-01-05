# Anonymous Submission

Due to the anonymous GitHub server crashing during submission, we are providing the code in this manner, and all information related to the author has been completely removed from the documentation and this account.
All identifying information has been removed to comply with double-blind review requirements.
---

## Overview

We study the privacy risks of LLM-based recommender systems under the in-context learning paradigm by evaluating multiple **membership inference attack strategies**. The attacks probe whether a user–item interaction has appeared in the training data of the underlying model.

The implemented attacks include:
1. **Inquiry-based Attack**
2. **Similarity-based Attack**
3. **Memorization-based Attack**
4. **Poisoning-based Attack**

The framework is model-agnostic and operates purely through black-box API access.

---

## Supported Language Models

All LLMs are accessed **via the Ollama API**, ensuring reproducibility without distributing proprietary or large model weights.

The following models are supported:

- **LLaMA3-8B**
- **LLaMA4-Scout**
- **Mistral-7B**
- **Gemma3:4B**
- **GPT-OSS (20B)**
- **GPT-OSS (120B)**

Model selection is controlled through command-line arguments.

> **Note:** No model weights are included in this repository.

---

## Dataset

Due to storage capacity limitations, we only provide the **MovieLens** dataset in this repository.

- MovieLens is sufficient to reproduce all main experimental results reported in the paper.
- The data are preprocessed into an ICL-compatible recommendation format.

The codebase is designed to support additional datasets, which can be plugged in following the same interface.

---

## Experimental Scope

Our evaluation spans a total of **200 experimental configurations**, obtained by varying:
- the target LLM (6 models),
- the number of in-context demonstrations (3 shot settings),
- the position of poisoned examples in the prompt (5 positions), and
- the poisoning attack strategy (3 variants).

This design enables a systematic analysis of how prompt composition and attack choices jointly affect membership inference vulnerability.

## Running an Inquiry Attack

The inquiry-based membership inference attack can be executed with the following command:

```bash
python3 inquiry.py \
  --models llama3 \
  --datasets ml1m \
  --num_seeds 100 \
  --all_shots 1 \
  --positions end
```
## Running an Similarity Attack

The inquiry-based membership inference attack can be executed with the following command:
```bash
python3 semantic.py \
  --models llama3 \
  --datasets ml1m \
  --num_seeds 100 \
  --all_shots 1 \
  --positions end
```

## Running an Memorization Attack

The memorization-based membership inference attack can be executed with the following command:
```bash
python3 repeat.py \
  --models llama3 \
  --datasets ml1m \
  --num_seeds 100 \
  --all_shots 1 \
  --positions end
```

## Running an Poisoning Attack

The poisoning-based membership inference attack can be executed with the following command:
```bash
python3 repeat.py \
  --models llama3 \
  --datasets ml1m \
  --num_seeds 100 \
  --all_shots 1 \
  --positions end \
  --poison_num 1
```

