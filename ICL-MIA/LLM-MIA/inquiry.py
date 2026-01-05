import sys
import os
import re
import argparse
import pickle
import random
from copy import deepcopy
import requests
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils import (
    load_dataset,
)

def main(models, datasets, num_seeds, positions, all_shots):
    default_params = {}

    # list of all experiment parameters to run
    all_params = []
    for model in models:
        for dataset in datasets:
            for position in positions:
                for num_shots in all_shots:
                    for seed in range(num_seeds):
                        p = deepcopy(default_params)
                        p["model"] = model
                        p["dataset"] = dataset
                        p["seed"] = seed
                        p["num_shots"] = num_shots
                        p['position'] = position
                        p["expr_name"] = f"{p['dataset']}_{p['model']}_subsample_seed{p['seed']}"
                        all_params.append(p)

    all_member_list = []
    all_nonmember_list = []

    for param_index, params in enumerate(all_params):
        prompt_subset = prepare_data(params)

        member_pool = prompt_subset[:len(prompt_subset)//2]
        nonmember_pool = prompt_subset[len(prompt_subset)//2:]

        random.seed(params["seed"])
        member_sentences = random.sample(member_pool, params['num_shots'])
        nonmember_sentences = random.sample(nonmember_pool, params['num_shots'])

        target_sentence = member_sentences[-1] if params['position'] == 'end' else member_sentences[0]

        nontarget_sentence = nonmember_sentences[0]

        required_for_mem = inquiry(params, member_sentences, target_sentence)
        if required_for_mem is None:
            continue

        print(100 * '-')
        required_for_nonmem = inquiry(params, member_sentences, nontarget_sentence)
        if required_for_nonmem is None:
            continue

        all_member_list.append(required_for_mem)
        all_nonmember_list.append(required_for_nonmem)

        save_path = f"../results/inquiry/{params['dataset']}/{params['model']}/{params['position']}/{params['num_shots']}_shots/"
        os.makedirs(save_path, exist_ok=True)

        with open(os.path.join(save_path, 'member.pkl'), "wb") as file:
            pickle.dump(all_member_list, file)
        with open(os.path.join(save_path, 'nonmember.pkl'), "wb") as file:
            pickle.dump(all_nonmember_list, file)

def prepare_data(params):
    print("\nExperiment name:", params["expr_name"])
    prompted_sentences = load_dataset(params)
    return prompted_sentences

def inquiry(params, member_sentences, test_sentence):
    if params['dataset'] == 'ml1m':
        watched_content = re.search(r"watched\s+(.*?)\s+and based on", test_sentence)
        if watched_content:
            watched_content = watched_content.group(1)
        else:
            return None

        query_sentence = f"Have you seen the user interacted {watched_content} before? Please answer one word: Yes or No"
        #print(f"query_sentence: {query_sentence}")
        input_to_model = construct_prompt_omit(params, member_sentences)
        print(f"input_to_model: {input_to_model}")
        print(f"query_sentence: {query_sentence}")
        #return_idx = query_ollama(input_to_model, params['model'])
        return_idx = query_ollama_chat(input_to_model,  query_sentence, params['model'])
        return return_idx
    elif params['dataset'] == 'beauty':
        beauty_content = re.search(r"bought\s+(.*?)\s+and based on", test_sentence)
        if beauty_content:
            beauty_content = beauty_content.group(1)
        else:
            return None
        query_sentence = f"Have you seen the user interacted {beauty_content} before? Please answer one word: Yes or No"
        input_to_model = construct_prompt_omit(params, member_sentences)
        print(f"input_to_model: {input_to_model}")
        print(f"query_sentence: {query_sentence}")
        #return_idx = query_ollama(input_to_model, params['model'])
        return_idx = query_ollama_chat(input_to_model,  query_sentence, params['model'])
        return return_idx
    elif params['dataset'] == 'book':
        book_content = re.search(r"bought\s+(.*?)\s+and based on", test_sentence)
        if book_content:
            book_content = book_content.group(1)
        else:
            return None
        query_sentence = f"Have you seen the user interacted {book_content} before? Please answer one word: Yes or No"
        #print(f"query_sentence: {query_sentence}")
        input_to_model = construct_prompt_omit(params, member_sentences)
        print(f"input_to_model: {input_to_model}")
        print(f"query_sentence: {query_sentence}")
        #return_idx = query_ollama(input_to_model, params['model'])
        return_idx = query_ollama_chat(input_to_model,  query_sentence, params['model'])
        return return_idx

def query_ollama_chat(prompt_setup, prompt_question, model, max_token = 2, temperature=0.0):
    url = "http://localhost:11434/api/chat"
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt_setup},
            {"role": "user", "content": prompt_question}
        ],
        "max_tokens": max_token,
        "temperature": temperature,
        "stream": False
    }

    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        data = response.json()
        raw_output = data.get("message", {}).get("content", "").strip().lower()
        print(f"[Ollama chat output]: {raw_output}")

        raw_output = raw_output.split()[0].strip(",.?!")
        #print(f"[Ollama chat output]: {raw_output}")

        if raw_output.startswith("yes"):
            return 1
        elif raw_output.startswith("no"):
            return 0
        else:
            print("[Warning] Unexpected response format.")
            return 0

    except requests.RequestException as e:
        print(f"[Error] Request to Ollama chat API failed: {e}")
        return -1

def construct_prompt_omit(params, train_sentences):
    prompt = params.get("prompt_prefix", "")
    prompt += "\n".join(train_sentences)
    return prompt

def convert_to_list(items, is_int=False):
    return [int(s.strip()) if is_int else s.strip() for s in items.split(",")]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", required=True)
    parser.add_argument("--datasets", required=True)
    parser.add_argument("--num_seeds", type=int, required=True)
    parser.add_argument("--all_shots", required=True)
    parser.add_argument("--positions", required=True)

    args = vars(parser.parse_args())
    args["models"] = convert_to_list(args["models"])
    args["datasets"] = convert_to_list(args["datasets"])
    args["positions"] = convert_to_list(args["positions"])
    args["all_shots"] = convert_to_list(args["all_shots"], is_int=True)

    main(**args)
