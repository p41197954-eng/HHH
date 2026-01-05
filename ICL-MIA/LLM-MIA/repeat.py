import sys
import os
import re
import argparse
import pickle
import random
import requests
import json
from copy import deepcopy

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

    # Load the model once

    all_member_list = []
    all_nonmember_list = []

    for param_index, params in enumerate(all_params):
        prompt_subset = prepare_data(params)

        member_pool = prompt_subset[:len(prompt_subset) // 2]
        nonmember_pool = prompt_subset[len(prompt_subset) // 2:]

        random.seed(params["seed"])
        member_sentences = random.sample(member_pool, params['num_shots'])
        nonmember_sentences = random.sample(nonmember_pool, params['num_shots'])

        target_sentence = member_sentences[-1] if params['position'] == 'end' else member_sentences[0]
        nontarget_sentence = nonmember_sentences[0]

        required_for_mem = repeat(params, member_sentences, target_sentence)
        if required_for_mem is None:
            continue

        print(100 * '-')
        required_for_nonmem = repeat(params, member_sentences, nontarget_sentence)
        if required_for_nonmem is None:
            continue

        all_member_list.append(required_for_mem)
        all_nonmember_list.append(required_for_nonmem)

        save_path = f"../results/repeat/{params['dataset']}/{params['model']}/{params['position']}/{params['num_shots']}_shots/"
        os.makedirs(save_path, exist_ok=True)

        with open(os.path.join(save_path, 'member.pkl'), "wb") as file:
            pickle.dump(all_member_list, file)
        with open(os.path.join(save_path, 'nonmember.pkl'), "wb") as file:
            pickle.dump(all_nonmember_list, file)


def prepare_data(params):
    print("\nExperiment name:", params["expr_name"])
    prompted_sentences = load_dataset(params)
    return prompted_sentences


def repeat(params, member_sentences, test_sentence):
    if params['dataset'] == 'ml1m':
        int_match = re.search(r'watched\s+(.*?)\s+and based on his or her watched history', test_sentence)
        rec_match = re.search(r'in the following:\s*(.+)$', test_sentence, flags=re.IGNORECASE)
        if int_match and rec_match:
            interaction_list = int_match.group(1)
            rec_list = rec_match.group(1).split('|')
            print(f"interaction_list:{interaction_list}")
            print(f"rec_list:{rec_list}")
        else:
            return None

        query_sentence = (
            f"The user has watched the following movies: {interaction_list}. "
            "Based on this watch history, please recommend the top 10 movies with descending order"
            "the user is most likely to watch next. "
            "Format the output as a numbered list of movie titles only. "
            "Do not include descriptions, dates, or any other text."
        )
        '''
        query_sentence = (
            f"The user with id {user_id} watched {interaction_list} "
            f"and based on his or her watched history, "
            f"Please recommend top-10 movies with descending order for {user_id}? "
            f"Only give movie name with a list and not give any description.\n"
            f"""for example:
            1.  xxx
            2.  xxx
            3.  xxx
            4.  xxx
            5.  xxx
            6.  xxx
            7.  xxx
            8.  xxx
            9.  xxx
            10.  xxx
            """
        )
        '''
        input_to_model = construct_prompt_cut(params, member_sentences, query_sentence)
        print(f"input_to_model: {input_to_model}")

        base_sentence = continue_generate(input_to_model, query_sentence, params["model"])
        print(f"return_sentence: {base_sentence}")

        new_movie_list = re.findall(r'^\s*\d+\.?\s+(.+)$', base_sentence, flags=re.MULTILINE)
        print(f"new_movie_list: {new_movie_list}")

        repeat_num = 0

        orginal_rec = [m.lower().replace(" ", "") for m in rec_list]
        updated_rec = [m.lower().replace(" ", "") for m in new_movie_list]

        for m in orginal_rec:
            if m in updated_rec:
                repeat_num += 1

        return repeat_num
    elif params['dataset'] == 'book':
        int_match = re.search(r'bought\s+(.*?)\s+and based on his or her purchased history', test_sentence)
        rec_match = re.search(r'in the following:\s*(.+)$', test_sentence, flags=re.IGNORECASE)
        if int_match and rec_match:
            interaction_list = int_match.group(1)
            rec_list = rec_match.group(1).split('|')
            print(f"interaction_list:{interaction_list}")
            print(f"rec_list:{rec_list}")
        else:
            return None

        query_sentence = (
            f"The user has bought the following books: {interaction_list}. "
            "Based on this purchased history, please recommend the top 10 books with descending order"
            "the user is most likely to read next. "
            "Format the output as a numbered list of book titles only. "
            "Do not include descriptions, dates, or any other text."
        )

        input_to_model = construct_prompt_cut(params, member_sentences, query_sentence)
        print(f"input_to_model: {input_to_model}")

        base_sentence = continue_generate(input_to_model, query_sentence, params["model"])
        print(f"return_sentence: {base_sentence}")

        new_book_list = re.findall(r'^\s*\d+\.?\s+(.+)$', base_sentence, flags=re.MULTILINE)
        print(f"new_book_list: {new_book_list}")

        repeat_num = 0

        orginal_rec = [m.lower().replace(" ", "") for m in rec_list]
        updated_rec = [m.lower().replace(" ", "") for m in new_book_list]

        for m in orginal_rec:
            if m in updated_rec:
                repeat_num += 1

        return repeat_num
    elif params['dataset'] == 'beauty':
        int_match = re.search(r'bought\s+(.*?)\s+and based on his or her bought history', test_sentence)
        rec_match = re.search(r'in the following:\s*(.+)$', test_sentence, flags=re.IGNORECASE)
        if int_match and rec_match:
            interaction_list = int_match.group(1)
            rec_list = rec_match.group(1).split('|')
            print(f"interaction_list:{interaction_list}")
            print(f"rec_list:{rec_list}")
        else:
            return None

        query_sentence = (
            f"The user has bought the following beauty product: {int_match}. "
            "Based on this purchased history, please recommend the top 10 beauty products with descending order"
            "the user is most likely to buy next. "
            "Format the output as a numbered list of beauty product titles only. "
            "Do not include descriptions, dates, or any other text."
        )

        input_to_model = construct_prompt_cut(params, member_sentences, query_sentence)
        print(f"input_to_model: {input_to_model}")

        base_sentence = continue_generate(input_to_model, query_sentence, params["model"])
        print(f"return_sentence: {base_sentence}")

        new_beauty_list = re.findall(r'^\s*\d+\.?\s+(.+)$', base_sentence, flags=re.MULTILINE)
        print(f"new_beauty_list: {new_beauty_list}")

        repeat_num = 0

        orginal_rec = [m.lower().replace(" ", "") for m in rec_list]
        updated_rec = [m.lower().replace(" ", "") for m in new_beauty_list]

        for m in orginal_rec:
            if m in updated_rec:
                repeat_num += 1

        return repeat_num
    else:
        raise Exception(f"Unknown dataset: {params['dataset']}")


def continue_generate(prompt_setup, prompt_question, model, max_token=256, temperature=0.0):
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
        return raw_output
    except requests.RequestException as e:
        print(f"[Error] Request to Ollama chat API failed: {e}")
        return -1


def construct_prompt_cut(params, member_sentence, query_sentence):
    prompt = params.get("prompt_prefix", "")
    print(f"type of member_sentence: {type(member_sentence)}")
    prompt += "\n".join(member_sentence) + "\n\n" + query_sentence
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
