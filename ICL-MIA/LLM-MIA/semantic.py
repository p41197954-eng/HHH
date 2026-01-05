import sys
import os
import re
import argparse
import pickle
import random
import requests
import json
from copy import deepcopy
import torch
from sentence_transformers import SentenceTransformer, util

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils import (
    load_dataset,
)

def main(models, datasets, num_seeds, positions, all_shots):

    default_params = {}

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

    semantic_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2',device='cuda')
    #semantic_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device="cuda")
    print("Embedding model device:", semantic_model.device)

    for param_index, params in enumerate(all_params):
        prompt_subset = prepare_data(params)
        #print(f"prompt_subset: {prompt_subset}")

        member_pool = prompt_subset[:len(prompt_subset)//2]
        nonmember_pool = prompt_subset[len(prompt_subset)//2:]

        random.seed(params["seed"])
        member_sentences = random.sample(member_pool, params['num_shots'])
        nonmember_sentences = random.sample(nonmember_pool, params['num_shots'])

        target_sentence = member_sentences[-1] if params['position'] == 'end' else member_sentences[0]

        nontarget_sentence = nonmember_sentences[0]

        required_for_mem = repeat(params, member_sentences, target_sentence, semantic_model)
        if required_for_mem is None:
            continue

        print(100 * '-')
        required_for_nonmem = repeat(params, member_sentences, nontarget_sentence,semantic_model)
        if required_for_nonmem is None:
            continue

        all_member_list.append(required_for_mem)
        all_nonmember_list.append(required_for_nonmem)

        save_path = f"../results/semantic/{params['dataset']}/{params['model']}/{params['position']}/{params['num_shots']}_shots/"
        os.makedirs(save_path, exist_ok=True)

        with open(os.path.join(save_path, 'member.pkl'), "wb") as file:
            pickle.dump(all_member_list, file)
        with open(os.path.join(save_path, 'nonmember.pkl'), "wb") as file:
            pickle.dump(all_nonmember_list, file)

def prepare_data(params):
    print("\nExperiment name:", params["expr_name"])
    prompted_sentences = load_dataset(params)
    return prompted_sentences

def repeat(params, member_sentences, test_sentence, semantic_model):
    if params['dataset'] == 'ml1m':
        hist_match = re.search(r"watched\s+(.*?)\s+and based on his or her watched", test_sentence)
        if hist_match:
            hist_list = hist_match.group(1)
        else:
            return None

        query_sentence = (
            f"The user has watched the following movies: {hist_list}. "
            "Based on this watch history, please recommend the top 10 movies with descending order"
            "the user is most likely to watch next. "
            "Format the output as a numbered list of movie titles only. "
            "Do not include descriptions, dates, or any other text."
        )

        input_to_model = construct_prompt_cut(params, member_sentences, query_sentence)
        print(f"input_to_model: {input_to_model}")
        return_sentence = continue_generate(input_to_model,query_sentence,params["model"])
        print(f"return_sentence: {return_sentence}")

        #movie_list = re.findall(r'^\s*\d+\.\s*(.+)$', return_sentence, flags=re.MULTILINE)
        movie_list = re.findall(r'^\s*\d+\.?\s+(.+)$',return_sentence,flags=re.MULTILINE)
        print(f"movie_list: {movie_list}")

        hist_list = [movie.strip() for movie in hist_list.split('|')]
        print(f"interaction_list: ", hist_list)

        if len(movie_list) == 0 or len(hist_list) == 0:
            return 0

        with torch.no_grad():
            interaction_embeddings = []
            for movie in hist_list:
                embedding = semantic_model.encode(movie, convert_to_tensor=True)
                interaction_embeddings.append(embedding)

            recommendation_embeddings = []
            for movie in movie_list:
                embedding = semantic_model.encode(movie, convert_to_tensor=True)
                recommendation_embeddings.append(embedding)

        # interaction mean embedding
        interaction_tensor = torch.stack(interaction_embeddings).to(semantic_model.device)  # [n, d]
        interaction_mean = interaction_tensor.mean(dim=0)  # [d]

        # recommendation mean embedding
        recommendation_tensor = torch.stack(recommendation_embeddings).to(semantic_model.device)  # [m, d]
        recommendation_mean = recommendation_tensor.mean(dim=0)  # [d]

        semantic_sim = util.cos_sim(interaction_mean,recommendation_mean).item()

        return semantic_sim
    elif params['dataset'] == 'book':
        hist_match = re.search(r"bought\s+(.*?)\s+and based on his or her purchased history", test_sentence)
        if hist_match:
            hist_list = hist_match.group(1)
        else:
            return None

        query_sentence = (
            f"The user has bought the following books: {hist_list}. "
            "Based on this purchased history, please recommend the top 10 books with descending order"
            "the user is most likely to read next. "
            "Format the output as a numbered list of book titles only. "
            "Do not include descriptions, dates, or any other text."
        )

        input_to_model = construct_prompt_cut(params, member_sentences, query_sentence)
        print(f"input_to_model: {input_to_model}")
        return_sentence = continue_generate(input_to_model, query_sentence, params["model"])
        print(f"return_sentence: {return_sentence}")

        # movie_list = re.findall(r'^\s*\d+\.\s*(.+)$', return_sentence, flags=re.MULTILINE)
        book_list = re.findall(r'^\s*\d+\.?\s+(.+)$', return_sentence, flags=re.MULTILINE)
        print(f"book_list: {book_list}")

        hist_list = [book.strip() for book in hist_list.split('|')]
        print(f"interaction_list: ", hist_list)

        if len(book_list) == 0 or len(hist_list) == 0:
            return 0

        with torch.no_grad():
            interaction_embeddings = []
            for book in hist_list:
                embedding = semantic_model.encode(book, convert_to_tensor=True)
                interaction_embeddings.append(embedding)

            recommendation_embeddings = []
            for book in book_list:
                embedding = semantic_model.encode(book, convert_to_tensor=True)
                recommendation_embeddings.append(embedding)

        # interaction mean embedding
        interaction_tensor = torch.stack(interaction_embeddings).to(semantic_model.device)  # [n, d]
        interaction_mean = interaction_tensor.mean(dim=0)  # [d]

        # recommendation mean embedding
        recommendation_tensor = torch.stack(recommendation_embeddings).to(semantic_model.device)  # [m, d]
        recommendation_mean = recommendation_tensor.mean(dim=0)  # [d]

        semantic_sim = util.cos_sim(interaction_mean, recommendation_mean).item()

        return semantic_sim
    elif params['dataset'] == 'beauty':
        hist_match = re.search(r"bought\s+(.*?)\s+and based on his or her bought history", test_sentence)
        if hist_match:
            hist_list = hist_match.group(1)
        else:
            return None

        query_sentence = (
            f"The user has bought the following beauty product: {hist_list}. "
            "Based on this purchased history, please recommend the top 10 beauty products with descending order"
            "the user is most likely to buy next. "
            "Format the output as a numbered list of beauty product titles only. "
            "Do not include descriptions, dates, or any other text."
        )

        input_to_model = construct_prompt_cut(params, member_sentences, query_sentence)
        print(f"input_to_model: {input_to_model}")
        return_sentence = continue_generate(input_to_model, query_sentence, params["model"])
        print(f"return_sentence: {return_sentence}")

        # movie_list = re.findall(r'^\s*\d+\.\s*(.+)$', return_sentence, flags=re.MULTILINE)
        beauty_list = re.findall(r'^\s*\d+\.?\s+(.+)$', return_sentence, flags=re.MULTILINE)
        print(f"beauty_list: {beauty_list}")

        hist_list = [beauty.strip() for beauty in hist_list.split('|')]
        print(f"interaction_list: ", hist_list)

        if len(beauty_list) == 0 or len(hist_list) == 0:
            return 0

        with torch.no_grad():
            interaction_embeddings = []
            for beauty in hist_list:
                embedding = semantic_model.encode(beauty, convert_to_tensor=True)
                interaction_embeddings.append(embedding)

            recommendation_embeddings = []
            for beauty in beauty_list:
                embedding = semantic_model.encode(beauty, convert_to_tensor=True)
                recommendation_embeddings.append(embedding)

        # interaction mean embedding
        interaction_tensor = torch.stack(interaction_embeddings).to(semantic_model.device)  # [n, d]
        interaction_mean = interaction_tensor.mean(dim=0)  # [d]

        # recommendation mean embedding
        recommendation_tensor = torch.stack(recommendation_embeddings).to(semantic_model.device)  # [m, d]
        recommendation_mean = recommendation_tensor.mean(dim=0)  # [d]

        semantic_sim = util.cos_sim(interaction_mean, recommendation_mean).item()

        return semantic_sim
    else:
        raise Exception(f"Unknown dataset: {params['dataset']}")


def continue_generate(prompt_setup, prompt_question, model, max_token = 256, temperature=0.0):
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

def construct_prompt_cut(params, train_sentences, query_sentence):
    prompt = params.get("prompt_prefix", "")
    prompt += "\n".join(train_sentences) + "\n\n" + query_sentence
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
