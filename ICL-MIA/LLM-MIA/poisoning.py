import sys
import os
import re
import argparse
import torch
import pickle
import random
import requests
import json
from copy import deepcopy
from sentence_transformers import SentenceTransformer, util
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils import (
    load_dataset,
)

def main(models, datasets, num_seeds, positions, all_shots,poison_num):

    default_params = {}

    # list of all experiment parameters to run
    all_params = []
    for model in models:
        for dataset in datasets:
            dataset_primary = load_dataset_primary(dataset)
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


    semantic_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    #semantic_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device="cuda")

    for param_index, params in enumerate(all_params):
        prompt_subset = prepare_data(params)

        member_pool = prompt_subset[:len(prompt_subset)//2]
        nonmember_pool = prompt_subset[len(prompt_subset)//2:]

        random.seed(params["seed"])
        member_sentences = random.sample(member_pool, params['num_shots'])
        nonmember_sentences = random.sample(nonmember_pool, params['num_shots'])

        target_sentence = member_sentences[-1] if params['position'] == 'end' else member_sentences[0]
        nontarget_sentence = nonmember_sentences[0]

        required_for_mem = posion(params, member_sentences, target_sentence,semantic_model,dataset_primary,poison_num)
        if required_for_mem is None:
            continue

        print(100 * '-')
        required_for_nonmem = posion(params, member_sentences, nontarget_sentence,semantic_model,dataset_primary,poison_num)
        if required_for_nonmem is None:
            continue

        all_member_list.append(required_for_mem)
        all_nonmember_list.append(required_for_nonmem)

        save_path = f"../results/poison/{params['dataset']}/{poison_num}/{params['model']}/{params['position']}/{params['num_shots']}_shots/"
        os.makedirs(save_path, exist_ok=True)

        with open(os.path.join(save_path, 'member.pkl'), "wb") as file:
            pickle.dump(all_member_list, file)
        with open(os.path.join(save_path, 'nonmember.pkl'), "wb") as file:
             pickle.dump(all_nonmember_list, file)


def load_dataset_primary(dataset_name):
    """
    Load dataset-specific primary item list.
    """
    if dataset_name.lower() == "beauty":
        file_path = "../data/Beauty/asin_title_map.csv"
        beauty_df = pd.read_csv(file_path)
        beauty_dataset_primary = beauty_df['clean_title'].dropna().tolist()
        return beauty_dataset_primary
    elif dataset_name.lower() == 'ml1m':
        file_path = "../data/IMDB/title.basics.tsv"
        IMDB = pd.read_csv(file_path, sep="\t", low_memory=False)
        IMDB_dataset_primary = list(IMDB['primaryTitle'].dropna())
        return IMDB_dataset_primary
    elif dataset_name.lower() == 'book':
        file_path = "../data/Book/asin_title_map.csv"
        book_df = pd.read_csv(file_path)
        book_dataset_primary = book_df['clean_title'].dropna().tolist()
        return book_dataset_primary
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

def prepare_data(params):
    print("\nExperiment name:", params["expr_name"])
    prompted_sentences = load_dataset(params)
    return prompted_sentences

def posion(params, member_sentences, test_sentence,semantic_model,dataset_primary, poison_number):
    if params["dataset"] == "ml1m":
        int_match = re.search(r'watched (.+?) and based on', test_sentence)
        rec_match = re.search(r'in the following:\s*(.+)$', test_sentence, flags=re.IGNORECASE)
        if int_match and rec_match:
            interaction_list = int_match.group(1).split('|')
            rec_list = rec_match.group(1)
            print(f"interaction_list:{interaction_list}")
            print(f"rec_list:{rec_list}")
        else:
            return None

        IMDB_sampled = random.sample(dataset_primary, 1000)

        IMDB_sampled_embeddings = semantic_model.encode(IMDB_sampled, convert_to_tensor=True)
        selected_movies = random.sample(interaction_list, poison_number)



        for movie in selected_movies:
            movie_embedding = semantic_model.encode(movie, convert_to_tensor=True)

            cos_scores = util.pytorch_cos_sim(movie_embedding, IMDB_sampled_embeddings)[0]

            best_idx = cos_scores.argmax().item()
            replacement_movie = IMDB_sampled[best_idx]

            movie_idx = interaction_list.index(movie)
            interaction_list[movie_idx] = replacement_movie

            print(f"Replaced '{movie}' ➝ '{replacement_movie}'")

        new_watched_str = '|'.join(interaction_list)
        query_sentence = (
            f"The user has watched the following movies: {new_watched_str}. "
            "Based on this watch history, please recommend the top 10 movies with descending order"
            "the user is most likely to watch next. "
            "Format the output as a numbered list of movie titles only. "
            "Do not include descriptions, dates, or any other text."
        )

        input_to_model = construct_prompt_cut(params, member_sentences, query_sentence)
        print(f"input_to_model: {input_to_model}")

        base_sentence = continue_generate(input_to_model, query_sentence, params["model"])
        print(f"return_sentence: {base_sentence}")

        #poison_movie_list = re.findall(r'\d+\.\s+(.*)', base_sentence)
        poison_movie_list = re.findall(r'^\s*\d+\.?\s+(.+)$',base_sentence,flags=re.MULTILINE)
        poison_list = "|".join(poison_movie_list)
        print(f"poison_movie_list: {poison_movie_list}")

        if len(poison_movie_list) == 0 or len(rec_list) == 0:
            return 0

        original_rec = semantic_model.encode(rec_list, convert_to_tensor=True)
        poision_rec = semantic_model.encode(poison_list, convert_to_tensor=True)

        semantic_gap = util.pytorch_cos_sim(original_rec, poision_rec).item()

        return semantic_gap
    elif params["dataset"] == "beauty":
        user_match = re.search(r'The user with id (\d+)', test_sentence)
        int_match = re.search(r'bought (.+?) and based on', test_sentence)
        rec_match = re.search(r'in the following:\s*(.+)$', test_sentence, flags=re.IGNORECASE)
        if user_match and int_match and rec_match:
            user_id = int(user_match.group(1))
            interaction_list = int_match.group(1).split('|')
            rec_list = rec_match.group(1)
            print(f"interaction_list:{interaction_list}")
            print(f"rec_list:{rec_list}")
        else:
            return None

        beauty_sampled = random.sample(dataset_primary, 1000)
        beauty_sampled_embeddings = semantic_model.encode(beauty_sampled, convert_to_tensor=True)
        selected_beauty = random.sample(interaction_list, poison_number)

        for beauty in selected_beauty:
            beauty_embedding = semantic_model.encode(beauty, convert_to_tensor=True)

            cos_scores = util.pytorch_cos_sim(beauty_embedding, beauty_sampled_embeddings)[0]

            best_idx = cos_scores.argmax().item()
            replacement_beauty = beauty_sampled[best_idx]

            beauty_idx = interaction_list.index(beauty)
            interaction_list[beauty_idx] = replacement_beauty

            print(f"Replaced '{beauty}' ➝ '{replacement_beauty}'")

        new_bought_str = '|'.join(interaction_list)
        query_sentence = (
            f"The user has bought the following beauty product: {new_bought_str}. "
            "Based on this purchased history, please recommend the top 10 beauty products with descending order"
            "the user is most likely to buy next. "
            "Format the output as a numbered list of beauty product titles only. "
            "Do not include descriptions, dates, or any other text."
        )

        input_to_model = construct_prompt_cut(params, member_sentences, query_sentence)
        print(f"input_to_model: {input_to_model}")

        base_sentence = continue_generate(input_to_model, query_sentence, params["model"])
        print(f"return_sentence: {base_sentence}")

        poison_beauty_list = re.findall(r'\d+\.\s+(.*)', base_sentence)
        poison_list = "|".join(poison_beauty_list)
        print(f"poison_beauty_list: {poison_beauty_list}")

        original_rec = semantic_model.encode(rec_list, convert_to_tensor=True)
        poision_rec = semantic_model.encode(poison_list, convert_to_tensor=True)

        semantic_gap = util.pytorch_cos_sim(original_rec, poision_rec).item()

        return semantic_gap
    elif params["dataset"] == "book":
        user_match = re.search(r'The user with id (\d+)', test_sentence)
        int_match = re.search(r'bought these books (.+?) and based on', test_sentence)
        rec_match = re.search(r'in the following:\s*(.+)$', test_sentence, flags=re.IGNORECASE)
        if user_match and int_match and rec_match:
            user_id = int(user_match.group(1))
            interaction_list = int_match.group(1).split('|')
            rec_list = rec_match.group(1)
            print(f"interaction_list:{interaction_list}")
            print(f"rec_list:{rec_list}")
        else:
            return None

        book_sampled = random.sample(dataset_primary, 1000)
        book_sampled_embeddings = semantic_model.encode(book_sampled, convert_to_tensor=True)
        selected_books = random.sample(interaction_list, poison_number)

        for book in selected_books:
            book_embedding = semantic_model.encode(book, convert_to_tensor=True)

            cos_scores = util.pytorch_cos_sim(book_embedding, book_sampled_embeddings)[0]

            best_idx = cos_scores.argmax().item()
            replacement_book = book_sampled[best_idx]

            book_idx = interaction_list.index(book)
            interaction_list[book_idx] = replacement_book

            print(f"Replaced '{book}' ➝ '{replacement_book}'")

        new_bought_str = '|'.join(interaction_list)

        query_sentence = (
            f"The user has bought the following books: {new_bought_str}. "
            "Based on this purchased history, please recommend the top 10 books with descending order"
            "the user is most likely to buy next. "
            "Format the output as a numbered list of book titles only. "
            "Do not include descriptions, dates, or any other text."
        )

        input_to_model = construct_prompt_cut(params, member_sentences, query_sentence)
        print(f"input_to_model: {input_to_model}")

        base_sentence = continue_generate(input_to_model, query_sentence, params["model"])
        print(f"return_sentence: {base_sentence}")

        poison_book_list = re.findall(r'\d+\.\s+(.*)', base_sentence)
        poison_list = "|".join(poison_book_list)
        print(f"poison_book_list: {poison_book_list}")

        original_rec = semantic_model.encode(rec_list, convert_to_tensor=True)
        poision_rec = semantic_model.encode(poison_list, convert_to_tensor=True)

        semantic_gap = util.pytorch_cos_sim(original_rec, poision_rec).item()

        return semantic_gap
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
    parser.add_argument("--poison_num", type=int, required=True)

    args = vars(parser.parse_args())
    args["models"] = convert_to_list(args["models"])
    args["datasets"] = convert_to_list(args["datasets"])
    args["positions"] = convert_to_list(args["positions"])
    args["all_shots"] = convert_to_list(args["all_shots"], is_int=True)
    main(**args)
