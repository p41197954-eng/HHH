

def load_ml1m():
    prompt_sentences = []
    with open(f"../data/processed_ml-1m/prompt_ml1m.txt", "r") as prompt_data:
        for line in prompt_data:
            prompt_sentences.append(line)
    return prompt_sentences

def load_book():
    prompt_sentences = []
    with open(f"../data/Book/prompt_Book.txt", "r") as prompt_data:
        for line in prompt_data:
            prompt_sentences.append(line)
    return prompt_sentences

def load_beauty():
    prompt_sentences = []
    with open(f"../data/Beauty/prompt_Beauty.txt", "r") as prompt_data:
        for line in prompt_data:
            prompt_sentences.append(line)
    return prompt_sentences

def load_dataset(params):
    if params["dataset"] == "ml1m":
        prompt_sentences = load_ml1m()
        params[
            "prompt_prefix"
        ] = "Pretend you are a movie recommender system. And You task is to recommend top 10 movies which user will watch and recommended movies should not be user watched movies.\n\n"
        params["task_format"] = "recommendation"
    elif params["dataset"] == "book":
        prompt_sentences = load_book()
        params[
            "prompt_prefix"
        ] = "Pretend you are a book recommender system, and your task is to recommend the top 10 books that the user may buy. The recommended books should not include those already bought by the user.\n\n"
        params["task_format"] = "recommendation"
    elif params["dataset"] == "beauty":
        prompt_sentences = load_beauty()
        params[
            "prompt_prefix"
        ] = "Pretend you are a beauty and personal care product recommender system. And Your task is to recommend top 10 beauty and personal care product which user will buy and recommended product should not be user bought product.\n\n"
        params["task_format"] = "recommendation"
    else:
        raise NotImplementedError
    return prompt_sentences

def load_defense_dataset(params):
    if params["dataset"] == "ml1m":
        prompt_sentences = load_ml1m()
        params[
            "prompt_prefix"
        ] = "Pretend you are a movie recommender system. And You task is to recommend top 10 movies which user will watch and recommended movies should not be user watched movies.\n\n"
        params["task_format"] = "recommendation"
    elif params["dataset"] == "book":
        prompt_sentences = load_book()
        params[
            "prompt_prefix"
        ] = "Pretend you are a book recommender system, and your task is to recommend the top 10 books that the user may buy. The recommended books should not include those already bought by the user.\n\n"
        params["task_format"] = "recommendation"
    elif params["dataset"] == "beauty":
        prompt_sentences = load_beauty()
        params[
            "prompt_prefix"
        ] = "Pretend you are a beauty and personal care product recommender system. And Your task is to recommend top 10 beauty and personal care product which user will buy and recommended product should not be user bought product.\n\n"
        params["task_format"] = "recommendation"
    else:
        raise NotImplementedError
    return prompt_sentences

if __name__ == "__main__":
    params = {}
    params["dataset"] = "ml1m"
    load_dataset(params)
    print(params)
