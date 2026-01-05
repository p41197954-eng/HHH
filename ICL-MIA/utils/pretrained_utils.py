def load_pretrained_ml1m():
    prompt_sentences = []
    with open(f"../pretrained_RecSys/ml1m_train.txt", "r") as prompt_data:
        for line in prompt_data:
            prompt_sentences.append(line)
    return prompt_sentences

def load_pretrained_beauty():
    prompt_sentences = []
    with open(f"../pretrained_RecSys/beauty_train.txt", "r") as prompt_data:
        for line in prompt_data:
            prompt_sentences.append(line)
    return prompt_sentences

def load_pretrained_book():
    prompt_sentences = []
    with open(f"../pretrained_RecSys/book_train.txt", "r") as prompt_data:
        for line in prompt_data:
            prompt_sentences.append(line)
    return prompt_sentences


def load_pretrained_dataset(params):
    if params["dataset"] == "ml1m":
        prompt_sentences = load_pretrained_ml1m()
        params["prompt_prefix"] = (
            #"You are a Recommender Systems. "
            #"Continue user-item interactions list providing the next interaction "
            #"based on the MovieLens-1M dataset. "
            #"When given 'UserID::CurrentInteraction', respond with 'UserID::NextInteraction'. "
            #"\nBelow are examples of queries and their correct responses:\n\n"
            #"Follow this pattern strictly. Let's think step by step."
            #"For example:\n"
            #"user query: UserID::CurrentInteraction\n"
            #"system reponse: UserID::NextInteraction\n"
            "You are a recommender system trained on the MovieLens-1M dataset. "
            "Given an input in the format:\n"
            "UserID::CurrentInteractions like UserID::ItemID_1::ItemID_2::...::ItemID_N\n"
            "You must respond in the format:\n"
            "UserID::NextInteraction like UserID::ItemID_N+1\n"
            "Output ONLY the response in this format. No additional text.\n\n"
        )
        params["task_format"] = "recommendation"
    elif params["dataset"] == "beauty":
        prompt_sentences = load_pretrained_beauty()
        params[
            "prompt_prefix"
        ] = (
            "You are a recommender system trained on the Amazon Beauty dataset. "
            "Given an input in the format:\n"
            "UserID::CurrentInteractions\n\n"
            "You must respond in the format:\n"
            "UserID::NextInteraction\n\n"
            "Output ONLY the response in this format. No additional text.\n\n"
        )
        params["task_format"] = "recommendation"
    elif params["dataset"] == "book":
        prompt_sentences = load_pretrained_book()
        params[
            "prompt_prefix"
        ] = (
            "You are a recommender system trained on the Amazon Book dataset. "
            "Given an input in the format:\n"
            "UserID::CurrentInteractions\n\n"
            "You must respond in the format:\n"
            "UserID::NextInteraction\n\n"
            "Output ONLY the response in this format. No additional text.\n\n"
        )
        params["task_format"] = "recommendation"
    else:
        raise NotImplementedError
    return prompt_sentences

if __name__ == "__main__":
    params = {}
    params["dataset"] = "ml1m"
    load_pretrained_dataset(params)
    print(params)
