import numpy as np
import pandas as pd
from tqdm import tqdm
from utils.runutils import *
import os

if __name__ == "__main__":
    TASK = "logprobs"

    # Parse command-line arguments.
    args = parse_args()
    args.task = TASK
    args.query = ""

    # Set random seed.
    np.random.seed(args.seed)

    # Set up model and other model-related variables.
    model = initialize_model(args)

    if not args.dataset == "SocialN400":
        dataset_type = "single_sentences"
    else:
        dataset_type = "context_plus_sentence"
    data_file = os.path.abspath(f"./datasets/{dataset_type}/{args.dataset}/corpus.csv")
    df = pd.read_csv(data_file)

    # Meta information.
    meta_data = {
        "model": args.model,
        "seed": args.seed,
        "eval_type": args.eval_type,
        "data_file": data_file,
        "timestamp": timestamp()
    }

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MAIN LOOP
    # Initialize results and get model outputs on each item.
    results = []

    for _, row in tqdm(list(df.iterrows()), total=len(df.index)):

        if args.dataset != "SocialN400":
            good_sentence = row.good_sentence
            bad_sentence = row.bad_sentence

            # Get standard full-sentence probabilities.
            logprob_of_good_sentence, num_tokens_good_sentence = model.get_full_sentence_logprob(
                sentence=good_sentence
            )
            logprob_of_bad_sentence, num_tokens_bad_sentence = model.get_full_sentence_logprob(
                sentence=bad_sentence
            )

            # Store results in dictionary.
            res = {
                "item_id": row.item_id,
                "good_sentence": good_sentence,
                "bad_sentence": bad_sentence,
                "logprob_of_good_sentence": logprob_of_good_sentence,
                "logprob_of_bad_sentence": logprob_of_bad_sentence,
                "num_tokens_good_sentence": num_tokens_good_sentence,
                "num_tokens_bad_sentence": num_tokens_bad_sentence
            }

        else:
            
            # Get standard full-sentence probabilities.
            logprob_of_sentence, num_tokens = model.get_full_sentence_logprob(
                sentence=row.sentence)
            # Get standard full-sentence probabilities within the context row.context.
            logprob_of_sentence_in_context, _ = model.get_full_sentence_logprob(
                sentence=row.sentence, context=row.context)

            # Store results in dictionary.
            res = {
                "item_id": row.item_id,
                "condition": row.condition,
                "context": row.context,
                "sentence": row.sentence,
                "logprob_of_sentence": logprob_of_sentence,
                "logprob_of_sentence_in_context": logprob_of_sentence_in_context,
                "num_tokens_sentence": num_tokens
            }

        results.append(res)


    # Combine meta information with model results into one dict.
    output = {
        "meta": meta_data,
        "results": results
    }

    # Save outputs to specified JSON file.
    output_dir, out_file = make_and_get_output_directory(args)
    dict2json(output, f"{output_dir}/{out_file}")
