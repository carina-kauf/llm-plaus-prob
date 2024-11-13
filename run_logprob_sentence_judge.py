import numpy as np
import pandas as pd
from tqdm import tqdm
from utils.runutils import *
import os

if __name__ == "__main__":
    TASK = "sentence_judge"

    # Parse command-line arguments.
    args = parse_args()
    args.task = TASK

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

    yes_continuation = "Yes"
    no_continuation = "No"

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MAIN LOOP
    # Initialize results and get model outputs on each item.
    results = []

    for _, row in tqdm(list(df.iterrows()), total=len(df.index)):

        if args.dataset != "SocialN400":
            good_sentence = row.good_sentence
            bad_sentence = row.bad_sentence

            good_prompt_yes, logprob_of_yes_good = \
                model.get_logprob_of_continuation(
                    prefix=good_sentence,
                    continuation=yes_continuation,
                    task=TASK,
                    query=args.query
                )

            _, logprob_of_no_good = \
                model.get_logprob_of_continuation(
                    prefix=good_sentence,
                    continuation=no_continuation,
                    task=TASK,
                    query=args.query
                )

            _, logprob_of_yes_bad = \
                model.get_logprob_of_continuation(
                    prefix=bad_sentence,
                    continuation=yes_continuation,
                    task=TASK,
                    query=args.query
                )

            _, logprob_of_no_bad = \
                model.get_logprob_of_continuation(
                    prefix=bad_sentence,
                    continuation=no_continuation,
                    task=TASK,
                    query=args.query
                )

            # Store results in dictionary.
            res = {
                "item_id": row.item_id,
                "good_prompt_yes": good_prompt_yes,
                "good_sentence": good_sentence,
                "bad_sentence": bad_sentence,
                "logprob_of_yes_good_sentence": logprob_of_yes_good,
                "logprob_of_yes_bad_sentence": logprob_of_yes_bad,
                "logprob_of_no_good_sentence": logprob_of_no_good,
                "logprob_of_no_bad_sentence": logprob_of_no_bad
            }
        else:
            prompt, logprob_of_yes_in_sent = \
                model.get_logprob_of_continuation(
                prefix=row.sentence,
                continuation=yes_continuation,
                task=TASK,
                query=args.query
                )

            _ , logprob_of_no_in_sent = \
                model.get_logprob_of_continuation(
                prefix=row.sentence,
                continuation=no_continuation,
                task=TASK,
                query=args.query
                )

            _ , logprob_of_yes_in_cont_and_sent = \
                model.get_logprob_of_continuation(
                prefix=f"{row.context} {row.sentence}",
                continuation=yes_continuation,
                task=TASK,
                query=args.query
                )

            _ , logprob_of_no_in_cont_and_sent = \
                model.get_logprob_of_continuation(
                prefix=f"{row.context} {row.sentence}",
                continuation=no_continuation,
                task=TASK,
                query=args.query
                )
            
            # Store results in dictionary.
            res = {
                "item_id": row.item_id,
                "condition": row.condition,
                "prompt_yes": prompt,
                "context": row.context,
                "sentence": row.sentence,
                "continuation": row.continuation,
                "logprob_of_yes_sentence": logprob_of_yes_in_sent,
                "logprob_of_no_sentence": logprob_of_no_in_sent,
                "logprob_of_yes_context_and_sentence": logprob_of_yes_in_cont_and_sent,
                "logprob_of_no_context_and_sentence": logprob_of_no_in_cont_and_sent
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