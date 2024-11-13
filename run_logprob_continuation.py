import numpy as np
import pandas as pd
from tqdm import tqdm
from utils.runutils import *
import os

if __name__ == "__main__":
    TASK = "sentence_comparison"


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
        "option_order": args.option_order,
        "data_file": data_file,
        "timestamp": timestamp()
    }

    good_continuation = "1" if args.option_order == "goodFirst" else "2"
    bad_continuation = "2" if args.option_order == "goodFirst" else "1"

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MAIN LOOP
    # Initialize results and get model outputs on each item.
    results = []

    for _, row in tqdm(list(df.iterrows()), total=len(df.index)):
            
        # Create "continuations". We're essentially asking the models
        # a multiple choice question.

        if args.dataset != "SocialN400":
            good_sentence = row.good_sentence
            bad_sentence = row.bad_sentence

            # Present a particular order of the answer options.
            if args.option_order == "goodFirst":
                options = [good_sentence, bad_sentence]
            else:
                options = [bad_sentence, good_sentence]

            # Create prompt and get outputs (2x2).
            prompt, logprob_of_good_cont = \
                model.get_logprob_of_continuation(
                    prefix="",
                    continuation=good_continuation,
                    task=TASK,
                    options=options,
                    query=args.query
                )
            # Create prompt and get outputs (2x2).
            _, logprob_of_bad_cont = \
                model.get_logprob_of_continuation(
                    prefix="",
                    continuation=bad_continuation,
                    task=TASK,
                    options=options,
                    query=args.query
                )

            # Store results in dictionary.
            res = {
                "item_id": row.item_id,
                "prompt_good": prompt,
                "good_sentence": good_sentence,
                "bad_sentence": bad_sentence,
                "good_continuation": good_continuation,
                "bad_continuation": bad_continuation,
                "logprob_of_good_cont": logprob_of_good_cont,
                "logprob_of_bad_cont": logprob_of_bad_cont,
            }
        else:
            sentence = f"{row.prefix} {row.continuation}"
            raise NotImplementedError("Adequate prompts not yet implemented for SocialN400.")


        results.append(res)


    # Combine meta information with model results into one dict.
    output = {
        "meta": meta_data,
        "results": results
    }

    # Save outputs to specified JSON file.
    output_dir, out_file = make_and_get_output_directory(args)
    dict2json(output, f"{output_dir}/{out_file}")
    