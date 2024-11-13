import numpy as np
import pandas as pd
from tqdm import tqdm
from utils.runutils import *
import os

if __name__ == "__main__":
    TASK = "word_comparison"

    # Parse command-line arguments.
    args = parse_args()
    args.task = TASK

    # Set random seed.
    np.random.seed(args.seed)

    # Set up model and other model-related variables.
    model = initialize_model(args)

    assert args.dataset == "DTFit", "This script is only for DTFit dataset."

    dataset_type = "single_sentences"
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

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MAIN LOOP
    # Initialize results and get model outputs on each item.
    results = []

    for _, row in tqdm(list(df.iterrows()), total=len(df.index)):

        prefix = " ".join(row.good_sentence.split()[:-1])
            
        good_continuation = row.good_sentence.split()[-1].strip(".")
        bad_continuation = row.bad_sentence.split()[-1].strip(".")

        if args.option_order == "goodFirst":
            options = [good_continuation, bad_continuation]
        else:
            options = [bad_continuation, good_continuation]

        print(f"Prefix : {prefix}")
        print(f"Good continuation : {good_continuation}")
        print(f"Bad continuation : {bad_continuation}")

        # Create prompt and get outputs (2x2).
        prompt, logprob_of_good_cont = \
            model.get_logprob_of_continuation(
                prefix=prefix,
                continuation=good_continuation,
                task=TASK,
                options=options,
                query=args.query
            )
        # Create prompt and get outputs (2x2).
        _, logprob_of_bad_cont = \
            model.get_logprob_of_continuation(
                prefix=prefix,
                continuation=bad_continuation,
                task=TASK,
                options=options,
                query=args.query
            )

        # Store results in dictionary.
        res = {
            "item_id": row.item_id,
            "prompt_good": prompt,
            "prefix": prefix,
            "good_continuation": good_continuation,
            "bad_continuation": bad_continuation,
            "logprob_of_good_cont": logprob_of_good_cont,
            "logprob_of_bad_cont": logprob_of_bad_cont,
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
    