import numpy as np
import pandas as pd
from tqdm import tqdm
from utils.runutils import *
import os
from utils.make_prompts import get_likert_regex
import re

if __name__ == "__main__":
    TASK = "sentence_judge_generation_likert"

    # Parse command-line arguments.
    args = parse_args()
    args.task = TASK
    args.eval_type = None

    # Sanity checks.
    assert args.gen_type in ["free", "constrained"], "gen_type must be either 'free' or 'constrained'"

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
        "gen_type": args.gen_type,
        "query": args.query,
        "fullInstruction" : args.fullInstruction,
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

            good_prompt, good_generation = \
                model.complete_likert(
                    sentence=good_sentence,
                    query=args.query,
                    fullInstruction=args.fullInstruction)
            bad_prompt, bad_generation = \
                model.complete_likert(
                    sentence=bad_sentence,
                    query=args.query,
                    fullInstruction=args.fullInstruction)

            # Store results in dictionary.
            res = {
                "item_id": row.item_id,
                "good_prompt": good_prompt,
                "bad_prompt": bad_prompt,
                "good_sentence": good_sentence,
                "bad_sentence": bad_sentence,
                "good_generation": good_generation,
                "bad_generation": bad_generation
            }

            if args.gen_type == "free":
                pattern = get_likert_regex()
                res["good_continuation_contains_pattern"] = bool(re.search(pattern, good_generation))
                res["bad_continuation_contains_pattern"] = bool(re.search(pattern, bad_generation))

        else:
            prompt, generation = \
                model.complete_likert(
                    sentence=row.sentence,
                    query=args.query,
                    context=row.context)
            
            # Store results in dictionary.
            res = {
                "item_id": row.item_id,
                "prompt": prompt,
                "context": row.context,
                "sentence": row.sentence,
                "generation": generation,
            }
            
            if args.gen_type == "free":
                pattern = get_likert_regex()
                res["continuation_contains_pattern"] = bool(re.search(pattern, generation))

        results.append(res)


    # Combine meta information with model results into one dict.
    output = {
        "meta": meta_data,
        "results": results
    }

    # Save outputs to specified JSON file.
    output_dir, out_file = make_and_get_output_directory(args)
    dict2json(output, f"{output_dir}/{out_file}")
