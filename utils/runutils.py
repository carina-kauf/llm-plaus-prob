import json
from time import gmtime, strftime
import argparse
import torch
import pathlib
import os

from . import models


# Extremely basic helper functions.
def timestamp():
    return strftime("%Y-%m-%d %H:%M:%S", gmtime())


def dict2json(d, out_file):
    with open(out_file, "w") as fp:
        json.dump(d, fp, indent=2)


def json2dict(in_file):
    with open(in_file, "r") as fp:
        d = json.load(fp)
    return d


# Helper function for parsing command-line arguments.
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate model using specified prompts")
    parser.add_argument("--model", "-M", type=str, default="tiiuae/falcon-rw-1b", help="Name of model",
                        choices=[
                            "gpt2-xl",
                            #
                            "tiiuae/falcon-7b",
                            "tiiuae/falcon-7b-instruct",
                            #
                            "mistralai/Mistral-7B-v0.1",
                            "mistralai/Mistral-7B-Instruct-v0.1",
                            #
                            "mosaicml/mpt-7b",
                            "mosaicml/mpt-7b-instruct"
                        ])
    parser.add_argument("--dataset", "-D", type=str, default="EventsRev", help="Name of dataset",
                        choices=[
                            "SocialN400",
                            "EventsRev",
                            "EventsAdapt",
                            "DTFit"
                        ])
    parser.add_argument("--task", "-T", type=str, default="",
                        choices=[
                            "sentence_judge",
                            "sentence_judge_generation_likert",
                            "sentence_comparison",
                            "sentence_comparison_generation_1vs2",
                            "word_comparison"
                        ],
                        help="Type of task (not sentence logprobs)")
    parser.add_argument("--eval_type", type=str, default="direct",
                        choices=[
                            "direct", 
                            "metaQuestionSimple", 
                            "metaInstruct", 
                            "metaQuestionComplex"
                        ], 
                        help="Type of evaluation (for prompt design)")
    parser.add_argument("--seed", "-S", type=int, default=0,
                        help="Random seed for reproducibility")
    parser.add_argument("--query", "-Q", type=str, default="is plausible",
                        choices=["makes sense", "is plausible", "sounds good"],
                        help="Query for meta evaluation")
    parser.add_argument("--gen_type", "-G", type=str, default="",
                        choices=["free", "constrained"],
                        help="Type of generation")
    parser.add_argument("--option_order", type=str, default="",
                        choices=["goodFirst", "badFirst"])
    parser.add_argument("--fullInstruction", type=str, default=None)
    args = parser.parse_args()

    if args.eval_type in ["metaQuestionSimple", "metaQuestionComplex", "metaInstruct"]:
        assert args.query, "query must be specified for meta evaluation, otherwise it's confusing"

    print(f"Arguments: {args}")

    return args


# Helper function for initializing models.
def initialize_model(args):
    # Set device to GPU if available, otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda":
        print("Set device to CUDA")
    else:
        print("Using CPU (CUDA unavailable); adjust your expectations")

    # Check if the specified model is "flan-t5"
    if "flan-t5" in args.model:
        model = models.T5_LLM(args, device=device)
    else:
        # Choose model based on the task
        model_type = models.Generation_LLM if "generation" in args.task else models.Causal_LLM
        try:
            model = model_type(args, device=device)
        except Exception as e:
            raise ValueError(f"Model not supported! (Your model: {args.model})") from e

    return model


def make_and_get_output_directory(args, TEST=False):
    # define good safe name for output file
    if "/" in args.model:
        args.model = args.model.split("/")[-1]

    # make results directory if it doesn't exist
    output_dir = f"./results/{args.task}"
    pathlib.Path(os.path.abspath(output_dir)).mkdir(parents=True, exist_ok=True)

    if args.eval_type == "direct":
        out_file = f"{args.dataset}_{args.model}_{args.eval_type}_q={'+'.join(args.query.split())}_s={args.seed}.json"
    else:
        if "generation" not in args.task:
            out_file = f"{args.dataset}_{args.model}_{args.task}_{args.eval_type}_q={'+'.join(args.query.split())}_s={args.seed}.json"
        else:
            out_file = f"{args.dataset}_{args.model}_{args.task}_q={'+'.join(args.query.split())}_s={args.seed}_{args.gen_type}.json"
    if args.option_order != "":
        out_file = f"{out_file.rstrip('.json')}_{args.option_order}.json"
    if args.fullInstruction:
        out_file = f"{out_file.rstrip('.json')}_{args.fullInstruction}.json"

    return output_dir, out_file

