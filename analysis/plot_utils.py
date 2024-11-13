# Shared variables and helper functions across all analysis notebooks.
import itertools
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.container import BarContainer

# =============================================================================
# GLOBAL VARIABLES
# =============================================================================

EVAL_TYPES = ["Direct", "MetaQuestionSimple", "MetaInstruct", "MetaQuestionComplex"]
META_EVAL_TYPES = [e for e in EVAL_TYPES if e.startswith("Meta")]
eval_type_pairs = list(itertools.combinations(EVAL_TYPES, 2))
direct_pairs = [pair for pair in eval_type_pairs if "Direct" in pair]

GLOBAL_MODEL_ORDER = ["human",
                      # Critical set
                      "Mistral-7B-v0.1",
                      "Mistral-7B-Instruct-v0.1",
                      #
                      "falcon-7b",
                      "falcon-7b-instruct",
                      #
                      "mpt-7b",
                      "mpt-7b-instruct",
                      # Baseline
                      "gpt2-xl"]

# prettify model names
PRETTYNAMES = {
    "human" : "Human",
    "Mistral-7B-v0.1" : "Mistral (base)",
    "Mistral-7B-Instruct-v0.1" : "Mistral (instr.)",
    "falcon-7b" : "Falcon (base)",
    "falcon-7b-instruct" : "Falcon (instr.)",
    'mpt-7b' : "MPT (base)",
    'mpt-7b-instruct' : "MPT (instr.)",
    "gpt2-xl" : "GPT2-xl"
}


# =============================================================================
# STYLING / AESTHETICS
# =============================================================================

# Define consistent color palette for all models.

paired = sns.color_palette(palette='Paired')

MODEL_PAL = {
    "human" : "dimgray",
    #
    "Mistral-7B-v0.1" : paired[1],
    "Mistral-7B-Instruct-v0.1" : paired[0],
    #
    "falcon-7b": paired[5],
    "falcon-7b-instruct": paired[4],
    #
    "mpt-7b" : paired[3],
    "mpt-7b-instruct" : paired[2],
    #
    "gpt2-xl" : "gainsboro"
}