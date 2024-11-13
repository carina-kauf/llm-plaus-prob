import torch
import typing
from .make_prompts import make_prompt, _format_choice_prompt, _format_likert_prompt, get_choice_regex, get_likert_regex
import numpy as np

from outlines.text import generate
import outlines.models as models

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    FalconForCausalLM,
)

# Helper function for loading Huggingface models and tokenizers.
def load_hf_model(model_name, device="cpu", cache_dir=".huggingface_cache", **kwargs):
    if "falcon" in model_name:
        model = FalconForCausalLM.from_pretrained(model_name).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    elif "mosaicml" in model_name:
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"Successfully loaded model ({model_name})")
    print(f"Successfully loaded tokenizer ({model_name})")

    return model, tokenizer


# Base class for large language model.
class LLM(object):
    def __init__(self, args, device="cpu"):
        self.eval_type = args.eval_type
        self.model = args.model
        self.seed = args.seed
        self.device = device


class Causal_LLM(LLM):
    def __init__(self, args, device="cpu"):
        super().__init__(args, device=device)
        self._model, self._tokenizer = load_hf_model(model_name=self.model, device=self.device)

    def get_full_sentence_logprob(self, sentence, context=None):
        """
        Calculate the log probability of a given sentence under a language model.
        If a context is provided, calculate the log probability of the sentence in the context of the context, but
        exclude the context from the log probability calculation.

        Parameters:
        - sentence (str): The input sentence for log probability calculation.
        - context (str, optional): A context sentence that precedes the main sentence (default: None).
        - **kwargs: Additional keyword arguments to customize the behavior.

        Returns:
        tuple: A tuple containing the log probability of the input sentence under the language model and the number of tokens.
        """
        # prep sentence
        if not context:  # not needed for sentences in a context.
            if self._tokenizer.bos_token and self._tokenizer.eos_token:
                print(f"Prepending the BOS token: {self._tokenizer.bos_token} and appending the EOS token: {self._tokenizer.eos_token}")
                sentence = self._tokenizer.bos_token + sentence + self._tokenizer.eos_token
        # Tokenize the sentence
        input_tokens = self._tokenizer(sentence, return_tensors="pt").input_ids.to(self.device)

        # Determine the number of tokens
        num_tokens = input_tokens.size(1)

        if context:
            context_tokens = self._tokenizer(context, return_tensors="pt").input_ids.to(self.device)
            tensor_input = torch.cat([context_tokens, input_tokens], dim=1)
        else:
            tensor_input = input_tokens

        logits = self._model(tensor_input).logits

        if context:
            # Prepare the input and label tensors for the loss calculation
            input_logits = logits[..., -input_tokens.shape[1]-1:-1, :].contiguous()
            input_labels = input_tokens[..., :].contiguous()
        else:  # single sentence scoring
            input_logits = logits[..., :-1, :].contiguous()  # exclude the last logit
            input_labels = input_tokens[..., 1:].contiguous()  # exclude the first label token

        # Flatten the tokens and calculate the loss
        loss_fct = torch.nn.CrossEntropyLoss()
        input_loss = -loss_fct(input_logits.view(-1, input_logits.size(-1)), input_labels.view(-1)).item()
        # Note to self: for single sentence inputs, this is the same as
        # input_loss = -self._model(tensor_input, labels=tensor_input).loss

        total_logprob = input_loss * num_tokens

        return total_logprob, num_tokens

    def get_token_logprobs(self, prompt):
        """
        Retrieve the per-token log probabilities for observed tokens in the context of a given prompt.
        The log probabilities are calculated by excluding the last token to consider the previous word
        in the context of the current word, because this is a causal language model
        """

        # Tokenize the prompt
        input_ids = self._tokenizer.encode(prompt, return_tensors="pt")
        # get word_ids to be able to sum over subtoken logprobs for desired continuation
        try:
            word_ids = self._tokenizer(prompt, return_tensors="pt").word_ids(batch_index=0)
        except:
            print("Word_ids not supported")
            word_ids = None

        # Get the model's output
        with torch.no_grad():
            outputs = self._model(input_ids)

        # Calculate log probabilities
        logsoftmax = torch.log_softmax(outputs.logits, dim=-1)
        indices = input_ids[0, 1:].unsqueeze(-1)
        observed_log_probs = torch.gather(logsoftmax[0, :-1, :], dim=-1, index=indices)
        observed_log_probs = torch.cat([torch.tensor([0.0]), observed_log_probs.flatten()])

        return {"tokens": input_ids[0], "token_logprobs": observed_log_probs, "word_ids": word_ids}

    def get_logprob_of_continuation(self, prefix, continuation, task, options=None, query="is plausible"):
        """
        Calculate the log probability of a given continuation in the context of a prefix.

        Parameters:
            prefix (str): The initial part of the text.
            continuation (str): The continuation to evaluate for log probability.
            **kwargs: Additional keyword arguments that can be passed to the underlying methods.

        Returns:
            tuple: A tuple containing the constructed prompt and the log probability of the given continuation.

        The function constructs a prompt by combining the provided prefix and continuation,
        then utilizes a local Hugging Face model to obtain the log probabilities of the tokens in the prompt.

        The function identifies the indices of the tokens corresponding to the relevant continuation
        (final word or sequence of words) and calculates the log probability of the continuation by
        summing the log probabilities of the relevant sub-word tokens.
        """
        # Construct prompt and get logprobs.
        prompt = make_prompt(
            prefix,
            continuation,
            eval_type=self.eval_type,
            task=task,
            options=options,
            query=query
        )

        print(f"Prompt: {prompt}")
        logprobs = self.get_token_logprobs(prompt)
        tokens, token_logprobs, word_ids = logprobs["tokens"], logprobs["token_logprobs"], logprobs["word_ids"]

        # Identify indices from `tokens` that correspond to the relevant continuation (final word).
        # This could be split into multiple tokens.
        # Obtain logprob of gold (ground-truth) word by summing logprobs of all sub-word tokens
        try:
            max_word_id = max(word_ids)
            inds = [i for i, x in enumerate(tokens) if word_ids[i] == max_word_id]

        except:
            n_tokens = len(tokens)
            tokens_decoded = [self._tokenizer.decode(t) for t in tokens]
            full_continuation_str = " " + continuation
            if task == "sentence_comparison":
                # The number tokens sometimes have preceding space, sometimes not.
                end_strs = [full_continuation_str, continuation]
            else:
                end_strs = full_continuation_str
            inds = []
            cur_word = ""
            for tok_idx in range(n_tokens-1, -1, -1):
                # Go backwards through the list of tokens.
                cur_tok = tokens_decoded[tok_idx]
                cur_word = cur_tok + cur_word
                if token_logprobs[tok_idx] is None:
                    break
                else:
                    inds = [tok_idx] + inds
                    if cur_word in end_strs:
                        break

        logprob_of_continuation = sum([token_logprobs[i] for i in inds])
        if len(inds) > 1:
            print(f"Summing logprobs of multiple tokens: {[self._tokenizer.decode(tokens[i]) for i in inds]}")

        return prompt, logprob_of_continuation.item()

    def get_logprob_likert(self, sentence, query, context=None):
        """Get logprob of likert continuation (not used currently, because we can just do constrained generation)
        """
        prompt = _format_likert_prompt(sentence, query, context)

        numbers_to_check = ["1", "2", "3", "4", "5", "6", "7"]

        max_logprob = float('-inf')
        max_number = None

        for number in numbers_to_check:
            prompt_plus_continuation = f"{prompt} {number}"
            logprobs = self.get_token_logprobs(prompt_plus_continuation)
            tokens, token_logprobs, _ = logprobs["tokens"], logprobs["token_logprobs"], logprobs["word_ids"]

            # Get the log probability of the number token, which is the last token in the prompt.
            logprob = token_logprobs[-1]

            # Update max_logprob and max_number if needed
            if logprob > max_logprob:
                max_logprob = logprob
                max_number = number

        # Print the number with the maximum log probability
        print(f"Likert score with max log probability: {max_number} (logprob: {max_logprob})")

        return prompt, max_number, max_logprob.item()


class Generation_LLM(LLM):
    def __init__(self, args, device="cpu"):
        super().__init__(args, device=device)
        self._model = models.transformers(self.model, device=self.device)
        #self._stop_token = "\n"

        # initiate generator
        if args.gen_type == "constrained":
            self._max_tokens = 1
            if "likert" in args.task:
                self._pattern = get_likert_regex()
            else:
                self._pattern = get_choice_regex()
            self._generator = self._get_generator(pattern=self._pattern)
        elif args.gen_type == "free":
            self._max_tokens = 20
            self._generator = self._get_generator()
        else:
            raise ValueError(f"Generation type `{args.gen_type}` not supported")

    def _get_generator(self, pattern=""):
        # tokenizer = TransformerTokenizer(self.model)
        # model = Transformer(self._model, tokenizer)
        if pattern:
            generator = generate.regex(self._model, pattern)
        else:
            generator = generate.continuation(
                self._model, max_tokens=self._max_tokens, #stop=self._stop_token
            )
        return generator

    def complete_choice(self, options: typing.List[str], query: str, context: str = "", fullInstruction: str = "True") -> (str, str):
        prompt = _format_choice_prompt(options, query, context, fullInstruction)
        completion = self._generator(prompt)
        return prompt, completion

    def complete_likert(self, sentence: str, query: str, context: str = "", fullInstruction: str = "True") -> (str, str):
        prompt = _format_likert_prompt(sentence, query, context, fullInstruction)
        completion = self._generator(prompt)
        return prompt, completion
