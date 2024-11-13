# Wrapper function for creating prompts.
def make_prompt(prefix, continuation, eval_type="direct", task="sentence_judge",
                options=None, query=None, context=None, fullInstruction="True"):
    full_sentence = prefix + " " + continuation
    
    # For "direct" evaluation, just use the full sentence.
    if eval_type == "direct":
        return full_sentence
                
    if task == "sentence_judge":
        prompt = _make_prompt_yesno(sentence=prefix, continuation=continuation, eval_type=eval_type, query=query, context=context)

    elif task == "sentence_judge_generation_likert":
        prompt = _format_likert_prompt(options, query=query, context=context, fullInstruction=fullInstruction)
                
    elif task == "sentence_comparison":
        if options is None:
            raise ValueError("`options` cannot be None for metalinguistic prompts")
        prompt = _make_prompt_forcedchoice(continuation=continuation, eval_type=eval_type, options=options, query=query,
                                           context=context)

    elif task == "sentence_comparison_generation_1vs2":
        prompt = _format_choice_prompt(options, query=query, context=context, fullInstruction=fullInstruction)

    elif task == "word_comparison":
        if options is None:
            raise ValueError("`options` cannot be None for metalinguistic prompts")
        prompt = _make_prompt_word_comparison(full_sentence=full_sentence, prefix=prefix, continuation=continuation, options=options, eval_type=eval_type)

    else:
        raise ValueError(f"Unknown task specification! '{task}'")
    
    return prompt


def _make_prompt_word_comparison(full_sentence, prefix, continuation, options, eval_type):
    assert len(options) == 2
    option_str = f"{options[0]}, or {options[1]}"
    if eval_type == "metaQuestionSimple":
        prompt = f"What word is most likely to come next in the following sentence ({option_str}?)?\n\n{full_sentence}"
    elif eval_type == "metaInstruct":
        prompt = f"You are a helpful writing assistant. Tell me what word is more likely to come next in the following sentence ({option_str}?):\n\n{full_sentence}"
    elif eval_type == "metaQuestionComplex":
        prompt = f"Here is the beginning of an English sentence: {prefix}... What word is more likely to come next: {option_str}?\n\nAnswer: {continuation}"
    else:
        raise NotImplementedError
    return prompt


def _make_prompt_yesno(sentence, continuation, eval_type, query, context):

    response_info = "Respond with either Yes or No as your answer."

    if query == "makes sense":
        MQC_string = "Does this sentence make sense considering the context?"
    elif query == "is plausible":
        MQC_string = "Is this a plausible sentence considering the context?"
    elif query == "sounds good":
        MQC_string = "Does this sentence sound good in the given context?"

    if context:
        if eval_type == "metaQuestionComplex":
            prompt = (f'Here is a context: "{context}", and here is a sentence: "{sentence}".'
                      f'{MQC_string} {response_info}\n\nAnswer: {continuation}')
        else:
            raise NotImplementedError

    else:  # if not context
        if query == "makes sense":
            MQS_string = "Does the following sentence make sense?"
            MQC_string = "Does this sentence make sense?"
            MI_string = "Tell me if the following sentence makes sense."
            prompt_string_plaus = "makes perfect sense"
            prompt_string_implaus = "makes no sense"
        elif query == "is plausible":
            MQS_string = "Is the following sentence plausible?"
            MQC_string = "Is this sentence plausible?"
            MI_string = "Tell me if the following sentence is plausible."
            prompt_string_plaus = "is completely plausible"
            prompt_string_implaus = "is completely implausible"
        elif query == "sounds good":
            MQS_string = "Does the following sentence sound good?"
            MQC_string = "Does this sentence sound good?"
            MI_string = "Tell me if the following sentence sounds good."
            prompt_string_plaus = "sounds perfect"
            prompt_string_implaus = "sounds horrible"
        else:
            raise NotImplementedError
    
        if eval_type == "metaQuestionSimple":
            prompt = f"{MQS_string} {sentence} {response_info}\n\n{continuation}"
        elif eval_type == "metaQuestionComplex":
            prompt = f"Here is a sentence:\n\n{sentence} {MQC_string} {response_info}\n\nAnswer: {continuation}"
        elif eval_type == "metaInstruct":
            prompt = f"You are evaluating the plausibility of sentences. " \
                     f"A sentence {prompt_string_plaus} if the situation it describes commonly occurs in the real world. " \
                     f"A sentence {prompt_string_implaus} if the situation it describes never occurs in the real world. " \
                     f"{MI_string}\n\n{sentence} {response_info}\n\n{continuation}"
        else:
            raise NotImplementedError
            
    return prompt


def _make_prompt_forcedchoice(continuation, eval_type, options, query, context=None):

    if context:
        raise NotImplementedError

    else:
        sentence1, sentence2 = options
        option_str = f"1) {sentence1} 2) {sentence2}"
        response_info = "Respond with either 1 or 2 as your answer."
    
        if query == "makes sense":
            MQS_string = "Which sentence makes more sense?"
            MQC_string = MQS_string
            MI_string = "Tell me which sentence makes more sense."
            prompt_string_plaus = "makes perfect sense"
            prompt_string_implaus = "makes no sense"
        elif query == "is plausible":
            MQS_string = "Which sentence is more plausible?"
            MQC_string = MQS_string
            MI_string = "Tell me which sentence is more plausible."
            prompt_string_plaus = "is completely plausible"
            prompt_string_implaus = "is completely implausible"
        elif query == "sounds good":
            MQS_string = "Which sentence sounds better?"
            MQC_string = MQS_string
            MI_string = "Tell me which sentence sounds better."
            prompt_string_plaus = "sounds perfect"
            prompt_string_implaus = "sounds horrible"
    
        if eval_type == "metaQuestionSimple":
            prompt = f"{MQS_string} {option_str} {response_info}\n\n{continuation}"
        elif eval_type == "metaQuestionComplex":
            prompt = f"Here are two English sentences:\n\n{option_str} {MQC_string} {response_info}\n\nAnswer: {continuation}"
        elif eval_type == "metaInstruct":
            prompt = f"You evaluating the plausibility of sentences. " \
                     f"A sentence {prompt_string_plaus} if the situation it describes commonly occurs in the real world. " \
                     f"A sentence {prompt_string_implaus} if the situation it describes never occurs in the real world. " \
                     f"{MI_string}\n\n{option_str} {response_info}\n\n{continuation}"
        else:
            raise NotImplementedError
    
    return prompt


#### GENERATION TASKS ####
def _format_choice_prompt(options, query, context=None, fullInstruction="True") -> str:
    """Format a prompt for a sentence comparison task, using generation and then a regex parser."""

    if not context:
        s1, s2 = options
    else:
        c1, c2 = context
        s1 = options[0]

    if query == "makes sense":
        instr_text = "pick the sentence that makes more sense"
        task_text = "makes more sense"
        likert_low = "makes no sense"
        likert_high = "makes perfect sense"
        if context:
            question = "Which context makes more sense given the context?"
        else:
            question = "Which sentence makes more sense?"
    elif query == "is plausible":
        instr_text = "pick the sentence that is more plausible"
        task_text = "is more plausible"
        likert_low = "is completely implausible"
        likert_high = "is completely plausible"
        if context:
            question = "Which context is more plausible given the context?"
        else:
            question = "Which sentence is more plausible?"
    elif query == "sounds good":
        task_text = "sounds better"
        if context:
            instr_text = "pick the context in which the sentence sounds better"
            question = "In which context does the sentence sound better?"
        else:
            instr_text = "pick the sentence that sounds better"
            question = "Which sentence sounds better?"
    else:
        raise NotImplementedError

    if not context:
        if query != "sounds good":
            if fullInstruction == "False":
                instructions = (
                  f'You will be given a two sentences. Your task is to read the two sentences and {instr_text}.'
            )
            else:
                instructions = (
                  f'You will be given two sentences. Your task is to read the two sentences and {instr_text}. '
                + f"A sentence {likert_high} if the situation it describes commonly occurs in the real world. "
                + f"A sentence {likert_low} if the situation it describes never occurs in the real world.\n"
            )
        else:
            instructions = (
                  f'You will be given a two sentences. Your task is to read the two sentences and {instr_text}.'
            )
        item = f'Here are the two sentences:\n1) "{s1}"\n2) "{s2}"'
        question = question + "\n"
        response = (f'Respond with either 1 or 2 as your answer.\nAnswer: ')

    else:
        if query != "sounds good":
            if fullInstruction == "False":
                instructions = (
                  f"You will be given two contexts and a sentence. Your task is to read the two contexts and the sentence and "
                + f"{task_text} considering the sentence that follows. "
            )
            else:
                instructions = (
                    f"You will be given two contexts and a sentence. Your task is to read the two contexts and the sentence and "
                    + f"{task_text} considering the sentence that follows. "
                    + f"A sentence {likert_high} if the situation it describes commonly occurs in the real world. "
                    + f"A sentence {likert_low} if the situation it describes never occurs in the real world. "
                )
        else:
            instructions = (
                f"You will be given two contexts and a sentence. Your task is to read the two contexts and the sentence and "
                + f"{task_text} considering the sentence that follows. "
            )
        item = f'Here are the two contexts:\n1) "{c1}"\n2) "{c2}"\nHere is the sentence:\n"{s1}"'
        question = question + "\n"
        response = (f'Respond with either 1 or 2 as your answer.\nResponse: ')

    return "\n".join([instructions, item, question, response])


def _format_likert_prompt(s: str, query, context=None, fullInstruction="True") -> str:
    """Format a prompt for a sentence judgment task, using a Likert scale."""

    if query == "makes sense":
        instr_text = "rate how much sense it makes"
        likert_low = "makes no sense"
        likert_high = "makes perfect sense"
        instr_text2 = "how much the sentence makes sense"
        if context:
            question = "How much sense does this sentence make given the context?"
        else:
            question = "How much sense does this sentence make?"

    elif query == "is plausible":
        instr_text = "rate how plausible it is"
        likert_low = "is completely implausible"
        likert_high = "is completely plausible"
        instr_text2 = "how plausible the sentence is"
        if context:
            question = "How plausible is the sentence given the context?"
        else:
            question = "How plausible is this sentence?"

    elif query == "sounds good":
        instr_text = "rate how good it sounds"
        instr_text2 = "how good the sentence sounds"
        likert_low = "sounds terrible"
        likert_high = "sounds perfect"
        if context:
            question = "How good does the sentence sound given the context?"
        else:
            question = "How good does this sentence sound?"
    else:
        raise NotImplementedError

    if not context:
        if query != "sounds good":
            if not fullInstruction:
                instructions = (
                      f'You will be given a sentence. Your task is to read the sentence and {instr_text}. '
                )
            else:
                instructions = (
                      f'You will be given a sentence. Your task is to read the sentence and {instr_text}. '
                    + f"A sentence {likert_high} if the situation it describes commonly occurs in the real world. "
                    + f"A sentence {likert_low} if the situation it describes never occurs in the real world.\n"
                )
        else:
            instructions = (
                  f'You will be given a sentence. Your task is to read the sentence and {instr_text}. '
            )
        item = f'Here is the sentence: "{s}"\n'
        question = question + "\n"
        response = (f'Respond with a number on a scale from 1 to 7 as your answer, with 1 meaning "{likert_low}", '
                    + f'and 7 meaning "{likert_high}".\nAnswer: ')

    else:  # if context
        if query != "sounds good":
            if not fullInstruction:
                instructions = (
                    f"You will be given a context and a sentence. Your task is to read the context and the sentence and "
                    + f"rate {instr_text2} considering the context. "
                )
            else:
                instructions = (
                    f"You will be given a context and a sentence. Your task is to read the context and the sentence and "
                    + f"rate {instr_text2} considering the context. "
                    + f"A sentence {likert_high} if the situation it describes commonly occurs in the real world. "
                    + f"A sentence {likert_low} if the situation it describes never occurs in the real world. "
                )
        else:
            instructions = (
                f"You will be given a context and a sentence. Your task is to read the context and the sentence and "
                + f"rate {instr_text2} considering the context. "
            )
        item = f'Here is the context: "{context}"\nHere is the sentence: "{s}"\n'
        question = question + "\n"
        response = (f'Respond with a number on a scale from 1 to 7 as your answer, with 1 meaning "{likert_low}", '
                    + f'and 7 meaning "{likert_high}".\nResponse: ')

    return "\n".join([instructions, item, question, response])


def get_choice_regex() -> str:
    return r"([1-2])"


def get_likert_regex() -> str:
    return r"([1-7])"
