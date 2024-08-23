import logging
import torch
from typing import Dict, List
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)

def eval_model(model, dataset, tokenizer: PreTrainedTokenizer, max_new_tokens: int):
    model.eval()
    results = []
    
    for example in dataset:
        result = eval_example(model, example, tokenizer, max_new_tokens)
        results.append(result)
    
    return results

def eval_example(model, example, tokenizer: PreTrainedTokenizer, max_new_tokens: int):
    logger.debug(f"Evaluating example: {example}")
    prompt = example["prompt"]
    input_ids = example["input_ids"]
    expected_output = example["expected_output"]
    
    generated_output = generate(model, input_ids, tokenizer, max_new_tokens)
    
    return {
        "prompt": prompt,
        "expected_output": expected_output,
        "generated_output": generated_output,
    }

def generate(model, input_ids, tokenizer: PreTrainedTokenizer, max_new_tokens: int):
    with torch.no_grad():
        sample_output = model.generate(
            input_ids=input_ids,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True,
        )
    
    logger.debug(
        f"Generated {len(sample_output.sequences[0]) - input_ids.shape[1]} tokens: {sample_output.sequences}"
    )
    
    truncated_prompt = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    generated = tokenizer.decode(sample_output.sequences[0], skip_special_tokens=True)
    
    logger.debug(f"Full generated text: {generated}")
    
    # Strip the prompt
    return generated[len(truncated_prompt):]

def save_results(results, path: str):
    import jsonlines
    with jsonlines.open(path, "w") as writer:
        for result in results:
            writer.write(result)