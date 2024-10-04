import os
import logging
import json
import torch
from src.modeling_mome import load_mome_model_for_inference, load_base_model_and_tokenizer
from safetensors.torch import save_file as safe_save_file

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def infer_device():
    return "cuda" if torch.cuda.is_available() and torch.cuda.device_count() > 0 else "cpu"

def load_jsonl_data(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(json.loads(line))
    return data

def evaluate_model(model, tokenizer, device, prompt):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    sample_output = model.generate(
        input_ids=input_ids,
        do_sample=False,
        max_new_tokens=100,
        return_dict_in_generate=True,
    )
    
    truncated_prompt = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    generated = tokenizer.decode(sample_output.sequences[0], skip_special_tokens=True)
    
    logger.debug(f"Prompt: {truncated_prompt}")
    logger.debug(f"Generated: {generated}")
    
    return generated[len(truncated_prompt):]

def test_mome_model(jsonl_file_path):
    base_model_name = "hf-internal-testing/tiny-random-MistralForCausalLM"
    mome_model_path = "model/checkpoint"
    device = infer_device()
    
    # Load models
    base_model = load_base_model_and_tokenizer(base_model_name, device)
    mome_model = load_mome_model_for_inference(base_model['model'], mome_model_path)
    
    print(mome_model)

    state_dict = mome_model.state_dict()
    safe_save_file(
        state_dict,
        os.path.join("full_model_file", "model.safetensors"),
        metadata={"format": "pt"},
    )

    # Load data from JSONL file
    data = load_jsonl_data(jsonl_file_path)

    # Process each entry in the JSONL file
    for entry in data:
        prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|> Consider the following company: {entry['ticker']} and quarter: {entry['quarter']}. {entry['question']} <|eot_id|><|start_header_id|>assistant<|end_header_id|>"

        result = evaluate_model(mome_model, base_model['tokenizer'], device, prompt)
        print(f"Company: {entry['ticker']}, Quarter: {entry['quarter']}")
        print(f"Question: {entry['question']}")
        print(f"Model response: {result}\n")

if __name__ == "__main__":
    jsonl_file_path = "data/banks_qa.jsonl" 
    test_mome_model(jsonl_file_path) 