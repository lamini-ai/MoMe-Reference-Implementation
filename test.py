import json
import torch
import os
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from raft_model.configuration_raft import LlamaRaftConfig
from raft_model.modeling_raft import LaminiRaftForCausalLM
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

try:
    from safetensors.torch import load_file
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False
    print("safetensors not available. Falling back to torch.load")

def load_data(file_path):
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]

def evaluate_model(model, tokenizer, data):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for item in tqdm(data):
            input_text = f"Question: {item['question']}\nAnswer:"
            inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

            outputs = model.generate(
                input_ids=inputs.input_ids,
                do_sample=False,
                max_new_tokens=50,
                return_dict_in_generate=True,
            )

            generated_ids = outputs.sequences[0]
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            generated_answer = generated_text.split("Answer:")[-1].strip()

            print(f"Question: {item['question']}")
            print(f"Model's answer: {generated_answer}")
            print(f"Correct answer: {item['answer']}")

    return 

def load_model_weights(model_path):
    if SAFETENSORS_AVAILABLE:
        try:
            return load_file(model_path)
        except Exception as e:
            print(f"Failed to load with safetensors: {e}")
    try:
        return torch.load(model_path, map_location='cpu')
    except Exception as e:
        print(f"Failed to load with torch.load: {e}")
        raise

def load_model_and_tokenizer(base_model_name, raft_model_path, index_path):

    token = os.getenv("HF_API_TOKEN")
    if not token:
        raise ValueError("HF_API_TOKEN environment variable is not set")

    logger.debug(f"Loading base model... {base_model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name, token=token)
    
    logger.debug(f"Loading RAFT config... {raft_model_path}")
    raft_config = LlamaRaftConfig.from_pretrained(raft_model_path)
    
    logger.debug("Initializing RAFT model...")
    raft_model = LaminiRaftForCausalLM(raft_config)
    
    logger.debug("Adding RAFT adaptor to model...")
    raft_model.initialize(base_model)
    
    logger.debug(f"Loading tokenizer... {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, token=token)
    
    logger.debug(f"RAFT model: {raft_model}")
    return raft_model, tokenizer

def main():
    # Load the data
    data = load_data("data/banks_qa.jsonl")

    # Set paths
    base_model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct" 
    raft_model_path = "model/checkpoint"
    index_path = "model/index"
    
    # Initialize the model and tokenizer
    model, tokenizer = load_model_and_tokenizer(base_model_name, raft_model_path, index_path)

    # Load RAFT adapter weights
    adapter_path = os.path.join(raft_model_path, "adapter_model.safetensors")
    if os.path.exists(adapter_path):
        try:
            state_dict = load_model_weights(adapter_path)
            model.load_state_dict(state_dict, strict=False)
            logger.info("Successfully loaded RAFT adapter weights")
        except Exception as e:
            logger.error(f"Failed to load RAFT adapter weights: {e}")
            logger.info("Using initialized RAFT adapter without pre-trained weights.")
    else:
        logger.info(f"RAFT adapter file {adapter_path} not found. Using initialized adapter without pre-trained weights.")

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Evaluate the model
    accuracy = evaluate_model(model, tokenizer, data)
    print(f"Model Accuracy: {accuracy:.2%}")

if __name__ == "__main__":
    main()