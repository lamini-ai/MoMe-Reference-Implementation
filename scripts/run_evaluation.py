import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transformers import AutoTokenizer
from datasets import load_dataset
import argparse
import torch
from raft_model.modeling_raft import RaftForCausalLM
from raft_model.configuration_raft import RaftConfig
from raft_model.lamini_index import LaminiIndex

def parse_args():
    parser = argparse.ArgumentParser(description="Run evaluation on RAFT model")
    parser.add_argument("--model_path", type="str", required=True, help="Path to the pretrained model")
    parser.add_argument("--tokenizer_path", type="str", help="Path to the tokenizer (if different from model)")
    parser.add_argument("--dataset_path", type="str", required=True, help="Path to the evaluation dataset")
    parser.add_argument("--dataset_split", type="str", default="train", help="Dataset split to use")
    parser.add_argument("--output_dir", type="str", default="./evaluation_results", help="Directory to save results")
    parser.add_argument("--device", type="str", default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run the model on (cuda or cpu)")
    return parser.parse_args()

def load_raft_model(model_path, device):
    config_path = os.path.join(model_path, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    
    config = RaftConfig.from_json_file(config_path)
    
    # Check if index files exist
    index_path = os.path.join(model_path, "index")
    if not os.path.exists(index_path):
        print(f"Index directory not found at {index_path}")
        print("Continuing without index...")
        config.index_path = None
    else:
        config.index_path = index_path

    model = RaftForCausalLM(config)
    
    # Load the model weights
    model_file = os.path.join(model_path, "pytorch_model.bin")
    if os.path.exists(model_file):
        state_dict = torch.load(model_file, map_location=device)
        model.load_state_dict(state_dict, strict=False)
    else:
        print(f"Warning: pytorch_model.bin not found in {model_path}")
    
    # Move model to the specified device
    model = model.to(device)
    
    return model

def main():
    args = parse_args()

    # Set the device
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Load your model and tokenizer
    model = load_raft_model(args.model_path, device)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path or args.model_path)

    # Attach tokenizer to model (required for evaluation)
    model.tokenizer = tokenizer

    # Load your evaluation dataset
    eval_dataset = load_dataset("json", data_files=args.dataset_path, split=args.dataset_split)

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, "eval_results.jsonl")

    # Run evaluation
    results = model.evaluate(
        eval_dataset, 
        max_new_tokens=model.config.max_position_embeddings, 
        save_path=save_path,
        device=device
    )

    print(f"Evaluation complete. Results saved to {save_path}")
    return results

if __name__ == "__main__":
    main()