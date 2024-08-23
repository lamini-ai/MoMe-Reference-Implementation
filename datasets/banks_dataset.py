import datasets
import random
import logging

logger = logging.getLogger(__name__)

path = "/Users/powerml/Desktop/MoME Reference Implementation/data/banks_qa.jsonl"

def load_financial_dataset(args, model):
    dataset = datasets.load_dataset(
        "json",
        data_files=path,
        streaming=True,
        split="train",
    )

    dataset = dataset.take(10)

    transformed_dataset = transform_dataset(dataset, model)

    return transformed_dataset

def load_financial_dataset_for_eval(args, model):
    dataset = datasets.load_dataset(
        "json",
        data_files=path,
        streaming=True,
        split="train",
    )

    tokenized_dataset = dataset.map(
        get_eval_tokenize_function(model),
    )

    return tokenized_dataset

def get_eval_tokenize_function(model):
    tokenizer = model["tokenizer"]

    def tokenize_function(example):
        text = make_eval_prompt(
            example["ticker"],
            example["date"],
            example["quarter"],
            example["question"],
        )
        expected_output = make_eval_result(
            example["answer"],
            example["value"],
            example["units"],
        )

        tokenizer.pad_token = tokenizer.eos_token

        logger.debug(f"Tokenizing text: {text}")

        tokenized_inputs = tokenizer(text, return_tensors="pt")

        tokenized_inputs["prompt"] = text
        tokenized_inputs["expected_output"] = expected_output

        return tokenized_inputs

    return tokenize_function

def make_eval_prompt(ticker, date, quarter, question):
    prompt = f"<INSTR>{question}</INSTR>\n\nTicker: {ticker}\nDate: {date}\nQuarter: {quarter}"
    return prompt

def make_eval_result(answer, value, units):
    result = f"\n\nAnswer: {answer}\nValue: {value}\nUnits: {units}</s>"
    return result

def transform_dataset(dataset, model):
    block_size = 64

    new_dataset = dataset

    new_dataset = new_dataset.map(
        get_llama_tokenize_function(model),
        batched=True,
        remove_columns=[
            "ticker",
            "date",
            "quarter",
            "question",
            "answer",
            "has_value",
            "value",
            "units",
        ],
    )

    logger.debug(f"new_dataset: {new_dataset}")

    new_dataset = new_dataset.map(
        get_group_texts_function(block_size),
        batched=True,
    )

    new_dataset = new_dataset.with_format("torch")
    return new_dataset

def get_llama_tokenize_function(model):
    tokenizer = model["tokenizer"]

    def tokenize_function(examples):
        text = [
            make_prompt(ticker, date, quarter, question)
            for ticker, date, quarter, question in zip(
                examples["ticker"],
                examples["date"],
                examples["quarter"],
                examples["question"],
            )
        ]

        tokenized_inputs = tokenizer(text)

        tokenized_inputs["input_ids"] = [
            input_ids + [tokenizer.eos_token_id]
            for input_ids in tokenized_inputs["input_ids"]
        ]
        tokenized_inputs["attention_mask"] = [
            attention_mask + [1]
            for attention_mask in tokenized_inputs["attention_mask"]
        ]

        return tokenized_inputs

    return tokenize_function

def make_prompt(ticker, date, quarter, question):
    prompt = f"<INSTR>{question}</INSTR>\n\nTicker: {ticker}\nDate: {date}\nQuarter: {quarter}"
    return prompt

def get_group_texts_function(block_size):
    def group_texts(examples):
        lengths = [len(input_ids) for input_ids in examples["input_ids"]]
        random.shuffle(examples["input_ids"])

        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])

        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size

        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }

        result["labels"] = result["input_ids"].copy()

        return result

    return group_texts