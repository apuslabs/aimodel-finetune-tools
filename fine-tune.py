import json
from datasets import Dataset, load_metric
from transformers import AutoTokenizer, AutoModelForCausalLM, Seq2SeqTrainingArguments, Trainer
import torch
import numpy as np

# Read dataset filepath, model name, and output directory from command line
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="./datasets/finetune-qa.json", help="Path to the dataset")
parser.add_argument("--model_name", type=str, default="microsoft/Phi-3-mini-4k-instruct", help="Model name")
parser.add_argument("--output_dir", type=str, default="./models/v3", help="Output directory")
args = parser.parse_args()
dataset_path = args.dataset
model_name = args.model_name
output_dir = args.output_dir

torch.cuda.empty_cache()
device = "cuda"

# Load your custom dataset from JSON file
def load_json_dataset(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data


data = load_json_dataset(dataset_path)

# Convert the loaded data to Hugging Face Dataset format
dataset = Dataset.from_list(data)
# Get first ten data
# dataset = dataset.select(range(10))

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Tokenize the dataset
def tokenize_function(examples):
    # Ensure responses are strings
    examples["response"] = [" ".join(r) + "<|endoftext|>" if isinstance(r, list) else r for r in examples["response"]]
    # add <|system|>You're Sam Williams.<|end|><|user|> to the begin and <|end|><|assistant|> to the end
    examples["context"] = ["<|system|>You're Sam Williams.<|end|><|user|>" + c + "<|end|><|assistant|>" for c in examples["context"]]
    
    inputs = tokenizer(
        text=examples["context"],
        text_pair=examples["response"],
        add_special_tokens=True,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_attention_mask=True,
    )

    # Create labels by shifting the input_ids to the right
    inputs["labels"] = tokenizer(
        text=examples["response"],
        add_special_tokens=True,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_attention_mask=True,
    )["input_ids"]
    return inputs

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Split the dataset into training and evaluation sets
train_test_split = tokenized_datasets.train_test_split(test_size=0.2)
train_dataset = train_test_split["train"]
eval_dataset = train_test_split["test"]

# Load the model
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

# Define training arguments with gradient accumulation and mixed precision training
training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,          # output directory to save model checkpoint
    num_train_epochs=5,              # increase epochs for better results
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,              # you can try different learning rates
    weight_decay=0.01,               # increase weight_decay to regularize and avoid overfitting
    fp16=True,
    evaluation_strategy="steps",
    save_total_limit=5,
    predict_with_generate=True,
)

# Define the metric for evaluation
# em_metric = load_metric("exact_match")
# f1_metric = load_metric("f1")
# bleu_metric = load_metric("bleu")
rouge_metric = load_metric("rouge")

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    # Calculate batch size and desired sub-batch size
    batch_size = preds.shape[0]
    sub_batch_size = 3  
    decoded_preds, decoded_labels = [], []
    for i in range(0, batch_size, sub_batch_size):
        sub_preds = preds[i:min(i + sub_batch_size, batch_size)]
        sub_labels = labels[i:min(i + sub_batch_size, batch_size)]
        # Decode the sub-batch using your current decoding logic
        sub_decoded_preds = tokenizer.batch_decode(sub_preds, skip_special_tokens=True)
        sub_decoded_labels = tokenizer.batch_decode(sub_labels, skip_special_tokens=True)
        # Append the decoded sub-batches to the main lists
        decoded_preds.extend(sub_decoded_preds)
        decoded_labels.extend(sub_decoded_labels)
    rouge_output = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels)
    return rouge_output

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    # compute_metrics=compute_metrics,
)

# Train the model
if __name__ == "__main__":
    try:
        trainer.train()
        # Save the model and tokenizer after training is done
        trainer.save_model(output_dir)  # save model
        tokenizer.save_pretrained(output_dir)  # save tokenizer
    except RuntimeError as e:
        if 'out of memory' in str(e):
            print('Out of memory error caught: Cleaning up GPU memory.')
            torch.cuda.empty_cache()
        else:
            raise e