import json
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
import numpy as np
import evaluate
import torch
import onnx
import onnxruntime

# Load your custom dataset from JSON file
def load_json_dataset(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data

data = load_json_dataset("./finetune.json")

# Convert the loaded data to Hugging Face Dataset format
dataset = Dataset.from_list(data)

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

# Tokenize the dataset
def tokenize_function(examples):
    # Ensure responses are strings
    examples["response"] = [" ".join(r) if isinstance(r, list) else r for r in examples["response"]]
    
    inputs = tokenizer(examples["context"], padding="max_length", truncation=True, max_length=512)
    outputs = tokenizer(examples["response"], padding="max_length", truncation=True, max_length=512)
    inputs["labels"] = outputs["input_ids"]
    return inputs

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Split the dataset into training and evaluation sets
train_test_split = tokenized_datasets.train_test_split(test_size=0.2)
train_dataset = train_test_split["train"]
eval_dataset = train_test_split["test"]

# Load the model
model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

# Define training arguments
training_args = TrainingArguments(output_dir="test_trainer")

# Define the metric for evaluation
metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)

# Assume the model and tokenizer loaded and paths set correctly in `export_onnx.py`
def export_onnx(model, tokenizer, export_path='model.onnx'):
    # Example input text for tracing
    text = "Hello, my dog is cute"

    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt")

    # Export the model to ONNX
    torch.onnx.export(
        model,                            # Model to export
        (inputs["input_ids"],),           # Example input
        export_path,                      # Path to save the ONNX file
        input_names=["input_ids"],        # Names of the input tensors
        output_names=["output"],          # Names of the output tensors
        dynamic_axes={"input_ids": {0: "batch_size"}, "output": {0: "batch_size"}},  # Dynamic axes
        opset_version=13                  # ONNX opset version
    )

    print(f"Model successfully exported to {export_path}")

    # Verify the ONNX model
    onnx_model = onnx.load(export_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model is valid.")

    # Run the model with ONNX Runtime
    ort_session = onnxruntime.InferenceSession(export_path)

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    # Get the ONNX model input
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(inputs["input_ids"])}

    # Run the model
    ort_outs = ort_session.run(None, ort_inputs)

    print("ONNX model output:", ort_outs)

# Train the model
if __name__ == "__main__":
    trainer.train()
    export_onnx(trainer.model, tokenizer)