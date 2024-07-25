# Apus fine-tune tutorial

This project focuses on fine-tune model for AO(llama.cpp), this project is an example of using sam's tweet to fine-tune Phi3-Mini-4k-Instruct model. Aims to provide an ai assiant of sam williams.

## Prerequesites

1. GPU Server with CUDA support, at least 32GB of VRAM
2. Python 3.10 or higher
3. Git submodule version 2.34 or higher
4. Huggingface
   1. `https://huggingface.co/docs/huggingface_hub/en/guides/cli`
   2. Sign Up & Get AccessToken `https://huggingface.co/docs/hub/en/security-tokens`
   3. `huggingface login`
   4. Agree to the terms of model, for example, Phi3 `https://huggingface.co/microsoft/Phi-3-mini-4k-instruct`

## Installation Steps

1. Clone the repository and initialize submodules:
   `git clone https://github.com/apuslabs/aimodel-finetune-tools.git && cd aimodel-finetune-tools && git submodule update --init --recursive`
2. Create a virtual environment and activate it:
   1. `python3 -m venv .venv && source .venv/bin/activate`
   2. Install the required packages: `pip install -r requirements.txt`

## Fine-tune Model

We will use the Phi3-Mini-4k-Instruct model for fine-tuning. Here are the steps to fine-tune the model:
1. Load dataset
2. Split dataset into training and validation sets
3. Load the Phi3-Mini-4k-Instruct model from Huggingface
4. Define the model and tokenizer
5. Define the training arguments and start the fine-tuning process
6. Save the fine-tuned model to the specified directory
7. Evaluate the fine-tuned model

**Fine-tune**
`python fine-tune.py --dataset ./datasets/finetune-qa.json --model_name microsoft/Phi-3-mini-4k-instruct --output_dir ./models/v3`

**Evaluate**
`python evaluate.py --model_dir ./models/v3`

### Fine-tune configuration

To fine-tune based on other pre-trained models, follow these additional steps:
1. Use different Model Loader based on the model's architecture and configuration, for Phi3, it's `transformers.AutoModelForCausalLM.from_pretrained`
2. Adjust the tokenizer settings to match the new model's requirements, ensuring compatibility with the fine-tuning process.

To fine-tune on other datasets, you need to prepare the dataset in the required format and ensure it aligns with the model's input specifications. Adjust the dataset loading and preprocessing steps accordingly to fit the new dataset structure. Modify `tokenize_function` function in the `fine-tune.py`.

If your GPU server has limited VRAM, consider using mixed precision training or reducing the batch size to fit the model within the available memory.

## Integrate with AO

AO use `llama.cpp` as the backend for its AI functionalities, which use gguf format for model storage and inference. Ensure compatibility with the latest version of `llama.cpp` to leverage the full capabilities of the fine-tuned model.

**Convert to GGUF**
`python llama.cpp/convert_hf_to_gguf.py models/v3   --outfile gguf/phi3-4k-v0.1.gguf   --outtype q8_0`

### Test in llama.cpp

1. Upload the fine-tuned model to the Arweave by Ardrive
2. Get the Data Tx ID from Arweave transaction
3. Using `Llama.load('/data/<your model data tx id>')` to load the fine-tuned model and start the inference process