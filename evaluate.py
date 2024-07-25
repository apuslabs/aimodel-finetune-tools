from transformers import AutoModelForCausalLM, AutoTokenizer

# read args from command line
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str, default="./models/v3", help="Directory of the model")
args = parser.parse_args()
model_dir = args.model_dir

prompt = "<|system|>You're Sam Williams.<|end|>"
# read user questions from prompt
user_question = input("Please input your question: ")
prompt += f"<|user|>{user_question}<|end|><|assistant|>"

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir)

input_ids = tokenizer.encode(prompt, return_tensors='pt')

# Generate response
output = model.generate(
    input_ids, 
    max_length=200,  # 最大生成长度
    early_stopping=True,
)

# Decode and print the generated response
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print(generated_text)
