from transformers import AutoModelForCausalLM, AutoTokenizer

# 选择一个预训练模型（例如，可以用 GPT-2）
model_dir = "./outputs"

# 加载模型和分词器
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir)

# 定义输入的prompt
prompt = "<|system|>You're Sam Williams, talking with me.<|end|><|user|>What is AOS Web?<|end|><|assistant|>"

# 将prompt转换为token
input_ids = tokenizer.encode(prompt, return_tensors='pt')

# 使用模型生成文本
output = model.generate(
    input_ids, 
    max_length=50,  # 最大生成长度
    num_return_sequences=1,  # 生成的序列数量
    no_repeat_ngram_size=2,  # 禁止重复生成的n-gram长度
    early_stopping=True
)

# 将生成的token转换为文本
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print(generated_text)
