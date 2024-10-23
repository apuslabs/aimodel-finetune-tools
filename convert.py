import json

# Step 1: Read the JSON file
input_file = 'datasets/finetune-qa.json'
output_file = 'datasets/finetune-qa-modified.json'

with open(input_file, 'r') as f:
  data = json.load(f)

# Step 2: Process each item in the JSON data
for item in data:
  context = item.get('context', '')
  response = item.get('response', '')
  item['content'] = f"{context} {response}"
  # Remove the original fields
  if 'context' in item:
    del item['context']
  if 'response' in item:
    del item['response']

# Step 3: Save the modified data back to a new JSON file
with open(output_file, 'w') as f:
  json.dump(data, f, indent=4)

print(f"Modified data has been saved to {output_file}")