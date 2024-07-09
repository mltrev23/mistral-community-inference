from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "mistral-community/Mixtral-8x22B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_id, use_flash_attention_2=True)

model = AutoModelForCausalLM.from_pretrained(model_id)

unlabeled_texts = [] 
with open('bittensor.txt', 'r') as f:
    for sentence in f.read().splitlines():
        unlabeled_texts.append(sentence)

class SampleDataset(Dataset):
    def __init__(self, tokenizer, size=100, max_length=512):
        self.tokenizer = tokenizer
        self.size = size
        self.max_length = max_length
    
    def __len__(self):
        return self.size

    def __getitem__(self, index):
        # Example input text and a random label
        text = unlabeled_texts[index]
        print(f'----------------------------------{text}---------------------------------------')
        inputs = self.tokenizer(text, max_length=self.max_length, padding='max_length', truncation=True, return_tensors="pt").to('cuda')
        # change the dtype of label to float
        _input = torch.tensor(inputs.input_ids, dtype=torch.long).to('cuda').squeeze(0)
        _attention_mask = torch.tensor(inputs.attention_mask, dtype=torch.long).to('cuda').squeeze(0)
        _label = torch.tensor(inputs.input_ids, dtype=torch.long).to('cuda').squeeze(0)
        return _input, _attention_mask, _label


inputs = tokenizer(unlabeled_texts, return_tensors="pt").to(0)

outputs = model.generate(**inputs, max_new_tokens=20)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
