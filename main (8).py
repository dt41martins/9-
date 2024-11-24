from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Lejupielādējam modeli un tokenizer
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Sākuma frāze
prompt = "Reiz kādā tālā zemē..."

# Tokenizējam sākuma frāzi
inputs = tokenizer.encode(prompt, return_tensors="pt")

# ģenerējam turpinājumu
output = model.generate(inputs, max_length=50, num_return_sequences=1, no_repeat_ngram_size=2, top_p=0.92, temperature=0.7)

# Dekodējam un izvadām ģenerēto tekstu
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print(generated_text)
