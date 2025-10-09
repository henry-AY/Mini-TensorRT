from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, time
import random
import numpy as np

model_name = "distilgpt2"
device = "cpu"

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
model.eval()

# For deterministic GPU behavior (slower but exact reproducibility)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

prompt = (
    "Explain supervised learning in simple terms.\n\n"
    "Supervised learning is a type of machine learning where"
)

inputs = tokenizer(prompt, return_tensors="pt").to(device)

start = time.time()
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.8,
        top_p=0.95,
        top_k=50,
        repetition_penalty=1.2,
        pad_token_id=tokenizer.eos_token_id
    )
end = time.time()

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Model response:\n", response)
print(f"\nInference time: {end - start:.2f}s")