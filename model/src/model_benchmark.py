from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, time, random, numpy as np
import csv, os
import logging

# Config
model_name = "distilgpt2"
device = "cpu"
seed = 42
num_rounds = 10
runs_per_round = 5
max_new_tokens = 200
csv_path = "benchmark_results.csv"

torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
model.eval()

prompt = (
    "Explain supervised learning in simple terms.\n\n"
    "Supervised learning is a type of machine learning where"
)

inputs = tokenizer(prompt, return_tensors="pt").to(device)

# Warm up (not timed)
with torch.no_grad(): 
    _ = model.generate(**inputs, max_new_tokens=50, pad_token_id=tokenizer.eos_token_id)

header = [
    "model_name", "device", "round", "run", 
    "elapsed_time_sec", "round_avg_sec", "round_std_sec", 
    "input_length", "generated_length", "tokens_per_sec", 
    "seed", "temperature", "top_p", "top_k", "response_preview"
]

file_exists = os.path.exists(csv_path)

with open(csv_path, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(header)
    
    round_averages = []

    for round_idx in range(num_rounds):
        times = []
        tps = []
        print(f"\nRound {round_idx + 1}/{num_rounds}")
        for i in range(runs_per_round):
            start = time.perf_counter()
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.95,
                    top_k=50,
                    repetition_penalty=1.2,
                    pad_token_id=tokenizer.eos_token_id,
                )
            end = time.perf_counter()

            elapsed = end - start
            times.append(elapsed)
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_length = len(tokenizer.decode(outputs[0]))
            tokens_per_sec = generated_length / elapsed if elapsed > 0 else 0

            # Append tokens per second
            tps.append(tokens_per_sec)

            print(f"\tRun {i+1}: {elapsed:.3f}s | Tokens/sec: {tokens_per_sec:.1f}")
            print("\tResponse preview:", response[:80].replace("\n", " ") + "...")

            # Log this run
            writer.writerow([
                model_name, device, round_idx + 1, i + 1,
                f"{elapsed:.4f}", "", "", 
                len(prompt), generated_length, f"{tokens_per_sec:.2f}",
                seed, 0.8, 0.95, 50, response[:80].replace("\n", " ") + "..."
            ])

        # Compute round stats
        round_avg = np.mean(times)
        round_std = np.std(times)
        round_averages.append(round_avg)
        tokens_per_sec_avg = np.mean(tps)

        print(f"→ Round average: {round_avg:.3f}s ± {round_std:.3f}s")

        # Update previous rows for this round with avg/std
        # (optional refinement – you can skip this if unnecessary)
        # For now, write summary line for this round:
        writer.writerow([
            "ROUND SUMMARY", model_name, device, round_idx + 1, "", 
            f"{round_avg:.4f}", f"{round_std:.4f}", "", "", f"{tokens_per_sec_avg:.4f}", "", "", "", "", ""
        ])

    # Log overall summary
    overall_avg = np.mean(round_averages)
    overall_std = np.std(round_averages)
    writer.writerow([])
    writer.writerow([
        "OVERALL SUMMARY", model_name, device, "", "",
        f"{overall_avg:.4f}", f"{overall_std:.4f}", "", "", "", "", "", "", "", ""
    ])
    writer.writerow([])


logging.info("\n Benchmarking complete. Results saved to", csv_path)