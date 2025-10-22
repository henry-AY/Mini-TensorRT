from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, time, random, numpy as np
import csv, os
import logging
from termcolor import colored
import argparse


# Parsing command-line arguments
parser = argparse.ArgumentParser(description="command-line arguments for model benchmarking")

parser.add_argument("--model", type=str, default="distilgpt2", help="HuggingFace model name")
parser.add_argument("--device", type=str, default="cpu", help="Device to run model on (CPU, MPS, CUDA/ROCm)")
parser.add_argument("--runs", type=int, default=10, help="number of total runs for benchmarking")
parser.add_argument("--seed", type=int, default=42, help="seed choice (default of 42)")

args = parser.parse_args()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Config
model_name = args.model
device = args.device
seed = 42
num_rounds = args.runs
runs_per_round = 5 # Default set to 5
max_new_tokens = 200
csv_path = "model/benchmark/benchmark_results.csv"

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
        elapsed_prev = 999 # Set to arbitrary large value
        tps_prev = 999 # Set to arbitrary large value

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

            time_output_color = "red" if (elapsed > elapsed_prev) else "light_green"
            tps_output_color = "red" if (tps[i] > tps_prev) else "light_green"

            print(f"\tRun {i + 1}: {colored(f'{elapsed:.3f}', time_output_color)}s | Tokens/sec: {colored(f'{tokens_per_sec:.1f}', tps_output_color)}")
            # print("\tResponse preview:", response[:80].replace("\n", " ") + "...")

            elapsed_prev = elapsed
            tps_prev = tps[i]

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

        # For now, write summary line for this round (optional refinement):
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


logging.info(f"\nBenchmarking complete. Results saved to {csv_path}")