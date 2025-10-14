import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("benchmark_results.csv")

# Drop any summary or blank rows
df = df.dropna(subset=["round", "round_avg_sec"])
df = df[df["round"] != "N/A"]

# Convert numeric columns properly
df["round"] = df["round"].astype(int)
df["round_avg_sec"] = df["round_avg_sec"].astype(float)

# Group and average per model/device per round
grouped = df.groupby(["model_name", "device", "round"])["round_avg_sec"].mean().reset_index()

# Plot
plt.figure(figsize=(10, 6))

for (model, device), group in grouped.groupby(["model_name", "device"]):
    plt.plot(
        group["round"],
        group["round_avg_sec"],
        marker="o",
        label=f"{model} ({device})"
    )

plt.title("Average Inference Time per Round")
plt.xlabel("Round #")
plt.ylabel("Average Inference Time (seconds)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()
