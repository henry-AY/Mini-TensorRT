import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("model/benchmark/benchmark_results.csv")

df_filtered = df[df['model_name'] == "ROUND SUMMARY"]
print(df_filtered)

DEVICE = df_filtered['device'].iloc[0]


df_filtered = df_filtered.drop(["model_name", "device", "round", "elapsed_time_sec", "input_length", "generated_length", "seed", "temperature", "top_p", "top_k", "response_preview"], axis=1)

df_tps = df_filtered.drop(["run", "round_avg_sec", "round_std_sec"], axis=1)
df_sec = df_filtered.drop(["tokens_per_sec"], axis=1)

print(df_tps)
print(df_sec)

plt.plot(df_tps)
plt.show()

plt.plot(df_sec)
plt.show()