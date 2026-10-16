
import pandas as pd
import matplotlib.pyplot as plt

# Load model comparison results
df = pd.read_csv("results/model_comparison.csv")

# Calculate average performance
summary = df.groupby("model").agg(
    average_reward=("reward", "mean"),
    success_rate=("success", "mean"),
    average_steps=("steps", "mean")
)

summary["success_rate"] *= 100

# Define metrics to graph
metrics = [
    ("average_reward", "Average Episode Reward", "average_reward.png"),
    ("success_rate", "Landing Success Proxy (%)", "success_rate.png"),
    ("average_steps", "Average Episode Length (Steps)", "average_steps.png")
]

# Create and save each graph
for column, title, filename in metrics:
    plt.figure(figsize=(7, 5))

    bars = plt.bar(
        summary.index,
        summary[column]
    )

    plt.ylabel(title)
    plt.title(f"Baseline vs. Time-Penalty PPO")

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.2f}",
            ha="center",
            va="bottom"
        )

    plt.ylim(0, summary[column].max() * 1.2)
    plt.tight_layout()

    plt.savefig(
        f"results/{filename}",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

print("All three comparison graphs saved!")
