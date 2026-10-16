
import pandas as pd
import matplotlib.pyplot as plt

# Load evaluation results
df = pd.read_csv("results/baseline_results.csv")

# Create graph
plt.figure(figsize=(10, 5))

plt.plot(
    df["episode"],
    df["reward"],
    alpha=0.6,
    label="Episode Reward"
)

# Draw average reward line
plt.axhline(
    y=df["reward"].mean(),
    linestyle="--",
    label=f"Mean: {df['reward'].mean():.2f}"
)

plt.xlabel("Evaluation Episode")
plt.ylabel("Total Reward")
plt.title("Baseline PPO: Rocket Landing Performance")
plt.legend()
plt.grid(alpha=0.3)

# Save graph
plt.savefig(
    "results/baseline_rewards.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
