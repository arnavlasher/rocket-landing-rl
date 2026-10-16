
import pandas as pd
import numpy as np

df = pd.read_csv("results/flight_trajectories.csv")

results = []

for (model, episode), group in df.groupby(["model", "episode"]):
    group = group.sort_values("step")
    actions = group["action"].to_numpy()

    # Consecutive runs of the same orientation engine
    runs = []
    current_run = 0
    previous_action = None

    for action in actions:
        if action in (1, 3):
            if action == previous_action:
                current_run += 1
            else:
                if current_run > 0:
                    runs.append(current_run)
                current_run = 1
        else:
            if current_run > 0:
                runs.append(current_run)
            current_run = 0

        previous_action = action

    if current_run > 0:
        runs.append(current_run)

    # Count immediate switches between orientation engines
    switches = sum(
        1
        for i in range(1, len(actions))
        if (actions[i - 1], actions[i]) in [(1, 3), (3, 1)]
    )

    results.append({
        "model": model,
        "episode": episode,
        "mean_run_length": np.mean(runs) if runs else 0,
        "max_run_length": max(runs) if runs else 0,
        "switch_rate": switches / max(len(actions) - 1, 1) * 100
    })

results_df = pd.DataFrame(results)

summary = results_df.groupby("model").agg(
    mean_run_length=("mean_run_length", "mean"),
    max_run_length=("max_run_length", "mean"),
    switch_rate=("switch_rate", "mean")
)

print("\n--- ENGINE PATTERN COMPARISON ---")
print(summary.round(3).to_string())

summary.to_csv("results/engine_patterns.csv")
