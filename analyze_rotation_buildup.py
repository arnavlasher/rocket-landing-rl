
import pandas as pd
import numpy as np

df = pd.read_csv("results/flight_trajectories.csv")

results = []

for (model, episode), group in df.groupby(["model", "episode"]):
    group = group.sort_values("step").reset_index(drop=True)

    actions = group["action"].to_numpy()
    angular = group["abs_angular_velocity"].to_numpy()

    start = 0

    while start < len(actions):
        action = actions[start]

        if action not in (1, 3):
            start += 1
            continue

        end = start

        while (
            end + 1 < len(actions)
            and actions[end + 1] == action
        ):
            end += 1

        run_length = end - start + 1

        if run_length >= 2:
            # Compare angular velocity after the run
            # with angular velocity after its first action
            change = angular[end] - angular[start]

            results.append({
                "model": model,
                "episode": episode,
                "run_length": run_length,
                "rotation_change": change,
                "rotation_increased": int(change > 0)
            })

        start = end + 1

results_df = pd.DataFrame(results)

summary = results_df.groupby("model").agg(
    average_run_length=("run_length", "mean"),
    average_rotation_change=("rotation_change", "mean"),
    rotation_increase_rate=("rotation_increased", "mean")
)

summary["rotation_increase_rate"] *= 100

print("\n--- ROTATION BUILDUP ANALYSIS ---")
print(summary.round(4).to_string())

results_df.to_csv(
    "results/rotation_buildup.csv",
    index=False
)

