
import pandas as pd

df = pd.read_csv("results/flight_trajectories.csv")

# Determine whether an orientation engine was used
df["orientation_engine"] = df["action"].isin([1, 3]).astype(int)

# Calculate usage percentage for each model
usage = df.groupby("model")["orientation_engine"].mean() * 100

print("\nOrientation Engine Usage (% of Steps):")
print(usage.round(2))

# Compare usage across episode progress
df["episode_length"] = df.groupby(
    ["model", "episode"]
)["step"].transform("max")

df["progress"] = df["step"] / df["episode_length"]

df["phase"] = pd.cut(
    df["progress"],
    bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
    labels=[
        "0-20%",
        "20-40%",
        "40-60%",
        "60-80%",
        "80-100%"
    ],
    include_lowest=True
)

phase_usage = (
    df.groupby(["model", "episode", "phase"], observed=True)
    ["orientation_engine"]
    .mean()
    .reset_index()
)

summary = (
    phase_usage.groupby(["phase", "model"], observed=True)
    ["orientation_engine"]
    .mean()
    .unstack() * 100
)

print("\nOrientation Engine Usage by Flight Phase (%):")
print(summary.round(2))

summary.to_csv("results/engine_usage_by_phase.csv")
