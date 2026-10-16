
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results/flight_trajectories.csv")

# Normalize progress so episodes of different lengths
# can be compared on the same scale
df["episode_length"] = df.groupby(
    ["model", "episode"]
)["step"].transform("max")

df["progress"] = df["step"] / df["episode_length"]

# Divide each episode into 5 equal progress phases
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

# Average rotation within each episode and phase
episode_phase = df.groupby(
    ["model", "episode", "phase"],
    observed=True
)["abs_angular_velocity"].mean().reset_index()

# Give each episode equal weight
phase_summary = episode_phase.groupby(
    ["model", "phase"],
    observed=True
)["abs_angular_velocity"].mean().unstack(0)

print("\nAverage Angular Velocity by Flight Phase:")
print(phase_summary.round(4))

# Plot comparison
phase_summary.plot(
    kind="line",
    marker="o",
    figsize=(9, 5)
)

plt.title("Angular Velocity Across Flight Progress")
plt.xlabel("Flight Progress")
plt.ylabel("Average Absolute Angular Velocity")
plt.grid(True)
plt.tight_layout()

plt.savefig(
    "results/rotation_by_phase.png",
    dpi=300
)

plt.show()
