
# Autonomous Rocket Landing Using Reinforcement Learning

A reinforcement learning project that trains an AI agent
to autonomously land a rocket in Gymnasium's LunarLander-v3
environment using Proximal Policy Optimization (PPO).

## Technologies
- Python
- Gymnasium
- Stable-Baselines3
- PyTorch
- Matplotlib
- Pandas

## Baseline Model
- Algorithm: PPO
- Training timesteps: 500,000
- Evaluation episodes: 100
- Average reward: 115.85
- Landing success proxy: 72%
- Average episode length: 498.5 steps

## Baseline Evaluation

![Baseline Results](results/baseline_rewards.png)

## Reward Shaping Experiment: Time Penalty

To investigate the effect of reward shaping on reinforcement learning performance, I trained two PPO agents for 500,000 timesteps each:

- **Baseline PPO:** Uses Gymnasium's default reward function.
- **Time-Penalty PPO:** Applies an additional penalty of 0.01 for every simulation step.

Both models were evaluated over 100 episodes using identical environment seeds and the original Gymnasium reward function.

### Experimental Results

| Metric | Baseline PPO | Time-Penalty PPO | Change |
|---|---:|---:|---:|
| Average Episode Reward | 115.85 | 189.40 | +63.5% |
| Landing Success Proxy | 72% | 76% | +4 percentage points |
| Average Episode Length | 498.54 | 382.86 | −23.2% |

### Performance Visualizations

![Average Reward](results/average_reward.png)

![Landing Success Proxy](results/success_rate.png)

![Average Episode Length](results/average_steps.png)

### Key Findings

The time-penalty model achieved higher average rewards, a higher preliminary landing-success rate, and shorter episodes than the baseline model.

These results suggest that adding a small time penalty may encourage more efficient landing behavior. However, additional training runs with different random seeds are needed to determine whether the improvement consistently results from the reward modification.

## Future Experiments
- Implement custom reward shaping
- Investigate reward hacking
- Compare multiple reward functions
- Improve landing reliability
