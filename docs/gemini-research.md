# Model Car RL Architecture Upgrade Proposals

This document outlines the implementation details for the next tier of optimization strategies to build on top of our existing gSDE, LR-decay, and adjusted initial action variance (`log_std_init=-2`).

---

## 1. Velocity-Linked Grass Penalty

To prevent the model from cutting corners aggressively or treating off-track areas as high-speed recovery zones, we introduce a continuous surface penalty scaled by the vehicle's momentum.

### Mathematical Formulation
Instead of a flat negative constant, the penalty is dynamically calculated at each timestep $t$:

$$\text{Penalty}_{\text{grass}} = -\beta \cdot \text{Surface}_{\text{grass}} \cdot v_t$$

Where:
* $\beta$: Scaling hyperparameter (tune based on reward scale, recommended starting value: `0.5` to `2.0`).
* $\text{Surface}_{\text{grass}}$: A scalar value indicating track violation (either a binary `1.0` if any part of the car touches grass, or a continuous value `[0.0, 1.0]` representing the percentage of wheels/chassis off-track).
* $v_t$: The current linear velocity of the vehicle.

### Rationale
A binary, flat penalty can lead to "paralysis behavior" where the car stops exploring boundaries entirely out of fear of the negative clip. Linking the penalty directly to velocity forces the network to learn real physical consequences: high-speed corner cutting is severely punished, while a low-speed recovery or minor tire slip is treated as a correctable mistake.

---

## 2. Action Continuity (Smoothness Penalty)

To eliminate erratic high-frequency steering oscillations ("wobbling" or "twitching") along straights and during corner entry, we must incentivize fluid control transitions.

### Mathematical Formulation
We add a penalty component to the reward function based on the absolute delta between the current action vector and the immediately preceding action vector:

$$\text{Penalty}_{\text{smooth}} = -\gamma \cdot \sum_{i \in A} |a_{i, t} - a_{i, t-1}|$$

Where:
* $\gamma$: Smoothness regularization coefficient (typically set small, e.g., `0.01` to `0.1`, to avoid overriding the main progress reward).
* $a_{i, t}$: The continuous action value for control output $i$ (e.g., steering, throttle) at the current timestep.
* $a_{i, t-1}$: The executed action value from the previous timestep.

### Rationale
gSDE secures exploratory smooth paths during training, but the final deterministic policy can still converge to noisy micro-corrections to stay perfectly centered. Forcing action continuity stabilizes the vehicle's physical weight distribution, matches real-world servo latency, and preserves kinetic energy down straightaways.

---

## 4. Evaluation-Driven Learning Rate Scheduling

Time-based or step-based learning rate decay drops training step sizes strictly according to an arbitrary clock, completely blind to how well the agent is actually performing. We propose shifting to a performance-contingent validation loop.

Time-based or step-based learning rate decay drops training step sizes strictly according to an arbitrary clock, completely blind to how well the agent is actually performing. This means a model can have its learning rate cut while it is in the middle of a massive breakthrough, or conversely, keep a high learning rate for too long after it has already stabilized. To fix this, we propose shifting to a performance-contingent validation loop that ties your optimization step size directly to your car's actual driving skill.

In this setup, you hook a custom evaluation callback into your training loop at fixed intervals, such as every 10,000 steps. During this checkpoint, the main training pauses briefly, and the agent runs a handful of deterministic evaluation episodes to calculate an average test score. If the evaluation score is actively improving compared to previous checkpoints, the learning rate remains completely unchanged, allowing the actor network to continue making aggressive, high-velocity gradient updates while the momentum is good.

The decay is only triggered when performance completely stalls. If the average evaluation score fails to improve for a pre-determined number of consecutive checkpoints—indicating a performance plateau—the callback automatically scales down the learning rate, typically cutting it in half or by a factor of ten. Once the learning rate drops, the stall counter resets, and the training loop resumes.

This approach acts as a dynamic safety net for your training pipeline. It ensures that large, exploratory gradient steps are preserved for early and mid-stages when the model is still mapping out fundamental racing lines. Then, it saves the micro-adjustments for the exact moments when the car's score flattens out, allowing the weights to settle precisely into an optimal configuration without overshooting the target policy.

## 5. Progress-Based Terminal States (Early Stopping)

When an agent enters a catastrophic state (e.g., completely wedged against an obstacle, caught in a permanent spinning loop on the grass, or driving backwards), continuing the episode to its max timestep limit introduces low-quality, repetitive negative noise into the experience replay buffer.

### Implementation Logic
At each environment interaction loop, maintain a sliding window or a step counter tracking track progress metrics (such as the track centerline index, checkpoints reached, or absolute distance traveled down the track vector).

```python
# Conceptual Environment Step Condition
if current_surface == "grass" and current_velocity < PROGRESS_THRESHOLD:
    no_progress_steps += 1
else:
    no_progress_steps = 0

if no_progress_steps >= MAX_STAGNANT_STEPS:
    terminated = True
    info["terminal_reason"] = "stagnant_off_track"

