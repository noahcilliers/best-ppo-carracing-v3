# Improving the Next CarRacing PPO Training Phase

## Baseline, constraints, and the shortest answer

I could not inspect `PROJECT_RECORD.md`, `CarRacing-RL-Plan.md`, `train.py`, or `grass_env.py` from the repo because those files were not available through any accessible source in this chat. The report below is therefore grounded in the public Gymnasium/CarRacing and Stable-Baselines3 sources, plus reproducible CarRacing and visual-control literature, and it is written to fit your current stack and your “change one thing at a time” rule.

The most important thing I found is that your current recipe is already close to the SB3 RL-Zoo PPO recipe in the parts that usually matter most for CarRacing, but it still leaves two unusually relevant gaps for **this** environment and **this** failure mode: the default CarRacing continuous action space is **not symmetric or normalized**, and the environment already contains explicit low-level driving telemetry—true speed, ABS, steering, and gyroscope—that your CNN currently has to infer from a tiny HUD strip in a 96×96 image. Gymnasium’s own source comments note the action space is not symmetric, while SB3’s docs explicitly recommend symmetric `[-1, 1]` action spaces for continuous control and warn that PPO/A2C clip continuous actions while tanh-squashing or Beta policies handle bounds more correctly. At the same time, CarRacing’s docs and source show the HUD renders exactly the signals you wish the agent used earlier for braking and turn stabilization. citeturn28view0turn22search0turn22search6turn14search0turn1view1turn29view0

My top-line recommendation is therefore: **test action-space redesign first, then explicit telemetry, then tail-aware checkpointing and PPO stability controls**. Those have the best impact-per-effort for reducing catastrophic sharp-turn failures while keeping you inside feedforward PPO and SB3. By contrast, the RL-Zoo deltas you intentionally skipped—especially `frame_skip=2`, `frame_stack=2`, and 64×64 resize—look less aligned with your current weakness, because your issue is not “insufficient throughput” but “late braking / missed sharp turns / tail crashes.” RL-Zoo does use those settings on CarRacing, but other CarRacing results suggest that **lower** frameskip at evaluation can materially improve score, and that **more** temporal context can help in this highly dynamic task. citeturn2view0turn19view0turn19view1

A concise priority order is below.

| Priority | Change | Expected effect on mean | Expected effect on variance | Cost and risk |
|---|---|---:|---:|---|
| Very high | Symmetric action wrapper plus simpler longitudinal control | Medium | High reduction | Low cost, low-to-medium risk |
| Very high | Add structured telemetry as a second input | Medium to high | High reduction | Medium cost, low risk |
| High | Tail-aware checkpoint selection on hard-turn seeds | Small to medium | High reduction | Low cost, very low risk |
| High | `target_kl` early stopping | Small to medium | Medium reduction | Very low cost, low risk |
| Medium | Potential-based progress and heading shaping | Medium | Medium to high reduction | Medium cost, medium risk |
| Medium | Tiny action-rate penalty or CAPS-style smoothness | Small to medium | Medium reduction | Medium cost, medium risk |
| Medium | Mild curriculum on completion threshold or track curvature | Small to medium | Medium reduction | Medium cost, medium risk |
| Medium | Custom wider CNN or image+vector extractor | Medium | Small to medium reduction | Medium cost, medium risk |
| Medium | DrAC-style image augmentation | Medium | Medium reduction | Higher cost, higher risk |
| Low for now | RL-Zoo’s `frame_skip=2`, `frame_stack=2`, 64×64 resize trio | Unclear | Unclear or negative | Low cost, but poor fit to failure mode |

## Highest-impact changes to test first

### Normalize the action space and simplify longitudinal control

**What it does.** Wrap CarRacing so PPO sees a symmetric action space in `[-1, 1]`, instead of the native `Box([-1, 0, 0], [1, 1, 1])`. My preferred first A/B is a **2D action space**: one steering axis and one longitudinal axis, where positive values map to gas and negative values map to brake. That automatically prevents simultaneous gas and brake, makes the action space symmetric, and gives the policy a much cleaner braking decision boundary. A slightly more conservative alternative is a symmetric 3D wrapper that still rescales to the original controls but leaves gas and brake separate. citeturn28view0turn22search0turn22search6turn21search5

**Why it should help.** This is one of the rare cases where the environment itself and the library docs line up with your observed failure mode. In Gymnasium’s CarRacing source, the authors explicitly comment that the continuous action space is not symmetric or normalized. SB3’s documentation separately recommends making continuous action spaces symmetric and normalized, because most policy-gradient methods use Gaussian action distributions and non-symmetric spaces can hurt learning. SB3 also explicitly warns that PPO clips continuous actions to bounds, while tanh-squashing or Beta-style bounded policies handle action bounds more correctly. On top of that, Gymnasium’s CarRacing docs remind human players not to accelerate and turn at the same time because the car is powerful and rear-wheel drive; the same structural prior is useful for a policy that is overdriving sharp corners. Put differently: this is a low-effort way to remove one source of control ambiguity and one source of distribution mismatch at the same time. citeturn28view0turn22search0turn22search6turn14search0turn1view1

**How to implement it in SB3.** Keep PPO, gSDE, and your current network. Add an `ActionWrapper` that exposes either `Box(low=-1, high=1, shape=(2,), dtype=np.float32)` or a symmetric 3D box. For the 2D version:

```python
class SymmetricCarAction(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

    def action(self, a):
        a = np.asarray(a, dtype=np.float32)
        steer = float(np.clip(a[0], -1.0, 1.0))
        longi = float(np.clip(a[1], -1.0, 1.0))
        gas = max(longi, 0.0)
        brake = max(-longi, 0.0)
        return np.array([steer, gas, brake], dtype=np.float32)
```

If you stay on Gaussian PPO with `use_sde=True`, also test `policy_kwargs=dict(..., squash_output=True)` in a follow-up run, because SB3 exposes this for gSDE policies. I would **not** combine the 2D wrapper and `squash_output=True` in the first A/B; test the wrapper first, because it is the more important change. citeturn11view0turn14search0turn22search7

**Expected effect.** I would expect **moderate mean improvement** and, more importantly, a **large variance reduction**. This is the single most plausible low-cost fix for “the turn is visible but the car does not brake correctly / soon enough,” because it simplifies the action decision the policy has to make at exactly the troublesome moments.

**Cost and risk.** Engineering cost is low. The real risk is a small loss in absolute ceiling if the 2D longitudinal axis removes useful trail-braking nuance. That is why I would test 2D first, and if it lowers peak lap pace too much, try a symmetric 3D wrapper next.

**Multi-car relevance.** High. Cleaner throttle-brake semantics, better bounded-action behavior, and lower tail risk all transfer directly to the multi-car phase.

### Add explicit telemetry as a second observation input

**What it does.** Convert the observation to a `Dict` with one key for the image stack and one key for a small vector of telemetry such as speed, wheel slip / ABS proxies, steering angle, and yaw rate. Then switch from `CnnPolicy` to `MultiInputPolicy`. CarRacing already renders these signals in the HUD; promoting them to a structured vector is the most direct feedforward way to close the “sees the turn but does not brake properly” gap without moving to recurrence. citeturn1view1turn29view0turn30view0turn11view0turn33search1

**Why it should help.** CarRacing’s docs say the bottom of the observation contains true speed, four ABS sensors, steering wheel position, and gyroscope. The source code shows those values are computed directly from the car’s linear velocity, wheel angular speeds, steering joint angle, and hull angular velocity. A scientific-Python CarRacing tutorial even includes code that parses speed, steering, and gyro from the bottom bar, which is a good reminder that the information is there but visually cramped. A CarRacing mixed-convolution paper likewise reported that combining image input with those sensor signals produced the best overall performance among its tested architectures. In your current setup, the CNN must both parse the road and decode tiny low-level telemetry from a small grayscale image. That is a bad trade when the failure mode is specifically late braking and over-speeding into corners. citeturn1view1turn29view0turn21search4turn30view0

**How to implement it in SB3.** Wrap the env to return:

```python
{
    "img": stacked_gray_image,          # e.g. 96x96x4
    "telemetry": np.array(
        [speed, abs0, abs1, abs2, abs3, steer_angle, yaw_rate],
        dtype=np.float32
    )
}
```

Use `PPO("MultiInputPolicy", env, policy_kwargs=...)`. SB3 PPO supports `MultiInputPolicy`, and `VecFrameStack` supports `Dict` observation spaces, including different stacking rules for keys. The simplest version is to frame-stack only the image key and leave `telemetry` unstacked or manually append the previous one or two telemetry vectors in the wrapper. Normalize the telemetry yourself to roughly `[-1, 1]` ranges rather than enabling blanket observation normalization on pixels. citeturn11view0turn33search0turn33search1

A practical extractor layout is:

- keep NatureCNN or your current image encoder for `img`,
- feed `telemetry` through a tiny MLP, such as `[32, 32]`,
- concatenate the two embeddings,
- keep your existing `net_arch`.

**Expected effect.** I expect **medium to high mean gain** and **large variance reduction**, especially if the catastrophic episodes are mostly “entered the corner too fast / did not realize current speed-yaw state implied braking now.”

**Cost and risk.** Engineering cost is medium, but the risk is low because you are not changing the algorithm, only the observation encoding.

**Multi-car relevance.** Very high. In multi-car racing, the image gets busier; explicit ego telemetry usually becomes more useful, not less.

### Select checkpoints on lower-tail performance, not average reward alone

**What it does.** Keep the policy and training loop, but change model selection. Build a fixed validation suite of deterministic CarRacing seeds, including a subset of **hard-turn tracks**, and score checkpoints with a tail-sensitive objective such as 10th percentile return, CVaR-like average of the worst episodes, or `mean - 0.5 * std`. This directly optimizes for your true problem: reducing catastrophic off-track runs. citeturn20search1turn16search0turn19view0turn32view0

**Why it should help.** The deep-RL reproducibility literature is blunt that variance and non-determinism make point estimates easy to misread. CarRacing PPO work that reached very high scores typically saved the best model based on repeated evaluation during training rather than assuming the final checkpoint was best. Since Gymnasium tracks are procedurally generated and reproducible under fixed seeds, you can go one step further and maintain a “stress suite” rich in sharp bends. This is the cheapest way to improve consistency **even if training itself is unchanged**. citeturn20search1turn19view0turn1view2

**How to implement it in SB3.** During periodic eval:

1. Reset across a fixed bank of seeds.
2. Compute track curvature after reset from the stored `track` list if you want a hard-turn subset; the source stores each track segment’s angle (`beta`) in `self.track`, and the border-generation logic already detects hard turns from angle changes.
3. Evaluate deterministic policy on:
   - a random validation set,
   - a hard-turn validation set.
4. Save the checkpoint that maximizes either:
   - `p10_hard_turn_return`, or
   - `0.5 * mean_random + 0.5 * p10_hard_turn`.

Use the standard mean reward too, but only as a secondary selection criterion. citeturn32view0turn19view0turn16search0

**Expected effect.** This can produce **small to medium mean gains** and a **large variance reduction** at almost no research risk.

**Cost and risk.** Very low implementation cost. The only real risk is overfitting to too-small a validation suite, so keep a held-out seed bank for final comparison.

**Multi-car relevance.** High. Tail-aware checkpointing becomes even more valuable once collisions and interaction failures enter the loop.

## Stability-focused training changes after the top three

### Turn on `target_kl` before you start sweeping many PPO hyperparameters

**What it does.** Add SB3’s `target_kl` to stop PPO updates early when the policy moves too far in one update cycle. This is a very cheap way to reduce late-training instability without rewriting the algorithm. SB3’s PPO docs explicitly say `target_kl` exists because clipping alone is not always enough to prevent large updates. The PPO implementation-details literature likewise describes KL-based early stopping as an explicit trust-region safeguard on top of fixed epoch counts. citeturn11view0turn12view0

**Why it should help.** Your baseline already uses linear learning-rate decay, which addresses one source of instability. But with eight environments, 512 steps, and ten epochs, PPO can still occasionally make relatively aggressive updates from a single rollout batch. The PPO implementation-details writeup notes that if approximate KL gets too high, the policy is often changing too quickly; as a rule of thumb, `approx_kl` staying below roughly `0.02` is a healthy sign. For a policy that is already reasonably strong but occasionally forgets itself in hard corners, KL-capping is exactly the kind of low-risk stabilization that can reduce destructive policy drift between “good” and “fragile” checkpoints. citeturn12view0turn11view0

**How to implement it in SB3.** Start with `target_kl=0.015`. Log `approx_kl`, `clip_fraction`, and eval tail metrics. If updates almost never hit the KL cap, maybe move to `0.02`. If they hit it constantly and learning slows too much, back off. I would test this **before** touching `n_epochs`, `batch_size`, `gamma`, or `gae_lambda`, because `target_kl` is both cheaper and more directly aimed at update stability. citeturn11view0turn12view0

**Expected effect.** I expect a **small to moderate mean gain** and **moderate variance reduction**.

**Cost and risk.** Very low cost. The main risk is under-updating if the threshold is too tight.

**Multi-car relevance.** High. PPO update stability matters even more once self-play enters the picture.

### Treat multi-seed confirmation as part of the intervention, not an afterthought

**What it does.** Use a staged protocol: quick single-seed pilot to reject obvious losers, then three-seed confirmation for any change that looks promising, and only then consider longer training. This is not glamorous, but for your current problem it is the difference between real consistency gains and noisy optimism. citeturn20search1

**Why it should help.** The broader deep-RL literature repeatedly shows that random seeds, environment stochasticity, and checkpoint choice can materially change measured conclusions. Also, 4M timesteps is **not obviously too short** for tuned CarRacing PPO: RL-Zoo uses 4M on CarRacing-v3, and a PPO-based raw-pixel CarRacing study reported top-tier score at 4M with repeated evaluations and checkpoint selection. That argues against “just train longer” as a first-line answer. Stabilize first; extend later if the stabilized variant is clearly better. citeturn20search1turn2view0turn19view0

**How to implement it.** My recommended protocol is:

- pilot A/B: one seed, 1.5M–2M steps, frequent evals,
- confirmatory A/B: three seeds to 4M,
- only winners advance to 6M–8M.

Track not just mean reward, but also standard deviation, 10th percentile return, catastrophic episode count, and maybe median number of tiles completed before first major off-track event.

**Expected effect.** The effect is mostly epistemic rather than algorithmic: fewer false positives and better model selection. It will save you time in this phase.

**Cost and risk.** Medium compute cost, almost no scientific risk.

**Multi-car relevance.** Essential. Self-play magnifies variance problems.

## Reward and curriculum changes worth trying after the above

### Use potential-based shaping on progress and heading, not a naive speed bonus

**What it does.** If you add shaping, add it in a way that respects the original task. In CarRacing, the base environment already rewards progress through tile visitation and penalizes wasted time each frame. A raw speed bonus is therefore not my first choice; it risks encouraging the exact “too fast into the corner” behavior you are trying to remove. Instead, I would test **potential-based shaping** on dense progress along the track and perhaps a very small heading-alignment term relative to the local track tangent. citeturn1view1turn10search0turn24view0turn32view0

**Why it should help.** Potential-based reward shaping is attractive here because it accelerates learning while preserving the optimal policy in an MDP, and the same policy-invariance idea has been extended to general-sum stochastic games, which is a nice property given your eventual self-play plans. In autonomous racing, recent work has emphasized balanced reward design because unstable decision quality around sharp bends leads to boundary hits and inconsistent training outcomes. That is exactly your present issue. The aim should not be “reward speed more,” but “make the agent value the *right* local precursor behaviors to safe fast driving.” citeturn10search0turn10search11turn24view0

**How to implement it in CarRacing.** The CarRacing source stores the track centerline as tuples of angles and coordinates, and each tile has an index. That makes practical shaping possible. A reasonable first version is:

- find nearest track segment,
- compute dense progress potential from nearest segment index or projected arc length,
- compute heading error between car orientation and local track tangent,
- shape with  
  `r' = r + γ * Φ(s') - Φ(s)`  
  where `Φ(s) = a * progress + b * cos(heading_error)`.

I would leave centerline-offset shaping for later or keep it extremely weak, because strong lane-centering rewards are fine in single-car mode but can become anti-overtaking priors in multi-car racing.

**Expected effect.** **Moderate mean gain** and **medium-to-high variance reduction** are plausible, but this is not as low-risk as the action / observation fixes.

**Cost and risk.** Medium implementation cost. Main risk is reward hacking if the shaping ceases to be truly potential-based or if the heading term is overweighted.

**Multi-car relevance.** Progress-based shaping: yes. Strong lane-centering shaping: no, or at least not unchanged.

### Add a tiny action-rate penalty, or go one step further with CAPS-style regularization

**What it does.** Encourage smoother control by penalizing abrupt action changes—especially sharp steering reversals and brake spikes—or, if you are willing to touch the PPO loss, by adding a CAPS-style regularization term that directly encourages temporal and spatial smoothness in the action policy. citeturn31view1

**Why it should help.** The “smooth control” literature argues that RL controllers often learn oscillatory actions even when reward engineering is already elaborate, and that direct smoothness regularization can improve behavior without redesigning the environment. CAPS specifically proposes temporal smoothness—adjacent actions should be similar—and spatial smoothness—similar states should map to similar actions. That matches the CarRacing failure pattern well: once the agent makes one slightly wrong correction in a sharp turn, oscillatory follow-up actions can turn a recoverable corner into an episode-ending grass excursion. citeturn31view1

**How to implement it in SB3.** The lowest-cost version is just an env wrapper that tracks previous action and subtracts a very small penalty such as:

- `λs * |Δsteer|`
- `λb * |Δbrake|`
- optionally `λg * |Δgas|`

Start extremely small because you already normalize rewards and because too-strong smoothing will cause understeer and late braking. A good first target is “remove twitch,” not “make steering slow.” If you later want the more principled version, CAPS can be added as a policy regularizer outside vanilla SB3, but that moves this from SB3-native to lightly external.

**Expected effect.** I would expect **small-to-moderate mean gain** and **moderate variance reduction** if the weights are tiny and well tuned.

**Cost and risk.** Medium. This is easy to overdo.

**Multi-car relevance.** High. Smooth low-level control usually helps collision avoidance and close-quarters racing.

### Use a mild curriculum only if the first wave does not solve the tails

**What it does.** Make early training slightly easier, then anneal back to the real environment. The built-in low-effort lever is `lap_complete_percent`, which Gymnasium exposes as an env argument. A more targeted curriculum would classify tracks by curvature and sample simpler ones first, then progressively introduce harder ones. citeturn1view1turn28view0turn23view0turn23view1

**Why it should help.** Curriculum learning has shown value in autonomous-driving and autonomous-racing settings, especially for high-curvature, dynamically difficult scenarios. One recent drifting paper explicitly decomposed the task into stages of increasing complexity to handle high-speed sharp-corner control, while a recent large-scale driving-simulator paper reported large efficiency wins from adaptive curricula over uniform domain randomization. That said, you are not at an early training stage anymore; you are near the feedforward ceiling. So I see curriculum as a **second-wave** intervention, not a first-wave fix. citeturn23view0turn23view1

**How to implement it.** The lightest variant is:

- phase A: `lap_complete_percent=0.80`
- phase B: `0.90`
- phase C: `0.95`

A better but more custom version is:

- reset env with a seed,
- compute max curvature from `track[i][1]`,
- keep only moderate-curvature tracks early,
- then widen the acceptance distribution over time.

Always finish with full-distribution fine-tuning.

**Expected effect.** **Small-to-moderate final mean gain** and **moderate variance reduction** are possible, but I would not spend your first experiments here.

**Cost and risk.** Medium. The main problem is distribution shift if the final stage is too short.

**Multi-car relevance.** Very high later, but only medium right now.

## External or higher-risk upgrades with real upside

### DrAC-style image augmentation is the most interesting non-SB3-native visual upgrade

**What it does.** Regularize PPO with image augmentation—especially random shifts or crops—but do it in a way that is theoretically aligned with actor-critic training. DrAC was designed specifically because naive augmentation can break actor-critic objectives; it regularizes both policy and value outputs with respect to augmented observations. citeturn27search2turn27search5turn27search13

**Why it should help.** In visual RL, data augmentation has strong empirical support. DrQ and RAD showed major gains for image-based continuous control, and DrAC extended augmentation to actor-critic methods like PPO. On Procgen, DrAC/UCB-DrAC improved training performance and significantly improved test-time generalization relative to standard PPO. The relevance to CarRacing is straightforward: procedural tracks already create some variation, but the policy still learns from pixels and still benefits from invariances to small translations and crops. citeturn5search0turn5search2turn27search2turn27search5

**How to implement it.** I would label this **external** rather than SB3-native. If you want the easy probe first, try a training-only random-shift wrapper on the image input and leave evaluation unaugmented—but treat that as exploratory, not principled, because naive augmentation with PPO can distort the actor loss. The more serious version is a custom PPO loss in the style of DrAC.

**Expected effect.** Potentially **moderate mean improvement** and **moderate variance reduction**, especially if visual generalization rather than control semantics is the remaining bottleneck.

**Cost and risk.** Higher cost. I would not attempt it before the action/telemetry/checkpoint wave.

**Multi-car relevance.** Very high. Interaction-rich scenes usually increase the value of robust visual features.

### If you want one external action-head experiment, make it Beta PPO

**What it does.** Replace Gaussian action sampling with a bounded Beta distribution, or at least approximate better-bounded control by pairing a symmetric wrapper with squashed outputs. This is the most compelling “action-space design” experiment beyond simple wrappers. citeturn13search2turn22search6turn14search0

**Why it should help.** The bounded-action argument is theoretically clean and empirically relevant here. Chou’s work on Beta policies argues that Gaussian policies are mismatched to bounded action spaces because of their infinite support. A later PPO-with-Beta paper reported better stability, faster convergence, better final reward, and—crucially for your case—a **63% higher CarRacing success rate** than Gaussian PPO on the image-based CarRacing task. SB3’s own docs broadly agree with the direction: clipping Gaussian actions is a bandage; squashing or Beta-style bounded policies are more correct treatments of action bounds. citeturn13search0turn13search2turn22search6turn14search0

**How to implement it.** This is not a drop-in SB3 PPO option today, so I would classify it as external. If you want the lowest-friction approximation first, use the **symmetric action wrapper** from the first recommendation and then test `squash_output=True` with gSDE. A true Beta head requires a custom policy distribution.

**Expected effect.** **Medium-to-high upside** on both mean and variance.

**Cost and risk.** Substantially higher engineering cost than the wrapper-only change.

**Multi-car relevance.** High.

### A wider custom CNN is worth trying, but only after observation and action fixes

**What it does.** Replace NatureCNN with a slightly wider custom feature extractor. I do **not** think you need a radically deeper vision stack for single-car CarRacing, but I do think a very modest capacity increase is a reasonable second-wave test—particularly if you add telemetry and want a better joint image-plus-vector extractor. citeturn17search0turn19view1turn18search0

**Why it should help.** SB3 supports custom feature extractors directly. Independent CarRacing reports have found that larger CNNs can learn materially faster than smaller ones, and one A3C CarRacing paper reported that a carefully designed CNN feature extractor improved average reward by about 100 points over a vanilla baseline. I would not overread those results because they are not the same algorithmic setup you are running, but they are enough to justify a capacity test once the more obvious control issues are handled. citeturn17search0turn19view1turn18search0

**How to implement it.** Keep kernels similar to NatureCNN, but widen channel counts modestly—for example 64/128/128—and use `features_dim` in the 512–1024 range. On your CPU-only setup, I would avoid residual towers or very deep backbones until simpler changes are exhausted.

**Expected effect.** Likely **moderate mean gain** and **small-to-moderate variance reduction**.

**Cost and risk.** Medium, mostly in wall-clock time.

**Multi-car relevance.** High.

## Changes I would explicitly deprioritize for this phase

I would **not** start by importing the RL-Zoo CarRacing deltas you deliberately skipped—especially `frame_skip=2`, `frame_stack=2`, and 64×64 resize. RL-Zoo indeed uses that throughput-oriented recipe, but your present problem is not a generic tuning deficit; it is a sharp-turn tail problem. Independent CarRacing evidence points the other direction on the most relevant axes: in one PPO-based CarRacing study, lowering frameskip at test time improved results by roughly 70–120 points, and a separate CarRacing study found that a more temporally extended state outperformed a shorter one in this dynamic task. In other words, **less actuation frequency and less temporal context are not the first levers I would pull when the issue is late braking and hard-turn consistency**. If you want to touch one skipped RL-Zoo delta early, the only one I would move up the list is `batch_size=128`; the others are low priority for your current diagnosis. citeturn2view0turn19view0turn19view1

I would also **not crop the bottom HUD** unless you first expose its contents as structured telemetry. An older CarRacing writeup removed the bottom panel and stacked grayscale frames, and that worked well in its setup, but Gymnasium’s current CarRacing docs make clear that the bottom strip contains true speed, ABS, steering, and gyroscope signals. For your specific braking-timing problem, throwing away that information before trying to use it explicitly feels backwards. citeturn8view0turn1view1turn29view0

I would **not switch back to RGB right now** for the single-car phase. RL-Zoo’s tuned PPO recipe uses grayscale, and your current grayscale baseline is already much stronger than a naive recipe. For eventual multi-car racing, color may become more important if opponent appearance or contact cues matter, but for single-car CarRacing I would spend that complexity budget on action semantics and telemetry first. citeturn2view0turn4search0

Finally, I would **not assume that longer training by itself is the answer**. Your 4M budget is already in the same ballpark as tuned public recipes and strong PPO CarRacing results. Without better checkpointing and harder stability controls, longer training can simply move you among fragile checkpoints rather than reliably reducing tail crashes. citeturn2view0turn19view0

## Recommended A/B order

The sequence below gives you the fastest path to useful signal while respecting your “one change at a time” rule.

1. **Symmetric action wrapper with 2D longitudinal control.**  
   This is my highest-confidence, lowest-cost intervention for reducing catastrophic sharp-turn episodes.

2. **Structured telemetry with `MultiInputPolicy`.**  
   If the wrapper helps but does not fully solve braking inconsistency, this is the next place I would go.

3. **Tail-aware checkpoint selection on fixed hard-turn seeds.**  
   You can implement this alongside any training change, but I would treat it as its own A/B because it can alter conclusions immediately.

4. **`target_kl=0.015` with monitoring of `approx_kl` and `clip_fraction`.**  
   Cheap stabilization pass.

5. **Tiny action-rate penalty.**  
   Only after the action wrapper and telemetry, because those are cleaner fixes to the same symptom.

6. **Potential-based progress and heading shaping.**  
   Start conservative. Avoid raw speed bonuses as the first reward-shaping test.

7. **Only then try second-wave architecture changes.**  
   First a wider extractor; later, if still needed, external visual augmentation or Beta PPO.

8. **Treat longer training as a multiplier, not a first move.**  
   Extend only the variants that already show better lower-tail behavior at 4M.

For evaluation, I would stop reporting only `mean ± std` and add at least three more quantities for every A/B: **10th percentile return, catastrophic-episode rate, and hard-turn validation-suite mean**. That will make the next phase much more aligned with the actual problem you described, and it will keep you from choosing a higher-mean but more brittle checkpoint by accident. citeturn20search1turn16search0turn19view0