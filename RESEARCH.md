# Self-Improving Behavioural Cloning for Robot Arm Control
## Research Explainer — What We're Building and Why

---

## 1. The Problem: Learning to Control a Robot Arm

We want a robot arm to move its gripper to any arbitrary 3-D position in its workspace — a task called **goal-conditioned reaching**. This is one of the simplest motor control tasks in robotics, but it captures the essential structure of almost every manipulation problem: you have a continuous state space, a continuous action space, a desired outcome, and you need a policy that reliably achieves it.

The question is: **how do you get that policy?**

There are three canonical approaches in modern robot learning:

| Approach | Core idea | Main limitation |
|---|---|---|
| Reinforcement Learning (RL) | Learn from scalar rewards via trial and error | Sample inefficient; reward engineering is hard |
| Behavioural Cloning (BC) | Supervised learning from expert demonstrations | Covariate shift; doesn't generalise beyond demos |
| **Self-Improvement (ours)** | BC cold start → rollout → filter successes → retrain | Middle ground: no reward design, generalises beyond demos |

This project implements the third approach and tests a specific hypothesis: **can a policy that was only demonstrated a narrow part of the goal distribution generalise to the full distribution through self-improvement alone?**

---

## 2. Why Behavioural Cloning Alone Fails

Behavioural Cloning treats robot learning as supervised learning. Given a dataset of (state, action) pairs collected from an expert:

```
π_θ = argmin_θ  E_{(s,a)~D_expert} [ || π_θ(s) - a ||² ]
```

This is intuitive and works surprisingly well. But it has a fundamental flaw identified by **Ross & Bagnell (2010)** in their seminal DAgger paper: **covariate shift**.

### 2.1 The Covariate Shift Problem

During training, the policy sees states sampled from the expert's state distribution `d_expert`. But at test time, the policy visits its *own* state distribution `d_π`. These are not the same.

Concretely: the expert always takes good actions, so it never visits "bad" states. The BC policy, being imperfect, occasionally drifts into states the expert never visited. In those states it has no supervised signal — it is extrapolating from the training distribution. Small errors **compound**: a slight deviation from the expert trajectory puts the agent in an out-of-distribution state, where it makes a larger error, which leads to an even more out-of-distribution state, etc.

Formally, BC error grows as `O(T²)` in the horizon `T` (Ross & Bagnell 2010). For a 50-step episode like FetchReach, compounding errors can completely derail performance despite high training accuracy.

### 2.2 The Goal Distribution Problem (Our Specific Setting)

Beyond covariate shift, we introduce a second failure mode: **incomplete goal coverage in demonstrations**.

In practice, collecting robot demonstrations is expensive. A human operator might only demonstrate reaching to goals in part of the workspace (e.g., goals above a certain height, or within a certain x-range). A pure BC policy trained on this restricted dataset will perform well on demonstrated goals but fail on unseen ones — not because of compounding errors, but because it has genuinely never seen a training example for those goals.

This is the experiment we run: we restrict demos to goals with `desired_goal[z] > 0.45`, and measure how well the policy generalises to goals with `z ≤ 0.45` (OOD region).

---

## 3. The Classical Fix: DAgger

The standard solution to covariate shift is **DAgger** (Dataset Aggregation, Ross et al. 2011). The algorithm:

```
D₀ = collect expert demos
π₀ = train_BC(D₀)
for i in 1..K:
    roll out πᵢ₋₁ in the environment
    for each state s visited:
        query expert: a* = expert_policy(s)   ← KEY STEP
    Dᵢ = Dᵢ₋₁ ∪ {(s, a*)}
    πᵢ = train_BC(Dᵢ)
```

DAgger directly addresses covariate shift: it asks the expert what to do from states the *learner* visits, not states the expert visits. This closes the distribution gap.

**DAgger's convergence guarantee** (Ross et al. 2011): with `N` iterations, the policy achieves expected error `O(ε* · T)` rather than `O(ε* · T²)`, where `ε*` is the irreducible expert mimicry error. This is a qualitative improvement.

### Why We Don't Use DAgger

DAgger requires **online access to an expert** — every state visited by the learner must be labelled. For robot learning this is often impractical:

- Human demonstrators cannot be on-call for thousands of rollouts.
- Scripted experts may not exist for complex tasks.
- Real robot rollouts are slow and expensive.

Our approach eliminates the online expert entirely after the initial demo collection. The self-improvement signal comes exclusively from the policy's own success/failure.

---

## 4. Our Approach: Offline Self-Improvement via Success Filtering

The algorithm is simple to state:

```
D₀ = collect_expert_demos(restrict goals to partial distribution)
π₀ = train_BC(D₀)

for i in 1..K:
    τ₁...τₙ = rollout(πᵢ₋₁, n=200 episodes)     # self-rollout
    S = {τ : τ.success == True}                    # filter successes
    Dᵢ = Dᵢ₋₁ ∪ S                                  # grow dataset
    πᵢ = train_BC(Dᵢ)                              # retrain from scratch
```

No expert queries after step 0. The success signal comes from the environment's `is_success` flag (which just checks if the gripper is within 5cm of the goal).

### 4.1 Why Does This Work?

The key insight is that successful self-rollouts are **valid imitation targets**. A trajectory that reaches the goal, even if it arrived there via a slightly suboptimal path, is a legitimate (state, action) sequence that the policy can learn from. Adding it to the training set:

1. **Covers more of the state space** — the learner has visited states the expert never did, and labelled them with actions that worked.
2. **Covers more of the goal space** — even if demos only covered goal region A, if the policy accidentally succeeds in goal region B, those transitions get added to the dataset.
3. **Reduces compounding error** — the policy has now been trained on states along its own trajectory, not just the expert's.

This is essentially **self-distillation**: the model's successful outputs become its own training targets for the next iteration.

### 4.2 Connection to the Literature

This approach sits at the intersection of several research threads:

**ReST (Reinforced Self-Training)** — Gulcehre et al., 2023 (Google DeepMind)  
Applied to language models: generate candidates, filter by reward, fine-tune. Our work is the robot learning analogue, where "reward" = binary success and "generation" = policy rollout. ReST shows that even a simple grow+improve loop provides consistent gains over BC.

**STaR (Self-Taught Reasoner)** — Zelikman et al., 2022 (Stanford)  
In language: generate chain-of-thought reasoning, filter correct answers, retrain. Directly inspired our filtering strategy.

**GAIL (Generative Adversarial Imitation Learning)** — Ho & Ermon, 2016  
Uses a discriminator to provide dense reward from demonstrations, then optimises with RL. More powerful but far more complex. Our approach is a simpler, stable alternative that does not require adversarial training.

**HER (Hindsight Experience Replay)** — Andrychowicz et al., 2017 (OpenAI)  
Relabels failed RL trajectories with the goal the agent actually achieved. Our filtering is the BC complement: instead of relabelling failures, we just keep the successes. HER + RL is generally stronger; our approach trades performance ceiling for simplicity and stability.

**EUREKA** — Ma et al., 2023 (NVIDIA)  
Uses LLMs to generate reward functions for RL. Our work is complementary: rather than designing rewards, we bypass reward design entirely via success filtering.

### 4.3 Why Full Retraining Rather Than Fine-Tuning?

At each iteration we reinitialise the policy and retrain from scratch on the full growing dataset. The alternative — fine-tuning the previous checkpoint — risks **catastrophic forgetting**: the network may "forget" expert demos as it adapts to self-rollout data.

The cost is compute. For a small MLP on a CPU this is fine (~2 minutes per iteration). For large neural policies (transformers, diffusion models) you would instead use:
- **Elastic Weight Consolidation (EWC)**: add a regularisation term that penalises moving far from the previous weights on tasks where those weights performed well.
- **Replay buffers with expert data upweighting**: always include a fixed fraction of expert demos in each mini-batch.
- **Low-rank adaptation (LoRA)**: fine-tune only a small subset of parameters, preserving the base policy.

---

## 5. The Experiment Design

### 5.1 Goal Distribution Split

```
Full goal space: (x, y, z) positions within the FetchReach workspace
Demo region:  z > 0.45  (upper half of the goal z-range)
OOD region:   z ≤ 0.45  (lower half — never demonstrated)
```

The split on z-axis is interpretable: the expert only demonstrates reaching to goals at or above a certain height. We then ask: can the policy learn to reach goals *below* that height, purely from its own rollouts?

### 5.2 Metrics

| Metric | What it measures |
|---|---|
| Overall success rate | General policy quality |
| OOD success rate | Generalisation beyond demo distribution |
| Dataset size | How much self-generated data accumulates |
| Self-generated successes per iteration | Whether the self-improvement loop is productive |

### 5.3 Expected Results

Based on the theoretical argument and the literature:

- **Iteration 0 (BC baseline)**: ~60-80% overall, ~10-30% OOD. The policy has learned to reach demonstrated goals reasonably well, but fails on undemonstraced ones.
- **Iterations 1-3**: Rapid OOD improvement as the policy occasionally reaches OOD goals by generalisation, those successes get added, and the next iteration has OOD training signal.
- **Iterations 4-6**: Saturation — most of the reachable goal space is covered, OOD rate approaches the overall rate.

The key empirical claim: **self-improvement closes the OOD gap** that pure BC cannot close, without any additional human intervention.

---

## 6. Implementation Details and Design Decisions

### 6.1 The Policy Network

```
Input: obs (10-dim) ‖ desired_goal (3-dim)  =  13-dim
Hidden: [256, 256, 256] with LayerNorm + ReLU
Output: action (4-dim) with Tanh activation
```

**Why LayerNorm instead of BatchNorm?** With small datasets (early iterations), BatchNorm statistics are noisy. LayerNorm normalises per-sample and is more stable across batch sizes.

**Why include desired_goal in the input?** This is the standard goal-conditioned policy architecture (Kaelbling 1993, Schaul et al. 2015 "Universal Value Function Approximators"). The policy must know where to go, not just where it is. Without the goal, the policy must either guess or can only learn a single fixed behaviour.

**Why Tanh output?** FetchReach actions are in `[-1, 1]^4`. Tanh guarantees outputs stay in range without clipping. Clipping during training can create zero-gradient regions and destabilise learning.

### 6.2 The Expert

We use a P-controller: `action = clip(10 * (goal - gripper_pos), -1, 1)`. This achieves ~99% success on FetchReach. We add Gaussian noise (σ=0.05) to simulate imperfect demonstrations and increase coverage.

The gain of 10 is calibrated to the FetchReach action space: the max workspace extent is ~0.2m, so a 0.2m error maps to action magnitude 2.0 → clipped to 1.0. In practice the gripper converges within 10-15 steps out of the 50-step budget.

### 6.3 Training Objective

```
L(θ) = E_{(s,a)~D} [ ||π_θ(s) - a||² ]
```

MSE is appropriate for continuous actions under a Gaussian noise model. An alternative is **Implicit Behavioural Cloning (IBC)** (Florence et al., 2021), which learns an energy-based model `E_θ(s, a)` and uses the action that minimises energy. IBC handles multimodal action distributions better than MSE but is significantly harder to implement and optimise. For unimodal FetchReach, MSE is sufficient.

### 6.4 Cosine Annealing LR Schedule

```python
scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-6)
```

Cosine annealing provides high initial LR for fast convergence and decays to near-zero for stable final weights. Empirically outperforms constant LR and step-decay for BC on small robot datasets.

---

## 7. Limitations and What Comes Next

### 7.1 The Bootstrapping Problem

Self-improvement requires the policy to succeed *at least occasionally* to generate training signal. If the initial BC policy has 0% success on some goal region, it can never bootstrap — no successes means no new training data, so the next policy is identical.

Mitigations:
- **Goal-conditioned HER**: relabel trajectories with the goal the gripper actually reached, not the desired goal. This creates dense training signal from every rollout, even failed ones.
- **Curriculum learning**: start with easy goals (nearby), gradually expand the distribution as success rate improves.
- **Exploration bonuses**: add a small intrinsic reward for visiting novel states, encouraging the policy to explore beyond its current competence boundary.

### 7.2 No Improvement Signal from Failures

We discard failed rollouts entirely. This is wasteful — a failed trajectory still contains information about what not to do, and often reaches interesting intermediate states. 

Better approaches:
- **IRL (Inverse Reinforcement Learning)**: learn a reward function from expert demos, then use RL to optimise it. Failed rollouts provide negative signal.
- **Contrastive IL**: train a discriminator to distinguish expert trajectories from policy trajectories; use discriminator reward for RL fine-tuning (this is GAIL).
- **Hindsight filtering**: relabel a failed trajectory with the goal it actually achieved. If the gripper ended up near `g'` instead of the desired `g`, treat the trajectory as a successful demo for goal `g'` and add it.

### 7.3 Distribution Shift in Self-Generated Data

As the dataset grows, an increasing fraction is self-generated. If the policy has systematic biases — e.g., it always approaches from one direction — those biases get amplified in subsequent iterations. This is the **mode collapse** problem familiar from GAN training.

Mitigations:
- **Expert data fraction floor**: always maintain a minimum fraction of expert transitions in each training batch.
- **Diversity filter**: among self-rollout successes, prefer trajectories that are dissimilar from existing training data (measured by trajectory embedding distance).

### 7.4 Scaling to Harder Tasks

FetchReach is a low-dimensional, nearly convex problem. For harder tasks:

| Task | Additional challenges |
|---|---|
| FetchPush / FetchPickAndPlace | Contact dynamics, object state, sparser success signal |
| Dexterous hand (Shadow Hand) | 24-DOF, highly multimodal, contact-rich |
| Real robot transfer | Sim-to-real gap in dynamics, perception noise |
| Long-horizon tasks | Multi-step dependencies, credit assignment |

For dexterous manipulation, you would need:
- Larger policy (transformer or diffusion head instead of MLP)
- Demonstrations from teleoperation rather than scripted expert
- Object-centric state representation or vision-based observations
- Possibly RL fine-tuning rather than pure BC on self-rollouts

---

## 8. How This Connects to Current Research

This project sits at the intersection of three active research threads:

**Thread 1: Scalable Imitation Learning**  
The field is moving away from pure BC (which requires many demos) toward methods that leverage the environment signal. Our work is in the "minimal RL" camp: we use only a binary success signal, no reward shaping.

**Thread 2: Self-Improvement in Foundation Models**  
The success of self-play (AlphaGo, AlphaZero), STaR, and ReST in language has renewed interest in self-improvement for robotics. The question is whether the "generate → filter → retrain" loop transfers to physical embodied agents. This project is an existence proof at small scale.

**Thread 3: Data-Efficient Robot Learning**  
π0, RT-2, OpenVLA and similar foundation models are trained on millions of trajectories. But for new tasks, you still need task-specific fine-tuning data. Self-improvement is a promising way to generate that data cheaply — you only need a handful of demos to cold-start, then let the robot improve itself.

The long-term vision: a robot that is given a small number of human demos for a new task, bootstraps a policy via BC, and then autonomously improves through self-rollout until it reaches deployment quality. This project is a minimal, interpretable implementation of that vision.

---

## References

- Ross, S., Gordon, G., & Bagnell, D. (2011). **A reduction of imitation learning and structured prediction to no-regret online learning.** AISTATS.
- Andrychowicz, M., et al. (2017). **Hindsight Experience Replay.** NeurIPS. (HER)
- Ho, J., & Ermon, S. (2016). **Generative adversarial imitation learning.** NeurIPS. (GAIL)
- Gulcehre, C., et al. (2023). **Reinforced Self-Training (ReST) for Language Modeling.** arXiv. (ReST)
- Zelikman, E., et al. (2022). **STaR: Bootstrapping Reasoning With Reasoning.** NeurIPS. (STaR)
- Florence, P., et al. (2021). **Implicit Behavioral Cloning.** CoRL. (IBC)
- Ma, Y. J., et al. (2023). **EUREKA: Human-Level Reward Design via Coding Large Language Models.** arXiv. (EUREKA)
- Schaul, T., et al. (2015). **Universal Value Function Approximators.** ICML. (UVFA / goal conditioning)
- Chi, C., et al. (2023). **Diffusion Policy: Visuomotor Policy Learning via Action Diffusion.** RSS.
