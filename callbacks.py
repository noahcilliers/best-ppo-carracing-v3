"""
Tail-aware checkpoint selection for training.

Phase A of the variance-reduction plan. SB3's default EvalCallback saves the best
model by mean reward over a few episodes -- which, at std ~191, easily saves a
LUCKY checkpoint. This callback instead evaluates on a FIXED set of seeds and
saves the best by a tail-aware score (default `mean - 0.5*std`), so we keep the
most RELIABLE checkpoint, not the luckiest one.

It also avoids SB3's VecNormalize sync path (we don't normalize obs), so it
sidesteps the eval-env type-mismatch issue entirely.
"""

import os
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class TailAwareEvalCallback(BaseCallback):
    def __init__(self, eval_env, seeds, eval_freq, save_path,
                 score_fn=None, verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.seeds = list(seeds)
        self.eval_freq = eval_freq          # in vec-steps (total_steps // n_envs)
        self.save_path = save_path
        # Default: reward reliability, not just average.
        self.score_fn = score_fn or (lambda r: float(np.mean(r) - 0.5 * np.std(r)))
        self.best_score = -np.inf

    def _init_callback(self):
        os.makedirs(self.save_path, exist_ok=True)

    def _evaluate(self):
        rewards = []
        for s in self.seeds:
            self.eval_env.seed(s)
            obs = self.eval_env.reset()
            done = np.array([False])
            total = 0.0
            while not done[0]:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, r, done, _ = self.eval_env.step(action)
                total += float(r[0])
            rewards.append(total)
        return np.array(rewards)

    def _on_step(self):
        if self.eval_freq <= 0 or self.n_calls % self.eval_freq != 0:
            return True

        rewards = self._evaluate()
        score = self.score_fn(rewards)

        self.logger.record("eval/mean_reward", float(rewards.mean()))
        self.logger.record("eval/std_reward", float(rewards.std()))
        self.logger.record("eval/min_reward", float(rewards.min()))
        self.logger.record("eval/p10_reward", float(np.percentile(rewards, 10)))
        self.logger.record("eval/score", score)

        if score > self.best_score:
            self.best_score = score
            self.model.save(os.path.join(self.save_path, "best_model"))
            if self.verbose:
                print(f"[eval] new best: score={score:.1f} "
                      f"mean={rewards.mean():.1f} std={rewards.std():.1f} "
                      f"min={rewards.min():.1f} -> saved best_model")
        return True
