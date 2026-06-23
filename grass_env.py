"""
Reward shaping for CarRacing (every term defaults to OFF, so the clean baseline is
reproducible).

SPEED terms (Phase A / "go faster", added after the instantaneous-speed reward was
reverted -- see docs/PROJECT_RECORD.md for why that one failed):

  * Dense time cost          k_time         -> -k_time  (every step)
        A stronger per-step time penalty ON TOP of the base game's -0.1/step. It
        integrates over the episode to -k_time * total_steps, i.e. it IS a total
        lap-time penalty -- but paid densely, so it actually trains (a sparse
        terminal lap-time bonus is crushed by gamma^~800 discounting). The ONLY
        way to score better is to finish the lap in fewer steps => drive faster.
        Net per-step reward stays NEGATIVE (the rule the reverted speed bonus
        broke), so the agent always prefers the episode to end sooner.
  * Progress-rate reward      k_progress     -> +k_progress * new_tiles_this_step
        Pays for forward progress (each newly-visited track tile). Farm-proof:
        only NEW tiles count and a lap has finitely many, so circling/dawdling
        earns nothing and it telescopes to a constant per lap. Pairs with k_time:
        time pushes for speed, progress keeps the car ON the racing line
        completing tiles instead of cutting corners to save time.

STABILITY / CONTROL terms (Phase B bundle that cut the 657 +/- 233 variance):

  * Flat grass penalty       k_grass        -> -k_grass * wheels_on_grass
  * Velocity-linked grass    k_grass_speed  -> -k_grass_speed * (wheels/4) * speed
        Punishes HIGH-SPEED corner-cutting hard, low-speed recovery lightly, so
        the car doesn't become timid (Gemini report's idea).
  * Action-smoothness        k_smooth       -> -k_smooth * sum((action - prev_action)^2)
        Curbs steering oscillation / overcorrection (all three reports agree).
        Squared (L2) delta matches the multi-car implementation that genuinely
        helped: it punishes violent jerks hard while leaving fine corrections
        almost free (an L1 penalty taxes every small adjustment).
  * Early-termination        grass_terminate_steps / grass_terminate_penalty
        Ends unrecoverable off-track spirals (>=3 wheels on grass for N steps).

Detection is exact: each Box2D wheel tracks the road tiles it touches; an empty
tile set means that wheel is on grass. Speed is the hull velocity magnitude.

Eval/inference can leave every term at 0 to get the RAW game score.
"""

import math
import numpy as np
import gymnasium as gym


class RewardShapingWrapper(gym.Wrapper):
    def __init__(self, env, k_grass=0.0, k_grass_speed=0.0, k_smooth=0.0,
                 k_time=0.0, k_progress=0.0,
                 grass_terminate_steps=0, grass_terminate_penalty=0.0):
        super().__init__(env)
        self.k_grass = float(k_grass)
        self.k_grass_speed = float(k_grass_speed)
        self.k_smooth = float(k_smooth)
        self.k_time = float(k_time)
        self.k_progress = float(k_progress)
        self.grass_terminate_steps = int(grass_terminate_steps)
        self.grass_terminate_penalty = float(grass_terminate_penalty)
        self._prev_action = None
        self._grass_steps = 0
        self._prev_tiles = 0

    def reset(self, **kwargs):
        self._prev_action = None
        self._grass_steps = 0
        self._prev_tiles = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Dense time cost: stronger per-step penalty on top of the base -0.1/step.
        # Sums to -k_time * total_steps over the episode = a total lap-time penalty,
        # paid densely so it trains. Unconditional (time passes on or off track).
        if self.k_time > 0.0:
            reward -= self.k_time

        # Progress-rate reward: pay for each newly-visited tile this step. Only NEW
        # tiles count, so it can't be farmed by dawdling/circling.
        if self.k_progress > 0.0:
            tiles = getattr(self.unwrapped, "tile_visited_count", self._prev_tiles)
            reward += self.k_progress * max(0, tiles - self._prev_tiles)
            self._prev_tiles = tiles

        car = getattr(self.unwrapped, "car", None)

        if car is not None:
            on_grass = sum(1 for w in car.wheels if len(w.tiles) == 0)
            v = car.hull.linearVelocity
            speed = math.hypot(v[0], v[1])
            info["grass_wheels"] = on_grass
            info["speed"] = speed

            if on_grass:
                if self.k_grass > 0.0:
                    reward -= self.k_grass * on_grass
                if self.k_grass_speed > 0.0:
                    reward -= self.k_grass_speed * (on_grass / 4.0) * speed

            if self.grass_terminate_steps > 0:
                self._grass_steps = self._grass_steps + 1 if on_grass >= 3 else 0
                if self._grass_steps >= self.grass_terminate_steps:
                    reward -= self.grass_terminate_penalty
                    terminated = True
                    info["terminal_reason"] = "sustained_grass"

        if self.k_smooth > 0.0:
            a = np.asarray(action, dtype=np.float32)
            if self._prev_action is not None:
                delta = a - self._prev_action
                # Squared (L2) delta, matching the multi-car k_smooth that worked:
                # quadratic in jerk, so violent swings hurt far more than fine ones.
                reward -= self.k_smooth * float(np.square(delta).sum())
            self._prev_action = a

        return obs, reward, terminated, truncated, info


def make_carracing(k_grass=0.0, k_grass_speed=0.0, k_smooth=0.0,
                   k_time=0.0, k_progress=0.0,
                   grass_terminate_steps=0, grass_terminate_penalty=0.0,
                   render_mode=None):
    """Factory: CarRacing-v3 + (optional) reward shaping + grayscale (keep_dim)."""
    def _init():
        env = gym.make("CarRacing-v3", render_mode=render_mode)
        env = RewardShapingWrapper(
            env,
            k_grass=k_grass,
            k_grass_speed=k_grass_speed,
            k_smooth=k_smooth,
            k_time=k_time,
            k_progress=k_progress,
            grass_terminate_steps=grass_terminate_steps,
            grass_terminate_penalty=grass_terminate_penalty,
        )
        env = gym.wrappers.GrayscaleObservation(env, keep_dim=True)
        return env
    return _init
