"""
Phase B reward shaping for CarRacing (every term defaults to OFF, so the clean
baseline is reproducible). Bundled "stability + control" knobs aimed at the
off-track / oscillation tail that drives the 657 +/- 233 variance:

  * On-track speed reward    k_speed        -> +k_speed * (wheels_on_road/4) * speed
        Phase A "go faster" term: the base game only pays the -0.1/step time
        penalty for dawdling, which the 868 model underweights (several eval
        seeds run the full 1000 steps without finishing). A direct on-track
        speed bonus pushes higher cornering/straight-line velocity. Gated on
        wheels-on-road so it rewards going fast ON the track, never grass-skating
        -- the exact mirror of the k_grass_speed penalty below.
  * Flat grass penalty       k_grass        -> -k_grass * wheels_on_grass
  * Velocity-linked grass    k_grass_speed  -> -k_grass_speed * (wheels/4) * speed
        Punishes HIGH-SPEED corner-cutting hard, low-speed recovery lightly, so
        the car doesn't become timid (Gemini report's idea).
  * Action-smoothness        k_smooth       -> -k_smooth * sum((action - prev_action)^2)
        Curbs steering oscillation / overcorrection (all three reports agree);
        the indispensable partner to k_speed -- it keeps "faster" from becoming
        "twitchier". Squared (L2) delta matches the multi-car implementation that
        genuinely helped: it punishes violent jerks hard while leaving fine
        corrections almost free (an L1 penalty taxes every small adjustment).
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
    def __init__(self, env, k_grass=0.0, k_grass_speed=0.0, k_speed=0.0,
                 k_smooth=0.0,
                 grass_terminate_steps=0, grass_terminate_penalty=0.0):
        super().__init__(env)
        self.k_grass = float(k_grass)
        self.k_grass_speed = float(k_grass_speed)
        self.k_speed = float(k_speed)
        self.k_smooth = float(k_smooth)
        self.grass_terminate_steps = int(grass_terminate_steps)
        self.grass_terminate_penalty = float(grass_terminate_penalty)
        self._prev_action = None
        self._grass_steps = 0

    def reset(self, **kwargs):
        self._prev_action = None
        self._grass_steps = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        car = getattr(self.unwrapped, "car", None)

        if car is not None:
            on_grass = sum(1 for w in car.wheels if len(w.tiles) == 0)
            v = car.hull.linearVelocity
            speed = math.hypot(v[0], v[1])
            info["grass_wheels"] = on_grass
            info["speed"] = speed

            if self.k_speed > 0.0:
                n_wheels = len(car.wheels)
                on_road = n_wheels - on_grass
                reward += self.k_speed * (on_road / n_wheels) * speed

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


def make_carracing(k_grass=0.0, k_grass_speed=0.0, k_speed=0.0, k_smooth=0.0,
                   grass_terminate_steps=0, grass_terminate_penalty=0.0,
                   render_mode=None):
    """Factory: CarRacing-v3 + (optional) reward shaping + grayscale (keep_dim)."""
    def _init():
        env = gym.make("CarRacing-v3", render_mode=render_mode)
        env = RewardShapingWrapper(
            env,
            k_grass=k_grass,
            k_grass_speed=k_grass_speed,
            k_speed=k_speed,
            k_smooth=k_smooth,
            grass_terminate_steps=grass_terminate_steps,
            grass_terminate_penalty=grass_terminate_penalty,
        )
        env = gym.wrappers.GrayscaleObservation(env, keep_dim=True)
        return env
    return _init
