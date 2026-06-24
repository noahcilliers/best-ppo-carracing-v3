"""Telemetry observation helpers for Phase C CarRacing experiments.

The image pipeline stays unchanged through grayscale + frame-stack. Telemetry is
captured per single env, then merged into a Dict observation after VecFrameStack
so only the image is stacked.
"""

from __future__ import annotations

import math

import gymnasium as gym
import numpy as np
import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, NatureCNN
from stable_baselines3.common.vec_env import VecEnv, VecEnvWrapper


TELEMETRY_DIM = 8
SPEED_SCALE = 80.0
WHEEL_OMEGA_SCALE = 120.0
STEER_ANGLE_SCALE = 0.6
YAW_RATE_SCALE = 5.0
TELEMETRY_SPACE = spaces.Box(-1.0, 1.0, shape=(TELEMETRY_DIM,), dtype=np.float32)


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _vec2_length(vec) -> float:
    if vec is None:
        return 0.0
    try:
        x, y = vec[0], vec[1]
    except (TypeError, IndexError):
        x = getattr(vec, "x", 0.0)
        y = getattr(vec, "y", 0.0)
    return math.hypot(_safe_float(x), _safe_float(y))


def read_telemetry(env) -> np.ndarray:
    """Read and normalize current CarRacing vehicle telemetry.

    Output order:
    [speed, wheel0 omega, wheel1 omega, wheel2 omega, wheel3 omega,
     steer angle, yaw rate, reserved]
    """
    root = getattr(env, "unwrapped", env)
    car = getattr(root, "car", None)
    if car is None:
        return np.zeros(TELEMETRY_DIM, dtype=np.float32)

    hull = getattr(car, "hull", None)
    if hull is None:
        return np.zeros(TELEMETRY_DIM, dtype=np.float32)

    wheels = list(getattr(car, "wheels", []) or [])
    telemetry = np.zeros(TELEMETRY_DIM, dtype=np.float32)
    telemetry[0] = _vec2_length(getattr(hull, "linearVelocity", None)) / SPEED_SCALE

    for idx in range(min(4, len(wheels))):
        telemetry[1 + idx] = _safe_float(getattr(wheels[idx], "omega", 0.0)) / WHEEL_OMEGA_SCALE

    if wheels:
        joint = getattr(wheels[0], "joint", None)
        telemetry[5] = _safe_float(getattr(joint, "angle", 0.0)) / STEER_ANGLE_SCALE
    telemetry[6] = _safe_float(getattr(hull, "angularVelocity", 0.0)) / YAW_RATE_SCALE
    return np.clip(telemetry, -1.0, 1.0).astype(np.float32)


class TelemetryObs(gym.Wrapper):
    """Snapshot telemetry while leaving the underlying image observation intact."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.telemetry = np.zeros(TELEMETRY_DIM, dtype=np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.telemetry = read_telemetry(self.env)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.telemetry = read_telemetry(self.env)
        if terminated or truncated:
            # The vec env auto-resets on episode end BEFORE VecTelemetryDict can
            # read telemetry via get_attr, so the terminal-step telemetry would be
            # lost. Stash it on the info dict to pair with `terminal_observation`.
            info["terminal_telemetry"] = self.telemetry.copy()
        return obs, reward, terminated, truncated, info


class VecTelemetryDict(VecEnvWrapper):
    """Merge stacked image observations with current-frame telemetry."""

    def __init__(self, venv: VecEnv, img_key: str = "img", telemetry_key: str = "telemetry"):
        self.img_key = img_key
        self.telemetry_key = telemetry_key
        observation_space = spaces.Dict(
            {
                self.img_key: venv.observation_space,
                self.telemetry_key: TELEMETRY_SPACE,
            }
        )
        super().__init__(venv, observation_space=observation_space)

    def reset(self):
        obs = self.venv.reset()
        return self._merge(obs)

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        merged = self._merge(obs)
        for info in infos:
            terminal_obs = info.get("terminal_observation")
            if terminal_obs is None:
                continue
            # SB3 runs `terminal_observation` through the policy too, so it must be
            # the SAME Dict the policy expects. Without this it stays a bare image
            # array and obs_to_tensor crashes in is_vectorized_dict_observation.
            terminal_telemetry = info.pop("terminal_telemetry", None)
            if terminal_telemetry is None:
                terminal_telemetry = np.zeros(TELEMETRY_DIM, dtype=np.float32)
            info["terminal_observation"] = {
                self.img_key: terminal_obs,
                self.telemetry_key: np.asarray(terminal_telemetry, dtype=np.float32),
            }
        return merged, rewards, dones, infos

    def step_async(self, actions) -> None:
        self.venv.step_async(actions)

    def _merge(self, img_obs):
        telemetry = np.asarray(self.venv.get_attr("telemetry"), dtype=np.float32)
        if telemetry.shape != (self.num_envs, TELEMETRY_DIM):
            telemetry = telemetry.reshape((self.num_envs, TELEMETRY_DIM))
        return {self.img_key: img_obs, self.telemetry_key: telemetry}


class TelemetryCombinedExtractor(BaseFeaturesExtractor):
    """NatureCNN for image frames plus a small MLP for normalized telemetry."""

    def __init__(
        self,
        observation_space: spaces.Dict,
        cnn_output_dim: int = 256,
        telemetry_features_dim: int = 32,
        normalized_image: bool = False,
    ):
        super().__init__(observation_space, features_dim=cnn_output_dim + telemetry_features_dim)
        img_space = observation_space.spaces["img"]
        telemetry_space = observation_space.spaces["telemetry"]
        telemetry_dim = int(np.prod(telemetry_space.shape))

        self.img_extractor = NatureCNN(
            img_space,
            features_dim=cnn_output_dim,
            normalized_image=normalized_image,
        )
        self.telemetry_extractor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(telemetry_dim, 64),
            nn.GELU(),
            nn.Linear(64, telemetry_features_dim),
            nn.GELU(),
        )

    def forward(self, observations: dict[str, th.Tensor]) -> th.Tensor:
        img_features = self.img_extractor(observations["img"])
        telemetry_features = self.telemetry_extractor(observations["telemetry"])
        return th.cat((img_features, telemetry_features), dim=1)
