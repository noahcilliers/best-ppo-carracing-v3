import numpy as np
import torch as th
from gymnasium import Env, spaces
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage

from telemetry_env import (
    SPEED_SCALE,
    STEER_ANGLE_SCALE,
    TELEMETRY_DIM,
    TelemetryCombinedExtractor,
    TelemetryObs,
    VecTelemetryDict,
    WHEEL_OMEGA_SCALE,
    YAW_RATE_SCALE,
    read_telemetry,
)


class _Joint:
    def __init__(self, angle):
        self.angle = angle


class _Wheel:
    def __init__(self, omega, angle=0.0):
        self.omega = omega
        self.joint = _Joint(angle)


class _Hull:
    def __init__(self, velocity, angular_velocity):
        self.linearVelocity = velocity
        self.angularVelocity = angular_velocity


class _Car:
    def __init__(self, velocity=(0.0, 0.0), angular_velocity=0.0, wheels=None):
        self.hull = _Hull(velocity, angular_velocity)
        self.wheels = wheels if wheels is not None else [_Wheel(0.0) for _ in range(4)]


class _TelemetrySource:
    @property
    def unwrapped(self):
        return self


def test_read_telemetry_returns_zero_without_car_or_hull():
    env = _TelemetrySource()
    np.testing.assert_array_equal(read_telemetry(env), np.zeros(TELEMETRY_DIM, dtype=np.float32))

    env.car = _Car()
    env.car.hull = None
    np.testing.assert_array_equal(read_telemetry(env), np.zeros(TELEMETRY_DIM, dtype=np.float32))


def test_read_telemetry_normalizes_and_clips():
    env = _TelemetrySource()
    env.car = _Car(
        velocity=(SPEED_SCALE * 0.3, SPEED_SCALE * 0.4),
        angular_velocity=-YAW_RATE_SCALE * 2.0,
        wheels=[
            _Wheel(WHEEL_OMEGA_SCALE * 0.25, STEER_ANGLE_SCALE * 0.5),
            _Wheel(-WHEEL_OMEGA_SCALE * 0.5),
            _Wheel(WHEEL_OMEGA_SCALE * 2.0),
            _Wheel(-WHEEL_OMEGA_SCALE * 2.0),
        ],
    )

    telemetry = read_telemetry(env)

    assert telemetry.dtype == np.float32
    np.testing.assert_allclose(
        telemetry,
        np.array([0.5, 0.25, -0.5, 1.0, -1.0, 0.5, -1.0, 0.0], dtype=np.float32),
    )


class TinyTelemetryEnv(Env):
    def __init__(self):
        self.action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(0, 255, shape=(96, 96, 1), dtype=np.uint8)
        self.car = _Car()
        self._step = 0

    def reset(self, **kwargs):
        self._step = 0
        self.car = _Car(velocity=(SPEED_SCALE * 0.5, 0.0))
        return np.zeros(self.observation_space.shape, dtype=np.uint8), {}

    def step(self, action):
        self._step += 1
        self.car = _Car(
            velocity=(SPEED_SCALE, 0.0),
            angular_velocity=YAW_RATE_SCALE * 0.25,
        )
        obs = np.full(self.observation_space.shape, self._step, dtype=np.uint8)
        return obs, 0.0, False, False, {}


def test_vec_telemetry_dict_merges_after_frame_stack():
    env = DummyVecEnv([lambda: TelemetryObs(TinyTelemetryEnv())])
    env = VecFrameStack(env, n_stack=4)
    env = VecTelemetryDict(env)
    try:
        obs = env.reset()
        assert set(obs) == {"img", "telemetry"}
        assert obs["img"].shape == (1, 96, 96, 4)
        assert obs["img"].dtype == np.uint8
        assert obs["telemetry"].shape == (1, TELEMETRY_DIM)
        np.testing.assert_allclose(obs["telemetry"][0, 0], 0.5)

        obs, rewards, dones, infos = env.step(np.zeros((1, 3), dtype=np.float32))
        assert obs["img"].shape == (1, 96, 96, 4)
        assert obs["telemetry"].shape == (1, TELEMETRY_DIM)
        np.testing.assert_allclose(obs["telemetry"][0, [0, 6]], [1.0, 0.25])
        assert rewards.shape == (1,)
        assert dones.shape == (1,)
        assert len(infos) == 1
    finally:
        env.close()


def test_transposed_dict_obs_feeds_telemetry_extractor():
    env = DummyVecEnv([lambda: TelemetryObs(TinyTelemetryEnv())])
    env = VecFrameStack(env, n_stack=4)
    env = VecTelemetryDict(env)
    env = VecTransposeImage(env)
    try:
        obs = env.reset()
        assert obs["img"].shape == (1, 4, 96, 96)
        assert obs["telemetry"].shape == (1, TELEMETRY_DIM)

        extractor = TelemetryCombinedExtractor(
            env.observation_space,
            cnn_output_dim=16,
            telemetry_features_dim=4,
        )
        features = extractor(
            {
                "img": th.as_tensor(obs["img"]).float(),
                "telemetry": th.as_tensor(obs["telemetry"]).float(),
            }
        )
        assert features.shape == (1, 20)
    finally:
        env.close()
