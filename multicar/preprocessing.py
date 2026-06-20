"""Observation preprocessing — turn per-car RGB frames into the exact tensor the
trained PPO policy expects.

The single-car policy was trained with this chain (see train.py):

    GrayscaleObservation(keep_dim=True)   ->  (96, 96, 1)   uint8
    VecFrameStack(n_stack=4)              ->  (96, 96, 4)   HWC, uint8
    VecTransposeImage (auto by SB3)       ->  (4, 96, 96)   CHW, uint8

So at inference time, *each car* must reproduce that chain independently before
its frame can be fed to ``model.predict``. This is the one piece whose bugs are
silent — get it wrong and the CNN runs on garbage, producing a car that merely
wanders, with no exception to point at. That is why it lives in its own module
and is the one component with unit tests.

Critical invariants (each is a real, easy-to-make mistake):
  * grayscale weights are gymnasium's exact (0.2125, 0.7154, 0.0721)
  * output is channel-first ``(n_stack, H, W)`` — mirrors VecTransposeImage
  * dtype stays uint8 in [0, 255] — do NOT divide by 255 (SB3's CnnPolicy
    normalizes internally; pre-dividing halves the input twice)
  * the newest frame is the LAST in the stack (matches SB3 VecFrameStack's roll)
  * each car keeps an independent rolling buffer; frames never leak between cars
"""

import numpy as np

# Exactly gymnasium.wrappers.GrayscaleObservation's luminosity weights.
# (They sum to 1.0, so a solid-grey RGB pixel maps to its own value.)
_GRAY_WEIGHTS = np.array([0.2125, 0.7154, 0.0721])


def to_grayscale(rgb):
    """``(H, W, 3)`` uint8 RGB -> ``(H, W)`` uint8 grayscale.

    Byte-identical to ``gymnasium.wrappers.GrayscaleObservation(keep_dim=False)``.
    """
    return np.sum(np.multiply(rgb, _GRAY_WEIGHTS), axis=-1).astype(np.uint8)


class ObsPreprocessor:
    """Per-car rolling grayscale frame-stacker producing model-ready batches.

    Usage::

        pre = ObsPreprocessor(num_agents=N, n_stack=4)
        batch = pre.reset(obs)              # obs: (N,96,96,3) or (96,96,3) if N==1
        action, _ = model.predict(batch)    # batch: (N, 4, 96, 96) uint8
        obs, *_ = env.step(action)
        batch = pre.observe(obs)            # roll the new frame(s) into the stack
    """

    def __init__(self, num_agents, n_stack=4):
        if num_agents < 1:
            raise ValueError("num_agents must be >= 1")
        if n_stack < 1:
            raise ValueError("n_stack must be >= 1")
        self.num_agents = num_agents
        self.n_stack = n_stack
        self._buffers = None  # list of (H, W, n_stack) uint8, one per car

    @staticmethod
    def _split_cars(obs):
        """Normalize env obs into a list of ``(H, W, 3)`` frames, one per car.

        Accepts a stacked ``(N, H, W, 3)`` array or a single ``(H, W, 3)`` frame
        (the shape the env returns when ``num_agents == 1``).
        """
        obs = np.asarray(obs)
        if obs.ndim == 3:
            return [obs]
        if obs.ndim == 4:
            return [obs[i] for i in range(obs.shape[0])]
        raise ValueError(
            f"expected obs of shape (H,W,3) or (N,H,W,3), got {obs.shape}"
        )

    def reset(self, obs):
        """Prime each car's buffer by repeating its first frame ``n_stack`` times.

        This matches VecFrameStack's cold-start, where the initial stack is the
        first observation duplicated across every slot.
        """
        frames = self._split_cars(obs)
        if len(frames) != self.num_agents:
            raise ValueError(
                f"expected {self.num_agents} car frames, got {len(frames)}"
            )
        self._buffers = [
            np.repeat(to_grayscale(f)[:, :, None], self.n_stack, axis=2)
            for f in frames
        ]
        return self.batch()

    def observe(self, obs):
        """Roll one new frame into each car's buffer (drop oldest, append newest)."""
        if self._buffers is None:
            raise RuntimeError("call reset() before observe()")
        frames = self._split_cars(obs)
        if len(frames) != self.num_agents:
            raise ValueError(
                f"expected {self.num_agents} car frames, got {len(frames)}"
            )
        for i, f in enumerate(frames):
            gray = to_grayscale(f)[:, :, None]
            self._buffers[i] = np.concatenate(
                [self._buffers[i][:, :, 1:], gray], axis=2
            )
        return self.batch()

    def batch(self):
        """Return the model-ready batch: ``(num_agents, n_stack, H, W)`` uint8 (CHW)."""
        return np.stack(
            [np.transpose(b, (2, 0, 1)) for b in self._buffers], axis=0
        ).astype(np.uint8)
