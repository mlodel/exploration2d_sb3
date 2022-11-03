from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

class LogInfoCallback(BaseCallback):
    """
    A custom callback that logs data from the info dict
    """

    def __init__(self, n_envs: int, n_steps: int, info_keys: list=None):
        super().__init__()

        # Store settings
        self.n_envs = n_envs
        self.n_steps = n_steps

        # Set info keys to log
        self.info_keys = [
            "ran_out_of_time",
            "deadlocked",
        ]

        # Create info rollout buffer
        self.info_buffer = {key: np.zeros((n_steps, n_envs)) for key in self.info_keys}

        self.step = 0

    def _on_step(self) -> None:
        # Fetch info dict
        infos = self.locals["infos"]
        # dones = self.locals["dones"]

        # Add info to buffer
        for key in self.info_keys:
            self.info_buffer[key][self.step, :] = np.array([info[key] for info in infos], dtype=np.float32)

        # Increment step
        self.step += 1

    def _on_rollout_end(self) -> None:
        # Log info
        for key in self.info_keys:
            self.logger.record("rollout/" + key, np.mean(np.sum(self.info_buffer[key], axis=0)))

        # Reset step
        self.step = 0