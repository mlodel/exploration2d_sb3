# Create EventCallback that is called after eval in EvalCallback
# should call env.close_video_recorder() and save video to disk

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecVideoRecorder


class StoreVideoCallback(BaseCallback):
    def __init__(self, eval_env: VecVideoRecorder):
        super(StoreVideoCallback, self).__init__()
        self.eval_env = eval_env

    def _on_step(self) -> bool:
        print(f"Saving video to {self.eval_env.video_recorder.path}")
        self.eval_env.close_video_recorder()
        return True
