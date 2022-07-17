import os
import glob
import wandb


def cleanup_log_dir(log_dir):
    try:
        os.makedirs(log_dir)
    except OSError:
        files = glob.glob(os.path.join(log_dir, "*.monitor.csv"))
        for f in files:
            os.remove(f)


def get_latest_run_id(log_path):
    p = os.listdir(log_path)
    p = [item for item in p if os.path.isdir(log_path + "/" + item)]

    if len(p) > 0:
        p = list(map(lambda fname: int(fname.split("_")[1]), p))
        p.sort()
        id = p[-1]
    else:
        id = 0

    return id


def get_save_path(resume=False, run: wandb.run = None):
    log_dir = os.getcwd() + "/logs"
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    if not resume:
        log_id = get_latest_run_id(log_dir) + 1
        save_path = os.path.join(log_dir, "log_{}".format(log_id))
        cleanup_log_dir(save_path)
    else:
        save_path = run.config.get("tensorboard_log")

    print("Log path: {}".format(save_path))

    return save_path
