import os
import glob

def cleanup_log_dir(log_dir):
    try:
        os.makedirs(log_dir)
    except OSError:
        files = glob.glob(os.path.join(log_dir, '*.monitor.csv'))
        for f in files:
            os.remove(f)


def get_latest_run_id(log_path):
    p = os.listdir(log_path)
    p = [item for item in p if os.path.isdir(log_path + '/' + item)]

    if len(p) > 0:
        p = list(map(lambda fname: int(fname.split('_')[1]), p))
        p.sort()
        id = p[-1]
    else:
        id = 0

    return id