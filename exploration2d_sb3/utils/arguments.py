import argparse

def get_args():
    parser = argparse.ArgumentParser(description='RL')

    parser.add_argument(
        '--resume',
        action='store_true',
        default=False,
        help='resume training from stored model')
    parser.add_argument(
        '--resume_run_id',
        type=str,
        default='akoq2gam',
        help='which log dir id to resume (default: 0 uses last run)')

    args = parser.parse_args()

    return args
