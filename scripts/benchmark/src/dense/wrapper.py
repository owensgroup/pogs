from __future__ import print_function
import os
import sys
import subprocess32 as subprocess
import argparse
import json


# Load schedule, figure out what node this is,
# set CUDA_VISIBLE_DEVICES to appropriate GPUs
# launch solver, maintaining stdout and stderr
def extract_gpu_indices(schedule_file, rank):
    schedule = json.load(schedule_file)
    if not isinstance(schedule, dict):
        return 'Config root must be an object'
    return schedule['info'][rank]['gpu_indices']


def main(argv, cmd_args):
    rank = int(os.environ['MV2_COMM_WORLD_RANK'])
    with open(cmd_args.schedule, 'r') as schedule_fo:
        gpu_indices = extract_gpu_indices(schedule_fo, rank)
    gpu_indices = ','.join(map(str, gpu_indices))
    cuda_env = os.environ.copy()
    cuda_env['CUDA_VISIBLE_DEVICES'] = gpu_indices
    cmd = ' '.join(argv[1:])
    subprocess.Popen(cmd, shell=True, env=cuda_env)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--schedule', nargs='?', default='spec.json')
    cmd_args, uk = parser.parse_known_args()
    main(sys.argv, cmd_args)
