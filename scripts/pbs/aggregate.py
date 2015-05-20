from __future__ import print_function
from collections import defaultdict
from datetime import datetime

import time
import os
import sys
import signal
import json
import subprocess32 as subprocess
import argparse

# 1. test type ('lasso_mn')
# 2. task type ('pogs_row_2')
# 3. trial # ('4')
# 4. array of increasing problem sizes
# ->
# 1. test type
# 2. task type ('pogs_row_2')
# 3. array of increasing problem sizes (avg over trials)


def sum_problem(pl, pr):
    for k in pl.keys():
        if not k == 'iter':
            pl[k] = pl[k] + pr[k]
    return pl


def average_trials(trials):
    avg_probs = trials[str(0)]
    for i in range(1, len(trials)):
        trial = trials[str(i)]
        for index, problem in enumerate(trial):
            avg_probs[index] = sum_problem(avg_probs[index], problem)
    for i in len(avg_probs):
        avg_probs[i]


def combine_results(rl, rr):
    for k in rr:
        if k in rl:
            rl[k].update(rr[k])
        else:
            rl[k] = rr[k]
    return rl


def main(args):
    final_results = {}
    for file in args.result:
        results = None
        with open(file, 'r') as r:
            results = json.load(r)
        final_results = combine_results(final_results, results)
    with open(args.o, 'w') as out_fo:
        json.dump(final_results, out_fo)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('result', nargs='+')
    parser.add_argument('-o')
    cmd_args = parser.parse_args()
    main(cmd_args)
