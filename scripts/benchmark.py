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

# Need a global plan results to save partial results when interrupted
cmd_args = None
plan_results = None

test_args_template = '--type {typ} -m {M} -n {N} --nnz {nnz} --seed {seed}'


def tprint(*args, **kwargs):
    print('[' + str(datetime.now()) + ']', *args, **kwargs)

# choose solvers (single directory compiled code)

# test configuration
# - name - identifier 
# - type - lasso, lp, etc
# - N length - 
# - M length
# - density %


# specify which solvers run which tests

# correlate results for same tests for easy graphing

# output
# - norms
# - total time
# - iteration time
# - iteration #
# - time for sub sections

def parse_test_spec(file):
    spec = json.load(file)
    if not isinstance(spec, dict):
        return 'Spec root must be an object'
    for solver in spec['solvers']:
        solver['directory'] = os.path.expanduser(solver['directory'])
    return spec


def process_test_spec(spec):
    plan = defaultdict(list)
    for solver in spec['solvers']:
        task = {
            'name': solver['name'],
            'run': solver['run'],
            'directory': solver['directory'],
        }
        for test in solver['tests']:
            plan[test].append(task)
    return plan


def make_tests(solvers):
    for solver in solvers:
        p = subprocess.Popen(['make', 'test'], cwd=solver['directory'])
        p.wait()


def parse_solver_output(output, error):
    if error is not None:
        pass
    rdc = lambda: defaultdict(rdc)
    result = defaultdict(rdc)
    for line in output.splitlines():
        kv = [x.strip() for x in line.split(':')]
        if len(kv) != 2:
            tprint('Error parsing task output, offending line follows')
            tprint(line)
            return {
                'out': output,
                'err': error
            }
        ks = [k.strip() for k in kv[0].split(',')]
        v = kv[1]
        temp = result
        for k in ks[:-1]:
            temp = temp[k]
        temp[ks[-1]] = v
    return result


def run_plan(plan, test_settings):
    global plan_results

    tprint('Starting test plan')
    plan_results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for test_name, tests in plan.iteritems():
        settings = test_settings[test_name]
        typ = settings['type']
        trials = int(settings['trials'])
        tprint('Starting test', test_name)
        for param in settings['params']:
            M = param['M']
            N = param['N']
            nnz = int(int(M) * int(N) * float(param['density']))
            tprint('Starting trials')
            for trial in range(trials):
                seed = 1000 + trial
                args = test_args_template.format(typ=typ, M=M, N=N, nnz=nnz,
                                                 seed=seed)
                tprint('Starting tasks for args:', args)
                for task in tests:
                    cmd = task['run'] + ' ' + args
                    tprint('Running task with run cmd:', cmd)
                    p = subprocess.Popen(
                        cmd,
                        cwd=task['directory'],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        shell=True
                    )
                    tout = 60 * 60
                    try:
                        sout, serr = p.communicate(timeout=tout)
                    except subprocess.TimeoutExpired:
                        p.kill()
                        tprint('Task timed out after', tout, 'seconds')
                        sout, serr = p.communicate()
                    if p.returncode == 0:
                        tprint('Finished task')
                        result = parse_solver_output(sout, serr)
                    else:
                        tprint('Error in task, return code:',
                               str(p.returncode))
                        result = {
                            'out': sout,
                            'err': serr
                        }
                    task_entry = plan_results[test_name][task['name']]
                    task_entry[str(trial)].append(result)
                    task_entry['M'] = M
                    task_entry['N'] = N
                    task_entry['nnz'] = nnz
                    task_entry['cmd'] = cmd
                tprint('Finished tasks')
            tprint('Finished trials')
        tprint('Finished test')
    tprint('Finished test plan')
    return plan_results


def save_test_results(results, out_fo):
    json.dump(results, out_fo)


def generate_test_graphs(results):
    pass


def main(args):
    start_time = time.time()
    tprint('Parsing test spec')
    with open(args.spec_file, 'r') as spec_fo:
        spec = parse_test_spec(spec_fo)
    tprint('Processing test spec')
    plan = process_test_spec(spec)
    tprint('Building test code')
    make_tests(spec['solvers'])
    test_results = run_plan(plan, spec['tests'])
    tprint('Saving results to', args.results_file)
    with open(args.results_file, 'w') as results_fo:
        save_test_results(test_results, results_fo)
    generate_test_graphs(test_results)
    tprint('Script took', time.time() - start_time, 'seconds')


def sigint_handler(signal, frame):
    tprint('Sigint received, exiting')
    if plan_results is not None:
        tprint('Saving partial results')
        with open(cmd_args.results_file, 'w') as results_fo:
            save_test_results(plan_results, results_fo)
    sys.exit(0)


if __name__ == '__main__':
    signal.signal(signal.SIGINT, sigint_handler)

    parser = argparse.ArgumentParser()
    parser.add_argument('spec_file', nargs='?', default='spec.json')
    parser.add_argument('results_file', nargs='?', default='results.json')
    cmd_args = parser.parse_args()
    main(cmd_args)
