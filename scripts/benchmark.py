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


def tprint(*args, **kwargs):
    print('[' + str(datetime.now()) + ']', *args, **kwargs)

# output
# - norms
# - total time
# - iteration time
# - iteration #
# - time for sub sections


def parse_config_spec(file):
    spec = json.load(file)
    if not isinstance(spec, dict):
        return 'Config root must be an object'
    for solver_name, solver in spec['solvers'].iteritems():
        solver['directory'] = os.path.expanduser(solver['directory'])
    return spec


def process_config_spec(spec):
    # plan = defaultdict(list)
    # # Process solvers
    # for solver in spec['solvers']:
    #     task = {
    #         'name': solver['name'],
    #         'run': solver['run'],
    #         'directory': solver['directory'],
    #     }
    #     for test in solver['tests']:
    #         plan[test].append(task)
    # # Process configs
    # # Process tests
    # return plan
    return spec


def parse_plan_spec(plan_file):
    pass


def parse_plan_arg(plan_arg):
    plan_spec = {}
    for pair in plan_arg.split(';'):
        pars = pair.split(':')
        config = pars[0]
        tests = pars[1].split(',')
        plan_spec[config] = tests
    return plan_spec


def process_plan_spec(plan_spec, config):
    return plan_spec
    pass


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
        if line.find('BMARK') != 0:
            continue
        line = line[len('BMARK'):]
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


def run_plan(plan, config):
    global plan_results

    solvers = config['solvers']
    configs = config['configs']
    tests = config['tests']

    tprint('Starting test plan')
    plan_results = defaultdict(lambda: defaultdict(list))
    for config_name, config_tests in plan.iteritems():
        config_info = configs[config_name]
        config_solver = solvers[config_info['solver']]
        tprint('Starting config', config_name)
        for test_name in config_tests:
            test_info = tests[test_name]
            tprint('Starting test', test_name)
            for param in test_info['params']:
                param['seed'] = 1000
                param['typ'] = test_info['type']
                args = config_solver['arg_template'].format(**param)
                cmd = config_info['run'] + ' ' + args
                tprint('Running task with run cmd:', cmd)
                p = subprocess.Popen(
                    cmd,
                    cwd=config_solver['directory'],
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
                plan_results[config_name][test_name].append(result)
                tprint('Finished tasks')
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
    with open(args.spec, 'r') as config_fo:
        config_spec = parse_config_spec(config_fo)

    tprint('Processing test spec')
    config = process_config_spec(config_spec)

    tprint('Parsing plan spec')
    if args.plan_file:
        with open(args.plan_file, 'r') as plan_fo:
            plan_spec = parse_plan_spec(plan_fo)
    elif args.plan:
        plan_spec = parse_plan_arg(args.plan)
    else:
        tprint('No specifed plan')

    tprint('Building test code')
    plan = process_plan_spec(plan_spec, config)

    test_results = run_plan(plan, config)
    tprint('Saving results to', args.results)

    with open(args.results, 'w') as results_fo:
        save_test_results(test_results, results_fo)
    generate_test_graphs(test_results)
    tprint('Script took', time.time() - start_time, 'seconds')


def sigint_handler(signal, frame):
    tprint('Sigint received, exiting')
    if plan_results is not None:
        tprint('Saving partial results')
        with open(cmd_args.results, 'w') as results_fo:
            save_test_results(plan_results, results_fo)
    sys.exit(0)


if __name__ == '__main__':
    signal.signal(signal.SIGINT, sigint_handler)

    parser = argparse.ArgumentParser()
    parser.add_argument('--spec', nargs='?', default='spec.json')
    parser.add_argument('--results', nargs='?', default='results.json')
    parser.add_argument('--plan', nargs='?')
    parser.add_argument('--plan_file', nargs='?')
    cmd_args = parser.parse_args()
    main(cmd_args)
