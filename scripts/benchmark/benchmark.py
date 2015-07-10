# System imports
from __future__ import print_function
from collections import defaultdict
import time
import os
import sys
import signal
import subprocess32 as subprocess
import argparse
import json

# Project imports
from util import (tprint,
                  parse_config_spec,
                  parse_plan_arg,
                  parse_plan_file)


# Need a global plan results to save partial results when interrupted
cmd_args = None
plan_results = None


def build_arg_string(args):
    arg = ''
    for k, v in args.iteritems():
        arg += ' --{key} {value}'.format(key=k, value=v)
    return arg


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
    plan_results = defaultdict(lambda: defaultdict(list))

    solvers = config['solvers']
    configs = config['configs']
    tests = config['tests']

    config_name = plan['config_name']
    test_name = plan['test_name']
    param_num = plan['param_num']

    config_info = configs[config_name]
    config_solver = solvers[config_info['solver']]
    tprint('Running', config_name, 'for', test_name,
           'for param num', param_num)

    test_info = tests[test_name]
    param = test_info['params'][param_num]
    param['seed'] = 1000
    param['typ'] = test_info['type']
    args = config_solver['arg_template'].format(**param)
    cmd = config_info['run'] + ' ' + args

    tprint('Running test with run cmd:', cmd)
    p = subprocess.Popen(
        cmd,
        cwd=config_solver['directory'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True
    )
    tout = 3 * 60 * 60
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
    tprint('Finished test')
    return plan_results


def run_plan_memory(plan, config):
    global plan_results
    plan_results = defaultdict(lambda: defaultdict(list))

    solvers = config['solvers']
    configs = config['configs']
    tests = config['tests']

    for config_name, test_names in plan.iteritems():
        test_name = test_names[0]

        tprint('Running', config_name, 'for', test_name)
        config_info = configs[config_name]
        config_solver = solvers[config_info['solver']]
        test_info = tests[test_name]

        for param in test_info['params']:
            param['seed'] = 1000
            param['type'] = test_info['type']
            args = build_arg_string(param)
            cmd = config_info['run'] + ' ' + args

            tprint('Running test with run cmd:', cmd)
            p = subprocess.Popen(
                cmd,
                cwd=config_solver['directory'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=True
            )
            tout = 3 * 60 * 60
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
        tprint('Finished test')
    return plan_results


def save_test_results(results, out_fo):
    json.dump(results, out_fo)


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
            plan_spec = parse_plan_file(plan_fo)
    elif args.plan:
        plan_spec = parse_plan_arg(args.plan)
    else:
        tprint('No specifed plan')

    tprint('Building test code')
    plan = process_plan_spec(plan_spec, config)

    if args.matrix:
        tprint('Running plan with saved matrix')
        test_results = run_plan(plan, config)
    else:
        tprint('Running plan with saved matrix')
        test_results = run_plan_memory(plan, config)
    tprint('Saving results to', args.results)

    with open(args.results, 'w') as results_fo:
        save_test_results(test_results, results_fo)
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
    parser.add_argument('--matrix', nargs='?')
    parser.add_argument('--plan_file', nargs='?')
    cmd_args = parser.parse_args()
    main(cmd_args)

# python benchmark.py --spec ref_spec.json --plan 'row_2_d:lasso_mn_d_1:2' --results fjkl.json
