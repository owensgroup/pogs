import sys
import json
import subprocess
import os
import argparse
from collections import defaultdict
from pprint import pprint

test_args_template = '{typ} {M} {N} {nnz}'

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
        ks = [k.strip() for k in kv[0].split(',')]
        v = kv[1]
        temp = result
        for k in ks[:-1]:
            temp = temp[k]
        temp[ks[-1]] = v
    return result


def run_plan(plan, test_settings):
    print('Starting test plan')
    results = defaultdict(lambda: defaultdict(list))
    for test_name, tests in plan.iteritems():
        settings = test_settings[test_name]
        typ = settings['type']
        print('')
        for param in settings['params']:
            M = param['M']
            N = param['N']
            nnz = int(int(M) * int(N) * float(param['density']))
            args = test_args_template.format(typ=typ, M=M, N=N, nnz=nnz)
            print('Starting tests for args: ' + args)
            for task in tests:
                cmd = task['run'] + ' ' + args
                print('Running task with run cmd: ' + cmd)
                p = subprocess.Popen(
                    cmd,
                    cwd=task['directory'],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    shell=True
                )
                sout, serr = p.communicate()
                print('Finished task')
                result = parse_solver_output(sout, serr)
                results[test_name][task['name']].append(result)
            print('Finished tests')
    print('Finished test plan')
    return results


def save_test_results(results, out_fo):
    json.dump(results, out_fo)


def generate_test_graphs(results):
    pass


def main(args):
    with args.spec_file:
        print('Parsing test spec')
        spec = parse_test_spec(args.spec_file)
    print('Processing test spec')
    plan = process_test_spec(spec)
    print('Building test code')
    make_tests(spec['solvers'])
    test_results = run_plan(plan, spec['tests'])
    pprint(test_results)
    with args.results_file:
        save_test_results(test_results, args.results_file)
    generate_test_graphs(test_results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('spec_file',
                        type=argparse.FileType('r'),
                        default='spec.json')
    parser.add_argument('results_file',
                        type=argparse.FileType('w'),
                        default='results.json')
    main(parser.parse_args())
