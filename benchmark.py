import sys
import json
import subprocess
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
    for test in spec:
        pass
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
        p = subprocess.Popen('make', 'test', cwd=solver['directory'])
        p.wait()


def parse_solver_output(output, error):
    if error is not None:
        pass
    result = dict()
    for line in output:
        kv = [x.trim() for x in line.split(':')]
        result[kv[0]] = kv[1]
    return result


def run_plan(plan, test_settings):
    print('Starting test plan')
    results = defaultdict(defaultdict(list))
    for test_name, tests in plan.iteritems():
        settings = test_settings[test_name]
        typ = settings['type']
        print('')
        for param in settings['params']:
            M = param['M']
            N = param['N']
            nnz = int(M) * int(N) * int(param['density'])
            args = test_args_template.format(typ=typ, M=M, N=N, nnz=nnz)
            print('Starting tests for args: ' + args)
            for task in tests:
                print('Running task with run cmd: ' + task['run'])
                p = subprocess.Popen(
                    task['run'],
                    args,
                    cwd=task['directory'],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                sout, serr = p.communicate()
                print('Finished task')
                result = parse_solver_output(sout, serr)
                results[test_name][task['name']].append(result)
            print('Finished tests')
    print('Finished test plan')
    return results


def save_test_results(results):
    pass


def display_test_results(results):
    pass


def main(argv):
    if len(argv) > 1:
        spec_file = argv[1]
    else:
        spec_file = 'spec.json'

    spec_fo = open(spec_file, 'r')
    print('Parsing test spec')
    spec = parse_test_spec(spec_fo)
    spec_fo.close()

    print('Processing test spec')
    plan = process_test_spec(spec)
    print('Building test code')
    make_tests(spec['solvers'])
    test_results = run_plan(plan)
    pprint(test_results)
    save_test_results()
    display_test_results(test_results)


if __name__ == '__main__':
    main(sys.argv)
