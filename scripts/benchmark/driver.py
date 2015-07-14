#! python

# System imports
import time
import os
import json
import subprocess32 as subprocess
import argparse

# Project imports
from util import (tprint, gen_matrix_filename)


# Config arg
def parse_config_spec(file):
    spec = json.load(file)
    if not isinstance(spec, dict):
        return 'Config root must be an object'
    for solver_name, solver in spec['solvers'].iteritems():
        solver['directory'] = os.path.expanduser(solver['directory'])
    return spec


# Plan arg
def parse_plan_arg(plan_arg):
    plan_spec = {}
    for pair in plan_arg.split(';'):
        pars = pair.split(':')
        test = pars[0]
        configs = pars[1].split(',')
        plan_spec[test] = configs
    return plan_spec


def parse_plan_file(plan_file):
    pass


def process_plan_spec(plan_spec, config):
    return plan_spec


def qsub(name, script, args, misc):
    cmd = ('qsub -F "{args}" -N {name} -o {name}.out -e {name}.err'
           '{misc} {script}')
    cmd = cmd.format(name=name, misc=misc, args=args, script=script)
    return subprocess.Popen(cmd,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)


def gen_matrix_job(test_name, params, cleanup_job_id):
    misc = '-l nodes=6:k40x6:ppn=6 -l walltime=24:00:00'
    if cleanup_job_id:
        depend = ' -W depend=afterok:' + cleanup_job_id
        misc = misc + depend

    matrix_file = gen_matrix_filename(test_name, params)
    matrix_params = '--m {m} --n {n} --type {typ} --out {matrix}'
    matrix_params = matrix_params.format(m=params['m'],
                                         n=params['n'],
                                         typ=params['type'],
                                         matrix=matrix_file)
    p = qsub('gen_matrix', 'gen_matrix.sh', matrix_params, misc)
    return p.communicate().readline()


def cleanup_matrix_job(test_name, params, job_ids):
    misc = '-l nodes=6:k40x6:ppn=6 -l walltime=24:00:00'
    depend = ' -W depend=afterok'
    for job_id in job_ids:
        depend = depend + ':' + job_id
    misc = misc + depend

    matrix_file = gen_matrix_filename(test_name, params)
    matrix_params = '--matrix {matrix}'.format(matrix=matrix_file)
    p = qsub('cleanup_matrix', 'cleanup_matrix.sh', matrix_params, misc)
    return p.communicate().readline()


def run_job_file(spec_file, config_name, config, test_name, param_num,
                 gen_job_id):
    test_resources = '-l {resources} -l walltime=24:00:00'
    test_resources.format(resources=config['resources'])
    depend_on = '-W depend=afterok:{matrix_id}'
    depend_on.format(matrix_id=gen_job_id)
    misc = test_resources + ' ' + depend_on
    job_plan = config_name + ':' + test_name + ':' + param_num
    results_file = job_plan.replace(':', '_') + '_results.json'
    args = '--spec {spec} --plan {plan} --results {results}'
    args = args.format(spec=spec_file, plan=job_plan, results=results_file)
    p = qsub(config_name, 'run_job.sh', args, misc)
    return p.communicate().readline()


def run_job_memory(spec_file, config_name, config, test_name):
    test_resources = '-l {resources} -l walltime=24:00:00'
    test_resources.format(resources=config['resources'])
    misc = test_resources
    job_plan = config_name + ':' + test_name
    results_file = job_plan.replace(':', '_') + '_results.json'
    args = '--spec {spec} --plan {plan} --results {results}'
    args = args.format(spec=spec_file, plan=job_plan, results=results_file)
    p = qsub(config_name, 'run_job_memory.sh', args, misc)
    return p.communicate().readline()


#
def launch_jobs_file(spec_file, plan, config):
    configs = config['configs']
    tests = config['tests']

    for test_name, configurations in plan.iteritems():
        test = tests[test_name]
        params = test['params']
        cleanup_job_id = None
        for param_num, param in enumerate(params):
            # Generate matrix for this param
            gen_job_id = gen_matrix_job(params, cleanup_job_id)
            run_job_ids = []
            for config_name in configurations:
                config = configs[config_name]
                job_id = run_job_file(spec_file,
                                      config_name,
                                      config,
                                      test_name,
                                      param_num,
                                      gen_job_id)

                run_job_ids.append(job_id)
            # cleanup matrix
            p = qsub()
            cleanup_job_id = p.communicate().readline()
        # Finish up after a single test
    # Finish up after all tests
    p = qsub()


def launch_jobs_memory(spec_file, plan, config):
    configs = config['configs']
    tests = config['tests']

    for test_name, configurations in plan.iteritems():
        if not test_name in tests:
            tprint('Test', test_name, 'not in tests')
            continue
        for config_name in configurations:
            config = configs[config_name]
            run_job_memory(spec_file,
                           config_name,
                           config,
                           test_name)


def main(args):
    with open(args.spec, 'r') as config_fo:
        config_spec = parse_config_spec(config_fo)
    config = config_spec

    if args.plan_file:
        with open(args.plan_file, 'r') as plan_fo:
            plan_spec = parse_plan_file(plan_fo)
    elif args.plan:
        plan_spec = parse_plan_arg(args.plan)
    plan = process_plan_spec(plan_spec, config)

    launch_jobs_memory(args.spec, plan, config)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec')
    parser.add_argument('--plan', nargs='?')
    parser.add_argument('--plan_file', nargs='?')
    cmd_args = parser.parse_args()

    main(cmd_args)
