# System imports
from __future__ import print_function
from datetime import datetime
import os
import json

# Project imports


def tprint(*args, **kwargs):
    print('[' + str(datetime.now()) + ']', *args, **kwargs)


def parse_config_spec(file):
    spec = json.load(file)
    if not isinstance(spec, dict):
        return 'Config root must be an object'
    for solver_name, solver in spec['solvers'].iteritems():
        solver['directory'] = os.path.expanduser(solver['directory'])
    return spec


def gen_matrix_filename(test_name, params):
    return '{test}_{m}_{n}'.format(test=test_name,
                                   m=params['m'],
                                   n=params['n'])
