import re
import sys
from collections import namedtuple


ALL = 'all'
CONTINUOUS = 'CONTINUOUS'
CATEGORICAL = 'CATEGORICAL'


def snake_case_class_name(obj):
    return re.sub(r'(?<!^)(?=[A-Z])', '_', obj.__class__.__name__).lower()


def _namedtuple(typename, field_names, defaults=None):
    '''
    quick utility decorator function for adding defaults to namedtuple
    inheriters. Only necessary if python < 3.7, as after that point the
    `defaults` kwarg was added to namedtuple.
    '''
    cls = namedtuple(typename, field_names)
    if defaults is not None:
        cls.__new__.__defaults__ = tuple(defaults)
    return cls
