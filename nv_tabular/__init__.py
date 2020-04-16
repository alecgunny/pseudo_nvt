import sys


py_ver = float(f'{sys.version_info.major}.{sys.version_info.minor}')
if py_ver >= 3.7:
   from collections import namedtuple
else:
   from .utils import _namedtuple as namedtuple

from . import ops, stats, writer
from .dataset import Dataset as dataset