#!/usr/bin/env python2

from __future__ import division
from __future__ import with_statement
from __future__ import print_function

import subprocess
import pipes
import sys

import dmt

if __name__=='__main__':
  args=sys.argv[1:]

  subprocess.check_call(['rsync','-avzP','mrmaster:/isis2b/git/deepmanifold/{}'.format(args[0]),'.'])
  dmt.reconstruct_traversal(args[0])
