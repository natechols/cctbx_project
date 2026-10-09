"""Quick reciprocal-space refinement of coordinates, ADPs and occupancies"""
# LIBTBX_SET_DISPATCHER_NAME mmtbx.quickref
from __future__ import absolute_import, division, print_function

from iotbx.cli_parser import run_program
from mmtbx.programs import quickref

if __name__ == '__main__':
  run_program(program_class=quickref.Program)
