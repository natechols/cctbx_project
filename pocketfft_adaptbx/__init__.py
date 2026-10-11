"""
3D FFTs with pocketfft (pocketfft/pocketfft_hdronly.h, BSD license),
with the interface of scitbx.fftpack.real_to_complex_3d and
complex_to_complex_3d (same array layouts, signs and scaling),
multi-threaded. The default number of threads is OMP_NUM_THREADS, or the
number of processors.
"""
from __future__ import absolute_import, division, print_function
import os
import boost_adaptbx.boost.python as bp
ext = bp.import_ext("pocketfft_adaptbx_ext")

def default_n_threads():
  value = os.environ.get("OMP_NUM_THREADS")
  if (value is not None):
    try:
      n = int(value.split(",")[0])
      if (n > 0): return n
    except ValueError:
      pass
  return 0 # number of processors

def real_to_complex_3d(n_real, n_threads=None):
  if (n_threads is None): n_threads = default_n_threads()
  return ext.real_to_complex_3d(tuple(n_real), n_threads)

def complex_to_complex_3d(n, n_threads=None):
  if (n_threads is None): n_threads = default_n_threads()
  return ext.complex_to_complex_3d(tuple(n), n_threads)
