"""pocketfft_adaptbx vs. scitbx.fftpack: same results (to rounding), array
layouts and accessors, for 1 and several threads."""
from __future__ import absolute_import, division, print_function
from scitbx import fftpack
import pocketfft_adaptbx as pocketfft
from scitbx.array_family import flex
import random

def flat(a):
  result = a.deep_copy()
  result.resize(flex.grid(result.size()))
  return result

def rel_diff(a, b):
  return flex.max(flex.abs(a - b)) / max(1.e-300, flex.max(flex.abs(b)))

def exercise_real_to_complex(n, n_threads):
  f = fftpack.real_to_complex_3d(n)
  p = pocketfft.real_to_complex_3d(n, n_threads)
  assert tuple(p.n_real()) == tuple(f.n_real())
  assert tuple(p.m_real()) == tuple(f.m_real())
  assert tuple(p.n_complex()) == tuple(f.n_complex())
  m = f.m_real()
  a = flex.random_double(m[0]*m[1]*m[2])
  a.reshape(flex.grid(m).set_focus(n))
  c_f = f.forward(a.deep_copy())
  c_p = p.forward(a.deep_copy())
  assert c_p.accessor().all() == c_f.accessor().all()
  assert c_p.accessor().focus() == c_f.accessor().focus()
  assert rel_diff(c_p.as_1d(), c_f.as_1d()) < 1.e-13
  r_f = f.backward(c_f.deep_copy())
  r_p = p.backward(c_f.deep_copy())
  assert r_p.accessor().all() == r_f.accessor().all()
  assert r_p.accessor().focus() == r_f.accessor().focus()
  # (the padding is undefined)
  focus = flex.size_t([i*m[1]*m[2] + j*m[2] + k
    for i in range(n[0]) for j in range(n[1]) for k in range(n[2])])
  assert rel_diff(flat(r_p).select(focus), flat(r_f).select(focus)) < 1.e-13
  # forward + backward: the input times the number of grid points
  assert rel_diff(flat(r_p).select(focus) / (n[0]*n[1]*n[2]),
    flat(a).select(focus)) < 1.e-13

def exercise_complex_to_complex(n, n_threads):
  f = fftpack.complex_to_complex_3d(n)
  p = pocketfft.complex_to_complex_3d(n, n_threads)
  assert tuple(p.n()) == tuple(f.n())
  size = n[0]*n[1]*n[2]
  c = flex.polar(flex.random_double(size), flex.random_double(size)*6.28)
  c.reshape(flex.grid(n))
  for direction in ["forward", "backward"]:
    c_f = getattr(f, direction)(c.deep_copy())
    c_p = getattr(p, direction)(c.deep_copy())
    assert c_p.accessor().all() == c_f.accessor().all()
    assert rel_diff(c_p.as_1d(), c_f.as_1d()) < 1.e-13

def run():
  random.seed(0)
  flex.set_random_seed(0)
  for n in [(1, 1, 1), (2, 3, 4), (12, 10, 14), (9, 15, 7), (30, 24, 25),
            (17, 19, 23)]:
    for n_threads in [1, 3]:
      exercise_real_to_complex(n, n_threads)
      exercise_complex_to_complex(n, n_threads)
  assert pocketfft.real_to_complex_3d((4, 4, 4)).n_threads() >= 1
  print("OK")

if (__name__ == "__main__"):
  run()
