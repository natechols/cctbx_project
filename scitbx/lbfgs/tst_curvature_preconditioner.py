"""
Tests for scitbx.lbfgs.curvature_preconditioner: with exact curvatures of a
separable quadratic, the first step of scitbx.lbfgs.run() is the Newton
step (independent of the gradient norm), i.e. the minimum is found in the
first iteration.
"""
from __future__ import absolute_import, division, print_function
from scitbx.array_family import flex
from scitbx.lbfgs.curvature_preconditioner import curvature_preconditioner
import scitbx.lbfgs
import random

class quadratic(object):

  def __init__(self, c, a, x):
    self.c = c
    self.a = a
    self.x = x.deep_copy()
    self.n = x.size()
    self.n_evaluations = 0
    # exact curvatures: no floor
    self.preconditioner = curvature_preconditioner(floor_fraction=0)
    self.diag_mode = "always"
    self.minimizer = scitbx.lbfgs.run(target_evaluator=self,
      termination_params=scitbx.lbfgs.termination_parameters(
        max_iterations=100))

  def compute_functional_and_gradients(self):
    self.n_evaluations += 1
    d = self.x - self.a
    return 0.5 * flex.sum(self.c * d * d), self.c * d

  def compute_functional_gradients_diag(self):
    def compute():
      f, g = self.compute_functional_and_gradients()
      return f, g, self.c
    return self.preconditioner(self.x, compute)

def exercise():
  n = 50
  c = flex.double([10**random.uniform(-3, 3) for i in range(n)])
  a = flex.double([random.uniform(-1, 1) for i in range(n)])
  for scale in [1.e-4, 1, 1.e4]:
    # starting points with small and large gradients
    x = a + flex.double([random.uniform(-1, 1) for i in range(n)]) * scale
    q = quadratic(c, a, x)
    assert flex.max(flex.abs(q.x - a)) < 1.e-6 * max(1, scale), scale
    # the first trial step is accepted: 2 evaluations for the first
    # iteration, then convergence
    assert q.n_evaluations <= 4, (scale, q.n_evaluations)

def run():
  random.seed(0)
  exercise()
  print("OK")

if (__name__ == "__main__"):
  run()
