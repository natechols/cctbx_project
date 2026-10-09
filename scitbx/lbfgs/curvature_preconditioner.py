"""
Diagonal preconditioning of scitbx.lbfgs.run() with approximate curvatures.

A target evaluator with diag_mode = "always" provides
compute_functional_gradients_diag(), which returns the functional, the
gradients and the diagonal of the initial inverse Hessian approximation
Hk0 (must be > 0). This module turns approximate curvatures (the diagonal
of the 2nd derivative matrix) into that diagonal:

  - curvatures below floor_fraction * mean(positive curvatures) (e.g.
    parameters without curvature information) are raised to that value;
  - the inverse curvatures D are scaled by s*y/(y*D*y) in each iteration
    (s: change of the parameters, y: change of the gradients), which
    compensates for an overall mis-scaling of the approximate curvatures.
    Without this scaling the minimizer can take excessively large steps.

scitbx.lbfgs.run() requests the diagonal at the start of each iteration
at the point just accepted by the line search, i.e. the point of the
previous evaluation. That evaluation is reused.

In the first iteration, scitbx.lbfgs uses the step length 1/|g| along
-D*g (appropriate for D = 1). The first diagonal is therefore multiplied
by |g|, so that the first step is the preconditioned Newton step -D*g
(otherwise it is too long if |g| < 1, e.g. near convergence, which can
move atoms by many Angstroms, and too short if |g| > 1).
"""
from __future__ import absolute_import, division, print_function
from scitbx.array_family import flex

class curvature_preconditioner(object):

  def __init__(self, floor_fraction=0.1):
    self.floor_fraction = floor_fraction
    self.last_evaluation = None
    self.last_request = None

  def __call__(self, x, compute):
    """
    x: the current parameters (flex.double, modified in place by the
       minimizer).
    compute(): returns f, g, curvatures at x.
    Returns f, g, diag for compute_functional_gradients_diag().
    """
    last = self.last_evaluation
    if (last is not None and last[0].all_eq(x)):
      x_last, f, g, d = last
      previous = self.last_request
      if (previous is not None):
        s = x_last - previous[0]
        y = g - previous[1]
        sy = s.dot(y)
        ydy = y.dot(d * y)
        if (sy > 0 and ydy > 0):
          d = d * (sy / ydy)
      self.last_request = (x_last, g)
      return f, g, d
    f, g, c = compute()
    c = c.deep_copy()
    positive = c.select(c > 0)
    if (positive.size() == 0):
      floor = 1
    else:
      floor = self.floor_fraction * flex.mean(positive)
    c.set_selected(c < floor, floor)
    d = 1 / c
    self.last_evaluation = (x.deep_copy(), f, g.deep_copy(), d)
    if (self.last_request is None):
      self.last_request = (self.last_evaluation[0], g.deep_copy())
      # first iteration: compensate the step length 1/|g| of scitbx.lbfgs
      g_norm = g.norm()
      if (g_norm > 0):
        return f, g, d * g_norm
    return f, g, d
