"""
Tests for cctbx/xray/targets/curvatures.h: expected (Fisher information)
curvatures of the mlf target and Gauss-Newton curvatures of the
least-squares target with respect to |Fcalc|.
"""
from __future__ import absolute_import, division, print_function
from cctbx import xray
from cctbx.array_family import flex
from libtbx.test_utils import approx_equal
import math, random, cmath

def exercise_fisher_information():
  J = xray.targets_amplitude_fisher_information
  for centric in [False, True]:
    assert J(nu=0, centric=centric) == 0
    # small nu: J ~ nu^2 (acentric), 2 nu^2 (centric)
    nu = 0.05
    assert approx_equal(J(nu=nu, centric=centric) / nu**2,
      (2 if centric else 1), eps=2.e-2)
    # monotonic, approaching 1
    values = [J(nu=0.1*i, centric=centric) for i in range(200)]
    assert all(b >= a for a, b in zip(values, values[1:]))
    assert 0.99 < values[-1] <= 1
    # continuity at the end of the table (nu_max = 10)
    assert approx_equal(J(nu=9.999, centric=centric),
      J(nu=10.001, centric=centric), eps=1.e-4)
  assert approx_equal(J(nu=20, centric=False), 1-0.5/400, eps=1.e-4)

def sample_f_obs(fc, a, b, k, e, centric, n):
  """|Fo| drawn from the distribution assumed by the mlf target."""
  x = a * k * fc
  result = flex.double()
  if (centric):
    sigma = math.sqrt(e * b * k * k)
    for i in range(n):
      result.append(abs(x + sigma * random.gauss(0, 1)))
  else:
    sigma = math.sqrt(e * b * k * k / 2)
    for i in range(n):
      result.append(abs(complex(x + sigma * random.gauss(0, 1),
                                sigma * random.gauss(0, 1))))
  return result

def exercise_mlf_monte_carlo(n=100000):
  for fc, a, b, k, e, centric in [
        (10, 0.8, 50, 1, 1, False),
        (10, 0.8, 50, 1, 1, True),
        (3, 0.5, 80, 1.5, 2, False),
        (3, 0.5, 80, 1.5, 2, True),
        (40, 0.95, 20, 0.7, 1, False),
        (1, 0.3, 100, 1, 1, False)]:
    f_obs = sample_f_obs(fc, a, b, k, e, centric, n)
    phase = cmath.exp(0.7j)
    f_calc = flex.complex_double(n, fc * phase)
    result = xray.mlf_target_and_gradients(
      f_obs=f_obs,
      r_free_flags=flex.bool(n, False),
      f_calc=f_calc,
      alpha=flex.double(n, a),
      beta=flex.double(n, b),
      scale_factor=k,
      epsilons=flex.double(n, e),
      centric_flags=flex.bool(n, centric),
      compute_gradients=True)
    # gradients_work() = conj(dT/dFc)/n_work; radial part = dT/d|Fc|
    g = result.gradients_work() * n
    d = flex.real(g * flex.conj(f_calc)) / fc
    expected = xray.targets_mlf_expected_curvature_one_h(
      fo=1, fc=fc, a=a, b=b, k=k, e=e, centric=centric)
    mc = flex.mean(d * d)
    # E[dT/d|Fc|] = 0 if f_obs follows the target's distribution
    assert abs(flex.mean(d)) < 4 * math.sqrt(mc / n)
    assert approx_equal(mc / expected, 1, eps=0.03)

def exercise_mlf_curvatures_work():
  random.seed(0)
  n = 50
  f_obs = flex.double([random.uniform(0.5, 50) for i in range(n)])
  f_obs[3] = 0 # ignored by the target
  f_calc = flex.complex_double([
    cmath.rect(random.uniform(0.5, 50), random.uniform(0, 6)) for i in range(n)])
  alpha = flex.double([random.uniform(0.3, 1) for i in range(n)])
  beta = flex.double([random.uniform(10, 100) for i in range(n)])
  alpha[5] = 0 # invalid
  epsilons = flex.double([random.choice([1,1,2]) for i in range(n)])
  centric_flags = flex.bool([random.random() < 0.2 for i in range(n)])
  r_free_flags = flex.bool([random.random() < 0.1 for i in range(n)])
  k = 1.3
  c = xray.targets_mlf_expected_curvatures_work(
    f_obs=f_obs, r_free_flags=r_free_flags, f_calc=f_calc, alpha=alpha,
    beta=beta, scale_factor=k, epsilons=epsilons,
    centric_flags=centric_flags)
  work = (~r_free_flags).iselection()
  assert c.size() == work.size()
  for j, i in enumerate(work):
    expected = xray.targets_mlf_expected_curvature_one_h(
      fo=f_obs[i], fc=abs(f_calc[i]), a=alpha[i], b=beta[i], k=k,
      e=epsilons[i], centric=centric_flags[i]) / work.size()
    assert approx_equal(c[j], expected)
    if (i in [3, 5]): assert c[j] == 0
    else: assert c[j] > 0

def exercise_least_squares():
  random.seed(1)
  n = 40
  f_calc = flex.complex_double([
    cmath.rect(random.uniform(0.5, 50), random.uniform(0, 6)) for i in range(n)])
  weights = flex.double([random.uniform(0.5, 2) for i in range(n)])
  r_free_flags = flex.bool([random.random() < 0.1 for i in range(n)])
  k = 1.3
  for obs_type in ["F", "I"]:
    c = flex.abs(f_calc)
    if (obs_type == "I"): c = c * c
    obs = k * c # perfect data: the exact Hessian is the Gauss-Newton term
    ls = xray.targets_least_squares(
      compute_scale_using_all_data=False, obs_type=obs_type, obs=obs,
      weights=weights, r_free_flags=r_free_flags, f_calc=f_calc,
      derivatives_depth=2, scale_factor=k)
    curvatures = xray.targets_least_squares_curvatures_work(
      obs_type=obs_type, obs=obs, weights=weights,
      r_free_flags=r_free_flags, f_calc=f_calc, scale_factor=k)
    fw = f_calc.select(~r_free_flags)
    assert curvatures.size() == fw.size()
    a = flex.real(fw) / flex.abs(fw)
    b = flex.imag(fw) / flex.abs(fw)
    h = ls.hessians_work()
    daa, dbb, dab = h.parts()
    radial = daa*a*a + dbb*b*b + 2*dab*a*b
    assert approx_equal(curvatures, radial)
    assert curvatures.all_gt(0)

def run():
  random.seed(0)
  exercise_fisher_information()
  exercise_mlf_monte_carlo()
  exercise_mlf_curvatures_work()
  exercise_least_squares()
  print("OK")

if (__name__ == "__main__"):
  run()
