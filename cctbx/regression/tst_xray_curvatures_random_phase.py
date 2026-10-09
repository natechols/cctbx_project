"""
Tests for cctbx/xray/curvatures_random_phase.h: the random-phase
approximation of the Gauss-Newton curvatures of a structure factor target
is compared with the exact Gauss-Newton curvatures, which are obtained
from finite differences of directly summed structure factors.
"""
from __future__ import absolute_import, division, print_function
from cctbx import xray, miller, adptbx
from cctbx.development import random_structure
from cctbx.array_family import flex
from libtbx.test_utils import approx_equal
import math, random, sys

def make_structure(space_group_symbol, n_atoms, anisotropic=False,
      anomalous=False):
  xs = random_structure.xray_structure(
    space_group_symbol=space_group_symbol,
    elements=(["C"]*6+["N"]*2+["O"]*2+["S"])*(n_atoms//11+1),
    volume_per_atom=20, min_distance=1.3,
    general_positions_only=True, random_u_iso=False)
  xs = xs.select(flex.size_t(range(n_atoms)))
  for i, sc in enumerate(xs.scatterers()):
    sc.u_iso = adptbx.b_as_u(random.uniform(5, 80))
    sc.occupancy = random.choice([1,1,1,0.6])
    if (anisotropic and i % 2 == 0):
      u = sc.u_iso
      u_cart = adptbx.random_u_cart(u_scale=u, u_min=0.2*u)
      sc.u_star = adptbx.u_cart_as_u_star(xs.unit_cell(), u_cart)
      sc.u_star = xs.space_group().average_u_star(sc.u_star)
      sc.set_use_u_aniso_only()
    if (anomalous and i % 3 == 0):
      sc.fp = -1.5
      sc.fdp = 2.5
  return xs

def target_curvatures(miller_set):
  # some positive per-reflection values with a resolution dependence
  d_star_sq = miller_set.d_star_sq().data()
  return flex.double([0.5 + random.random() for i in range(d_star_sq.size())]) \
       * flex.exp(-d_star_sq)

def exact_gauss_newton(xs, miller_set, curvatures, eps=1.e-5):
  """sum_h curvatures_h * (d|F_h|/dp)^2 by finite differences."""
  def f_calc(s):
    return miller_set.structure_factors_from_scatterers(
      s, algorithm="direct").f_calc().data()
  f0 = f_calc(xs)
  a0 = flex.abs(f0)
  def gn(shifted_plus, shifted_minus):
    df = (f_calc(shifted_plus) - f_calc(shifted_minus)) / (2*eps)
    # derivative of |F|: projection of dF onto F
    da = (flex.real(df)*flex.real(f0) + flex.imag(df)*flex.imag(f0)) / a0
    return flex.sum(curvatures * da * da)
  uc = xs.unit_cell()
  sites = flex.vec3_double()
  u_iso = flex.double()
  u_cart = flex.double() # 6 values per anisotropic scatterer
  occupancy = flex.double()
  for i_sc in range(xs.scatterers().size()):
    c = []
    for k in range(3):
      shifted = []
      for e in [eps, -eps]:
        s = xs.deep_copy_scatterers()
        sc = s.scatterers()[i_sc]
        site_cart = list(uc.orthogonalize(sc.site))
        site_cart[k] += e
        sc.site = uc.fractionalize(site_cart)
        shifted.append(s)
      c.append(gn(*shifted))
    sites.append(c)
    for attr, values in [("u_iso", u_iso), ("occupancy", occupancy)]:
      sc = xs.scatterers()[i_sc]
      if (attr == "u_iso" and sc.flags.use_u_aniso()):
        values.append(0)
        continue
      shifted = []
      for e in [eps, -eps]:
        s = xs.deep_copy_scatterers()
        sc = s.scatterers()[i_sc]
        setattr(sc, attr, getattr(sc, attr) + e)
        shifted.append(s)
      values.append(gn(*shifted))
    if (xs.scatterers()[i_sc].flags.use_u_aniso()):
      for k in range(6):
        shifted = []
        for e in [eps, -eps]:
          s = xs.deep_copy_scatterers()
          sc = s.scatterers()[i_sc]
          u = list(adptbx.u_star_as_u_cart(uc, sc.u_star))
          u[k] += e
          sc.u_star = adptbx.u_cart_as_u_star(uc, u)
          shifted.append(s)
        u_cart.append(gn(*shifted))
  return sites, u_iso, u_cart, occupancy

def random_phase(xs, miller_set, curvatures, **kwds):
  return xray.structure_factors_curvatures_random_phase(
    unit_cell=xs.unit_cell(),
    space_group=xs.space_group(),
    scatterers=xs.scatterers(),
    scattering_type_registry=xs.scattering_type_registry(),
    miller_indices=miller_set.indices(),
    target_curvatures=curvatures,
    **kwds)

def compare(approx, exact, label, max_rms_log, verbose):
  sel = exact > 0
  a = approx.select(sel)
  e = exact.select(sel)
  assert a.all_gt(0)
  log_ratio = flex.log(a / e)
  mean_ratio = flex.mean(a / e)
  rms_log = math.sqrt(flex.mean(log_ratio**2))
  corr = flex.linear_correlation(flex.log(a), flex.log(e)).coefficient()
  if (verbose):
    print("  %-10s n=%3d mean ratio %.3f rms(log) %.3f corr(log) %.4f" % (
      label, e.size(), mean_ratio, rms_log, corr))
  assert abs(mean_ratio - 1) < max_rms_log
  assert rms_log < max_rms_log
  assert corr > 0.9

def exercise(space_group_symbol, n_atoms=40, d_min=1.5, anisotropic=False,
      anomalous=False, verbose=False):
  xs = make_structure(space_group_symbol, n_atoms, anisotropic, anomalous)
  miller_set = miller.build_set(
    xs.crystal_symmetry(), anomalous_flag=False, d_min=d_min)
  curvatures = target_curvatures(miller_set)
  if (verbose):
    print("%s: %d atoms, %d reflections, anisotropic=%s, anomalous=%s" % (
      space_group_symbol, n_atoms, miller_set.size(), anisotropic, anomalous))
  exact_sites, exact_u_iso, exact_u_cart, exact_occ = exact_gauss_newton(
    xs, miller_set, curvatures)
  r = random_phase(xs, miller_set, curvatures)
  compare(r.sites_cart.as_double(), exact_sites.as_double(), "sites", 0.25,
    verbose)
  compare(r.u_iso, exact_u_iso, "u_iso", 0.25, verbose)
  compare(r.occupancy, exact_occ, "occupancy", 0.25, verbose)
  aniso = xs.use_u_aniso()
  if (aniso.count(True) > 0):
    approx_u_cart = r.u_cart.select(aniso).as_double()
    compare(approx_u_cart, exact_u_cart, "u_cart", 0.25, verbose)
  assert r.u_cart.select(~aniso).as_double().all_eq(0)
  # tabulation vs. direct summation
  d = random_phase(xs, miller_set, curvatures, use_table=False)
  for step, eps in [(1, 2.e-3), (0.25, 2.e-4)]:
    t = random_phase(xs, miller_set, curvatures, b_iso_grid_step=step)
    for name in ["sites_cart", "u_iso", "occupancy"]:
      tv = getattr(t, name)
      dv = getattr(d, name)
      if (name == "sites_cart"):
        tv = tv.as_double(); dv = dv.as_double()
      sel = dv > 0
      assert flex.max(flex.abs(tv.select(sel)/dv.select(sel) - 1)) < eps
  for i, sc in enumerate(xs.scatterers()):
    if (sc.flags.use_u_aniso()):
      assert r.u_iso[i] == 0

def exercise_scaling():
  # curvatures are linear in target_curvatures and scale with occupancy^2
  xs = make_structure("P212121", 20)
  ms = miller.build_set(xs.crystal_symmetry(), anomalous_flag=False,
    d_min=2)
  c = target_curvatures(ms)
  r1 = random_phase(xs, ms, c)
  r2 = random_phase(xs, ms, 3*c)
  assert approx_equal(r2.sites_cart.as_double(), 3*r1.sites_cart.as_double())
  assert approx_equal(r2.occupancy, 3*r1.occupancy)
  xs2 = xs.deep_copy_scatterers()
  for sc in xs2.scatterers(): sc.occupancy *= 0.5
  r3 = random_phase(xs2, ms, c)
  assert approx_equal(r3.sites_cart.as_double(),
    0.25*r1.sites_cart.as_double())
  assert approx_equal(r3.u_iso, 0.25*r1.u_iso)
  assert approx_equal(r3.occupancy, r1.occupancy)

def run(args):
  verbose = "--verbose" in args
  random.seed(0)
  flex.set_random_seed(0)
  exercise_scaling()
  for space_group_symbol in ["P1", "P-1", "P21", "P212121", "C2", "R3",
                             "P6122", "I41"]:
    exercise(space_group_symbol, verbose=verbose)
  exercise("P21", anisotropic=True, anomalous=True, verbose=verbose)
  exercise("P6122", anisotropic=True, verbose=verbose)
  print("OK")

if (__name__ == "__main__"):
  run(sys.argv[1:])
