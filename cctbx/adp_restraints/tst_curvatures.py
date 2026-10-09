"""
Tests for the curvatures (diagonal of the 2nd derivatives) of the ADP
restraints used by mmtbx: crystal.adp_iso_local_sphere_restraints_energies
(Gauss-Newton), adp_restraints.energies_iso and
adp_restraints.eval_adp_aniso_restraints (exact).
"""
from __future__ import absolute_import, division, print_function
from cctbx import adp_restraints, crystal, adptbx, xray
from cctbx import geometry_restraints
from cctbx.development import random_structure
from cctbx.array_family import flex
from libtbx.test_utils import approx_equal
from libtbx import group_args
import random

def local_sphere(xrs, pair_sym_table, u_isos, average_power,
      compute_curvatures=True):
  return crystal.adp_iso_local_sphere_restraints_energies(
    pair_sym_table=pair_sym_table,
    orthogonalization_matrix=xrs.unit_cell().orthogonalization_matrix(),
    sites_frac=xrs.sites_frac(),
    u_isos=u_isos,
    selection=flex.bool(u_isos.size(), True),
    use_u_iso=flex.bool(u_isos.size(), True),
    grad_u_iso=flex.bool(u_isos.size(), True),
    sphere_radius=5.0,
    distance_power=0.69,
    average_power=average_power,
    min_u_sum=1.e-6,
    compute_gradients=True,
    collect=False,
    compute_curvatures=compute_curvatures)

def fd_second_derivatives(f, x, eps=1.e-5):
  result = flex.double()
  f0 = f(x)
  for i in range(x.size()):
    xp = x.deep_copy(); xp[i] += eps
    xm = x.deep_copy(); xm[i] -= eps
    result.append((f(xp) - 2*f0 + f(xm)) / eps**2)
  return result

def make_structure(n_atoms=12):
  xrs = random_structure.xray_structure(
    space_group_symbol="P212121",
    elements=["C"]*n_atoms,
    volume_per_atom=20, min_distance=1.3,
    general_positions_only=True)
  pair_sym_table = xrs.pair_asu_table(
    distance_cutoff=5.0).extract_pair_sym_table()
  return xrs, pair_sym_table

def exercise_local_sphere():
  xrs, pst = make_structure()
  n = xrs.scatterers().size()
  u_random = flex.double([adptbx.b_as_u(random.uniform(10, 60))
    for i in range(n)])
  # quadratic for average_power=0: curvatures are the exact diagonal
  e = local_sphere(xrs, pst, u_random, average_power=0)
  assert e.number_of_restraints > 0
  expected = fd_second_derivatives(
    lambda u: local_sphere(xrs, pst, u, 0).residual_sum, u_random)
  assert approx_equal(e.curvatures, expected, eps=1.e-3*flex.max(expected))
  # at equal u's all residual terms are zero: Gauss-Newton is exact
  u_equal = flex.double(n, adptbx.b_as_u(30))
  e = local_sphere(xrs, pst, u_equal, average_power=1.7)
  expected = fd_second_derivatives(
    lambda u: local_sphere(xrs, pst, u, 1.7).residual_sum, u_equal)
  assert approx_equal(e.curvatures, expected, eps=1.e-3*flex.max(expected))
  # general case: non-negative, gradients unchanged
  e = local_sphere(xrs, pst, u_random, average_power=1.7)
  e0 = local_sphere(xrs, pst, u_random, average_power=1.7,
    compute_curvatures=False)
  assert e.curvatures.all_ge(0)
  assert flex.max(e.curvatures) > 0
  assert e0.curvatures.size() == 0
  assert approx_equal(e.gradients, e0.gradients)
  assert approx_equal(e.residual_sum, e0.residual_sum)

def exercise_local_sphere_two_atoms():
  # closed form for a single pair
  xrs = xray.structure(crystal_symmetry=crystal.symmetry(
    unit_cell=(20,20,20,90,90,90), space_group_symbol="P1"))
  for site in [(0.1,0.1,0.1), (0.17,0.1,0.1)]:
    xrs.add_scatterer(xray.scatterer("C", site=site))
  pst = xrs.pair_asu_table(distance_cutoff=2).extract_pair_sym_table()
  u = flex.double([0.2, 0.5])
  p = 1.7
  e = local_sphere(xrs, pst, u, average_power=p)
  dist = 0.07*20
  w = 1 / dist**0.69
  s = (u[0] + u[1]) / 2
  d = u[0] - u[1]
  dr1 = s**(-p/2) * (1 - p*d/(4*s))
  dr2 = s**(-p/2) * (-1 - p*d/(4*s))
  # residual = w r^2, r = d / s^(p/2)
  assert approx_equal(e.residual_sum, w * (d/s**(p/2))**2)
  assert approx_equal(e.gradients, [2*w*(d/s**(p/2))*dr1,
                                    2*w*(d/s**(p/2))*dr2])
  assert approx_equal(e.curvatures, [2*w*dr1**2, 2*w*dr2**2])

def exercise_energies_iso():
  xrs, pst = make_structure()
  for sc in xrs.scatterers():
    sc.u_iso = adptbx.b_as_u(random.uniform(10, 60))
    sc.flags.set_grad_u_iso(True)
  parameters = group_args(sphere_radius=5.0, distance_power=0.69,
    average_power=1.7, wilson_b_weight=0.5, wilson_b_weight_auto=False)
  kwds = dict(plain_pair_sym_table=pst, xray_structure=xrs,
    parameters=parameters, use_u_local_only=False, use_hd=False)
  plain = adp_restraints.energies_iso(compute_curvatures=True, **kwds)
  wilson = adp_restraints.energies_iso(compute_curvatures=True,
    wilson_b=25, **kwds)
  assert adp_restraints.energies_iso(**kwds).curvatures is None
  n = xrs.scatterers().size()
  # the Wilson term w*(mean(u)-u_w)^2/u_w is quadratic
  assert approx_equal(wilson.curvatures - plain.curvatures,
    flex.double(n, 2*0.5/(n*n*adptbx.b_as_u(25))))
  # normalization applies to the curvatures as to the gradients
  normalized = adp_restraints.energies_iso(compute_curvatures=True,
    normalization=True, **kwds)
  assert approx_equal(normalized.curvatures,
    plain.curvatures * normalized.normalization_factor)

def exercise_aniso():
  xrs, pst = make_structure(n_atoms=8)
  uc = xrs.unit_cell()
  for i, sc in enumerate(xrs.scatterers()):
    if (i % 2 == 0):
      sc.u_star = adptbx.u_cart_as_u_star(uc, adptbx.random_u_cart(
        u_scale=0.3, u_min=0.05))
      sc.set_use_u_aniso_only()
      sc.flags.set_grad_u_aniso(True)
    else:
      sc.u_iso = adptbx.b_as_u(random.uniform(10, 60))
      sc.flags.set_grad_u_iso(True)
  proxies = geometry_restraints.shared_bond_simple_proxy()
  # aniso-aniso, iso-iso, aniso-iso, iso-aniso pairs
  for i, j in [(0,2), (1,3), (2,5), (3,4), (4,6), (5,7), (0,1)]:
    proxies.append(geometry_restraints.bond_simple_proxy(
      i_seqs=(i,j), distance_ideal=1.5, weight=1))
  scatterers = xrs.scatterers()
  u_cart = scatterers.extract_u_cart(uc)
  u_iso = scatterers.extract_u_iso()
  def evaluate(u_cart, u_iso):
    return adp_restraints.eval_adp_aniso_restraints(
      scatterers=scatterers, u_cart=u_cart, u_iso=u_iso,
      bond_proxies=proxies, selection=flex.bool(scatterers.size(), True),
      hd_selection=flex.bool(scatterers.size(), False),
      n_grad_u_iso=xrs.n_grad_u_iso(), use_hd=False)
  r = evaluate(u_cart, u_iso)
  # finite differences (exact for the quadratic target)
  x = u_cart.as_double()
  fd_aniso = fd_second_derivatives(
    lambda x: evaluate(flex.sym_mat3_double(x), u_iso).target, x, eps=1.e-3)
  fd_iso = fd_second_derivatives(
    lambda u: evaluate(u_cart, u).target, u_iso, eps=1.e-3)
  aniso = scatterers.extract_use_u_aniso()
  c_aniso = r.curvatures_aniso_cart()
  c_iso = r.curvatures_iso()
  for i in range(scatterers.size()):
    if (aniso[i]):
      assert approx_equal(c_aniso[i], fd_aniso[6*i:6*i+6], eps=1.e-4)
      assert c_iso[i] == 0
    else:
      assert approx_equal(c_iso[i], fd_iso[i], eps=1.e-4)
      assert c_aniso[i] == (0,0,0,0,0,0)
  assert flex.max(c_iso) == 2 + 6 # e.g. atom 3: one iso-iso, one iso-aniso pair
  # the python wrapper
  class rm(object):
    def pair_proxies(self, sites_cart):
      return group_args(bond_proxies=group_args(simple=proxies))
  w = adp_restraints.adp_aniso_restraints(xray_structure=xrs,
    restraints_manager=rm(), use_hd=False)
  assert approx_equal(w.curvatures_iso, c_iso)
  assert approx_equal(w.curvatures_aniso_cart, c_aniso)

def run():
  random.seed(0)
  flex.set_random_seed(0)
  exercise_local_sphere()
  exercise_local_sphere_two_atoms()
  exercise_energies_iso()
  exercise_aniso()
  print("OK")

if (__name__ == "__main__"):
  run()
