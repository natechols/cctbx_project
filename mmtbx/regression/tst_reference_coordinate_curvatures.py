"""
Tests for the curvatures (diagonal 2nd derivatives) of the reference
coordinate restraints (mmtbx/geometry_restraints/reference_coordinate.h),
which are used for preconditioning in mmtbx.refinement.minimization.lbfgs
(use_curvatures=True).
"""
from __future__ import absolute_import, division, print_function
import boost_adaptbx.boost.python as bp
from mmtbx.geometry_restraints.reference import add_coordinate_restraints
from mmtbx.regression.tst_minimization_curvatures import get_model
from cctbx.array_family import flex
from libtbx.test_utils import approx_equal
import random

ext = bp.import_ext("mmtbx_reference_coordinate_ext")

def exercise_finite_differences(top_out):
  n = 20
  ref_sites = flex.vec3_double([[random.uniform(-5, 5) for k in range(3)]
    for i in range(n)])
  proxies = add_coordinate_restraints(sites_cart=ref_sites, sigma=0.3,
    limit=1.0, top_out_potential=top_out)
  # displacements on both sides of the inflection point of the top-out
  # potential
  sites = ref_sites + flex.vec3_double(
    [[random.uniform(-1.5, 1.5) for k in range(3)] for i in range(n)])
  def residual(s):
    return ext.reference_coordinate_residual_sum(sites_cart=s,
      proxies=proxies, gradient_array=flex.vec3_double())
  g = flex.vec3_double(n, (0,0,0))
  c = flex.vec3_double(n, (0,0,0))
  r = ext.reference_coordinate_residual_sum(sites_cart=sites,
    proxies=proxies, gradient_array=g, curvature_array=c)
  assert approx_equal(r, residual(sites))
  # gradients are unchanged by the curvature computation
  g0 = flex.vec3_double(n, (0,0,0))
  ext.reference_coordinate_residual_sum(sites_cart=sites, proxies=proxies,
    gradient_array=g0)
  assert approx_equal(g, g0)
  eps = 1.e-4
  n_clipped = 0
  for i in range(n):
    for k in range(3):
      fs = []
      for e in [eps, -eps]:
        s = sites.deep_copy()
        x = list(s[i]); x[k] += e; s[i] = x
        fs.append(residual(s))
      g_fd = (fs[0] - fs[1]) / (2*eps)
      c_fd = (fs[0] - 2*r + fs[1]) / eps**2
      assert approx_equal(g[i][k], g_fd, eps=1.e-5*max(1, abs(g_fd)))
      # exact where positive, clipped at zero otherwise
      if (c_fd < 0):
        assert c[i][k] == 0
        n_clipped += 1
      else:
        assert approx_equal(c[i][k], c_fd, eps=1.e-3*max(1, c_fd)), (
          top_out, i, k, c[i][k], c_fd)
  if (top_out):
    assert n_clipped > 0
  else:
    assert n_clipped == 0
    assert approx_equal(c.as_double(), flex.double(3*n, 2/0.3**2))

def exercise_energies():
  """geometry_restraints.manager.energies_sites adds the curvatures"""
  model = get_model()
  geometry = model.get_restraints_manager().geometry
  sites_cart = model.get_sites_cart()
  isel = flex.size_t(range(0, sites_cart.size(), 2))
  geometry.add_reference_coordinate_restraints_in_place(
    pdb_hierarchy=model.get_hierarchy().select(isel),
    selection=isel,
    sigma=0.2,
    n_atoms_in_target_model=sites_cart.size(),
    exclude_outliers=False)
  assert geometry.reference_coordinate_proxies.size() == isel.size()
  shifted = sites_cart + flex.vec3_double(sites_cart.size(), (0.1,0,0))
  e_all = geometry.energies_sites(sites_cart=shifted,
    compute_gradients=True, compute_curvatures=True)
  expected = flex.vec3_double(sites_cart.size(), (0,0,0))
  expected.set_selected(isel, (2/0.2**2,)*3)
  geometry.remove_reference_coordinate_restraints_in_place()
  e_none = geometry.energies_sites(sites_cart=shifted,
    compute_gradients=True, compute_curvatures=True)
  assert approx_equal(e_all.curvatures.as_double(),
    e_none.curvatures.as_double() + expected.as_double())

def run():
  random.seed(0)
  flex.set_random_seed(0)
  exercise_finite_differences(top_out=False)
  exercise_finite_differences(top_out=True)
  exercise_energies()
  print("OK")

if (__name__ == "__main__"):
  run()
