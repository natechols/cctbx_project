"""
Tests for riding hydrogens in mmtbx.refinement.minimization.lbfgs
(riding_h=True): the hydrogens are not parameters, their positions are
idealized from the parent atoms and their ADPs are those of the parent
atoms (isotropic: U_eq of anisotropic parents); gradients vs. finite
differences for simultaneous refinement of coordinates and ADPs, with and
without curvatures.
"""
from __future__ import absolute_import, division, print_function
import mmtbx.f_model
import mmtbx
import mmtbx.refinement.adp_refinement
import mmtbx.refinement.refinement_flags
from mmtbx.refinement import minimization, weights
from mmtbx.regression.tst_minimization_curvatures import get_model
from cctbx import xray
from cctbx.array_family import flex
from libtbx.test_utils import approx_equal
from libtbx.utils import null_out
from libtbx import group_args
import scitbx.lbfgs
import random

def get_model_with_h():
  from mmtbx.hydrogens import reduce_hydrogen
  model = get_model()
  # (the charged N-terminus needs the chemical components library)
  rh = reduce_hydrogen.place_hydrogens(model = model,
    n_terminal_charge = "no_charge")
  rh.run()
  model = rh.get_model()
  model.set_log(null_out())
  model.process(make_restraints = True)
  model.setup_riding_h_manager(idealize = True)
  return model

def get_minimizer(model, fmodel, use_curvatures, max_iterations):
  xrs = model.get_xray_structure()
  aniso = xrs.use_u_aniso()
  xray.set_scatterer_grad_flags(scatterers = xrs.scatterers(), site = True,
    u_iso = True, u_aniso = (aniso.count(True) > 0))
  n = xrs.scatterers().size()
  model.set_refinement_flags(mmtbx.refinement.refinement_flags.manager(
    individual_sites = True, individual_adp = True,
    sites_individual = flex.bool(n, True),
    adp_individual_iso = xrs.use_u_iso(),
    adp_individual_aniso = (aniso if aniso.count(True) > 0 else None)))
  w = weights.weights(wx = 1, wx_scale = 1, w = 1)
  return minimization.lbfgs(
    fmodels = mmtbx.fmodels(fmodel_xray = fmodel),
    restraints_manager = model.restraints_manager,
    model = model,
    target_weights = group_args(xyz_weights_result = w, adp_weights_result = w),
    refine_xyz = True,
    refine_adp = True,
    iso_restraints = mmtbx.refinement.adp_refinement.\
      adp_restraints_master_params.extract().iso,
    h_params = group_args(refine = "riding"),
    lbfgs_termination_params = scitbx.lbfgs.termination_parameters(
      max_iterations = max_iterations),
    collect_monitor = False,
    log = null_out(),
    use_curvatures = use_curvatures,
    riding_h = True)

def exercise(anisotropic = False):
  true_model = get_model_with_h()
  if (anisotropic):
    # anisotropic non-H atoms (hydrogens are isotropic, with U_eq of the
    # parent atoms)
    xrs = true_model.get_xray_structure()
    xrs.convert_to_anisotropic(selection = ~true_model.get_hd_selection())
    true_model.set_xray_structure(xrs)
  hd = true_model.get_hd_selection()
  assert hd.count(True) > 20
  xrs_true = true_model.get_xray_structure()
  f_obs = abs(xrs_true.structure_factors(d_min = 1.5,
    algorithm = "direct").f_calc())
  flags = f_obs.generate_r_free_flags(fraction = 0.1)
  sfg = mmtbx.f_model.sf_and_grads_accuracy_master_params.extract()
  sfg.algorithm = "direct"
  def setup():
    model = true_model.deep_copy()
    xrs = model.get_xray_structure()
    xrs.shake_sites_in_place(rms_difference = 0.2)
    xrs.shake_adp(spread = 5, keep_anisotropic = True)
    model.set_xray_structure(xrs)
    fmodel = mmtbx.f_model.manager(f_obs = f_obs, r_free_flags = flags,
      xray_structure = model.get_xray_structure(), target_name = "ls_wunit_k1",
      sf_and_grads_accuracy_params = sfg)
    fmodel.update_all_scales(remove_outliers = False)
    return model, fmodel
  # gradients with respect to the parameters vs. finite differences
  model, fmodel = setup()
  m = get_minimizer(model, fmodel, use_curvatures = False, max_iterations = 1)
  n_non_h = (~hd).count(True)
  # sites and ADPs of non-H atoms only
  assert m.x.size() == [4, 9][int(anisotropic)] * n_non_h
  m._scatterers_start = m.xray_structure.scatterers().deep_copy()
  x0 = flex.double([random.uniform(-0.02, 0.02) for i in range(m.x.size())])
  m.x = x0.deep_copy()
  f0, g0 = m.compute_functional_and_gradients()
  g0 = g0.deep_copy()
  eps = 1.e-5
  for i in random.sample(range(m.x.size()), 30):
    fs = []
    for e in [eps, -eps]:
      m.x = x0.deep_copy(); m.x[i] += e
      fs.append(m.compute_functional_and_gradients()[0])
    g_fd = (fs[0] - fs[1]) / (2*eps)
    assert approx_equal(g0[i], g_fd, eps = 1.e-3 * max(1, abs(g_fd))), (
      i, g0[i], g_fd)
  # refinement: hydrogens at ideal positions, ADPs of the parent atoms
  for use_curvatures in [False, True]:
    model, fmodel = setup()
    r_start = fmodel.r_work()
    m = get_minimizer(model, fmodel, use_curvatures, max_iterations = 50)
    assert fmodel.r_work() < 0.7 * r_start, (r_start, fmodel.r_work())
    xrs = m.xray_structure
    sites = xrs.sites_cart()
    ideal = sites.deep_copy()
    model.riding_h_manager.idealize_riding_h_positions(sites_cart = ideal)
    assert flex.max((sites - ideal).norms()) < 1.e-6
    # (hydrogens are isotropic, u_iso = U_eq of the parent)
    u = xrs.extract_u_iso_or_u_equiv()
    r = m._riding
    assert approx_equal(u.select(r.h_isel), u.select(r.p_isel))
    assert xrs.use_u_aniso().select(r.h_isel).count(True) == 0
    if (use_curvatures):
      assert m.curvatures.size() == m.x.size()
      assert m.curvatures.all_gt(0)

def run():
  random.seed(0)
  flex.set_random_seed(0)
  exercise()
  exercise(anisotropic = True)
  print("OK")

if (__name__ == "__main__"):
  run()
