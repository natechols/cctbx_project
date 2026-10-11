"""
Tests for TLS groups refined simultaneously with the coordinates and ADPs
in mmtbx.refinement.minimization.lbfgs (tls_groups): gradients with respect
to all parameters (including the TLS parameters) vs. finite differences,
and refinement with and without curvatures.
"""
from __future__ import absolute_import, division, print_function
import mmtbx.f_model
import mmtbx
import mmtbx.refinement.adp_refinement
import mmtbx.refinement.refinement_flags
from mmtbx.refinement import minimization, weights
from mmtbx.regression.tst_minimization_curvatures import get_model
from mmtbx.tls import tools
from cctbx import xray, adptbx
from cctbx.array_family import flex
from libtbx.test_utils import approx_equal
from libtbx.utils import null_out
from libtbx import group_args
import scitbx.lbfgs
import random

def tls_model():
  """Model with two TLS groups: U = U_tls + u_iso (residual)."""
  model = get_model()
  xrs = model.get_xray_structure()
  n = xrs.scatterers().size()
  selections = [flex.size_t(range(0, n//2)), flex.size_t(range(n//2, n))]
  sites = xrs.sites_cart()
  tlsos = []
  for i, sel in enumerate(selections):
    origin = sites.select(sel).mean()
    tlsos.append(tools.tlso(
      t = (0.10+0.02*i, 0.12, 0.08, 0.01, -0.005, 0.004),
      l = (2.0, 3.0+i, 1.5, 0.3, -0.2, 0.1),
      s = (0.02, 0.01, -0.01, 0.005, -0.03, 0.02, 0.01, -0.005, 0.01),
      origin = origin))
  u_tls = tools.u_cart_from_tls(sites_cart=sites, selections=selections,
    tlsos=tlsos)
  scatterers = xrs.scatterers()
  for sc in scatterers:
    sc.flags.set_use_u_iso(True)
    sc.flags.set_use_u_aniso(True)
  residual = flex.double([adptbx.b_as_u(random.uniform(3, 12))
    for i in range(n)])
  scatterers.set_u_iso(residual, flex.bool(n, True), xrs.unit_cell())
  xrs.set_u_cart(u_cart=u_tls, selection=flex.size_t(range(n)))
  model.set_xray_structure(xrs)
  return model, selections, tlsos

def get_minimizer(model, fmodel, selections, tlsos, use_curvatures,
                  max_iterations, refine_trace_t = False):
  xrs = model.get_xray_structure()
  xray.set_scatterer_grad_flags(scatterers = xrs.scatterers(), site = True,
    u_iso = True)
  n = xrs.scatterers().size()
  model.set_refinement_flags(mmtbx.refinement.refinement_flags.manager(
    individual_sites = True, individual_adp = True,
    sites_individual = flex.bool(n, True),
    adp_individual_iso = xrs.use_u_iso()))
  # (restraints weight such that the data dominate: synthetic data)
  w = weights.weights(wx = 1, wx_scale = 1, w = 0.01)
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
    tls_groups = group_args(selections = selections, tlsos = tlsos),
    tls_refine_trace_t = refine_trace_t)

def perturbed_tlsos(tlsos):
  result = []
  for t in tlsos:
    result.append(tools.tlso(
      t = [v * random.uniform(0.6, 1.4) for v in t.t],
      l = [v * random.uniform(0.6, 1.4) for v in t.l],
      s = [v * random.uniform(0.5, 1.5) for v in t.s],
      origin = t.origin))
  return result

def u_total(xrs):
  """U_cart + u_iso (as flex.double, 6 per atom)"""
  a = xrs.scatterers().extract_u_cart(xrs.unit_cell()).as_double()
  u = xrs.scatterers().extract_u_iso()
  for k in range(3):
    i = flex.size_t_range(k, a.size(), 6)
    a.set_selected(i, a.select(i) + u)
  return a

def exercise():
  true_model, selections, true_tlsos = tls_model()
  xrs_true = true_model.get_xray_structure()
  f_obs = abs(xrs_true.structure_factors(d_min = 1.5,
    algorithm = "direct").f_calc())
  flags = f_obs.generate_r_free_flags(fraction = 0.1)
  sfg = mmtbx.f_model.sf_and_grads_accuracy_master_params.extract()
  sfg.algorithm = "direct"
  def setup():
    model = true_model.deep_copy()
    xrs = model.get_xray_structure()
    xrs.shake_sites_in_place(rms_difference = 0.15)
    # perturbed TLS (U_tls) and residual u_iso
    tlsos = perturbed_tlsos(true_tlsos)
    xrs.set_u_cart(u_cart = tools.u_cart_from_tls(
      sites_cart = xrs.sites_cart(), selections = selections, tlsos = tlsos),
      selection = flex.size_t(range(xrs.scatterers().size())))
    u = xrs.scatterers().extract_u_iso()
    u = u * flex.double([random.uniform(0.7, 1.3) for i in range(u.size())])
    xrs.scatterers().set_u_iso(u, flex.bool(u.size(), True), xrs.unit_cell())
    model.set_xray_structure(xrs)
    fmodel = mmtbx.f_model.manager(f_obs = f_obs, r_free_flags = flags,
      xray_structure = model.get_xray_structure(), target_name = "ls_wunit_k1",
      sf_and_grads_accuracy_params = sfg)
    fmodel.update_all_scales(remove_outliers = False)
    return model, fmodel, tlsos
  # gradients with respect to the parameters vs. finite differences
  for refine_trace_t in [False, True]:
    model, fmodel, tlsos = setup()
    m = get_minimizer(model, fmodel, selections, tlsos,
      use_curvatures = False, max_iterations = 1,
      refine_trace_t = refine_trace_t)
    n_atoms = model.get_xray_structure().scatterers().size()
    # sites and u_iso of all atoms, and 20 TLS parameters per group (21
    # with the trace of T)
    n_tls = (20 + int(refine_trace_t)) * len(selections)
    assert m.x.size() == 4 * n_atoms + n_tls, m.x.size()
    m._scatterers_start = m.xray_structure.scatterers().deep_copy()
    x0 = flex.double([random.uniform(-0.002, 0.002)
      for i in range(m.x.size())])
    m.x = x0.deep_copy()
    f0, g0 = m.compute_functional_and_gradients()
    g0 = g0.deep_copy()
    i_tls = list(range(m.x.size() - n_tls, m.x.size()))
    eps = 1.e-5
    for i in random.sample(range(m.x.size() - n_tls), 20) + i_tls:
      fs = []
      for e in [eps, -eps]:
        m.x = x0.deep_copy(); m.x[i] += e
        fs.append(m.compute_functional_and_gradients()[0])
      g_fd = (fs[0] - fs[1]) / (2*eps)
      # (S33 is not a parameter: trace(S) = 0)
      assert approx_equal(g0[i], g_fd, eps = 1.e-4 * max(1, abs(g_fd))), (
        refine_trace_t, i, g0[i], g_fd)
  # refinement: R lower, total ADPs (U_tls + u_iso) closer to the true ADPs
  # (the trace of T is not refined: an isotropic error of T is compensated
  # by u_iso)
  u_true = u_total(xrs_true)
  for use_curvatures in [False, True]:
    model, fmodel, tlsos = setup()
    r_start = fmodel.r_work()
    e_start = flex.mean_sq(u_total(model.get_xray_structure()) - u_true)**0.5
    m = get_minimizer(model, fmodel, selections, tlsos, use_curvatures,
      max_iterations = 100)
    xrs = m.xray_structure
    e_final = flex.mean_sq(u_total(xrs) - u_true)**0.5
    assert fmodel.r_work() < 0.4 * r_start, (r_start, fmodel.r_work())
    assert e_final < 0.8 * e_start, (e_start, e_final)
    # the U_cart of the atoms are those of the refined TLS parameters
    u_tls = tools.u_cart_from_tls(sites_cart = m.tls_sites_cart_start(),
      selections = selections, tlsos = m.tls_result)
    u_star = xrs.scatterers().extract_u_star()
    for i in range(xrs.scatterers().size()):
      assert approx_equal(adptbx.u_star_as_u_cart(xrs.unit_cell(), u_star[i]),
        u_tls[i], eps = 1.e-6)
    # trace of T is not refined
    for t, t0 in zip(m.tls_result, tlsos):
      assert approx_equal(sum(t.t[:3]), sum(t0.t[:3]), eps = 1.e-8)

def run():
  random.seed(0)
  flex.set_random_seed(0)
  exercise()
  print("OK")

if (__name__ == "__main__"):
  run()
