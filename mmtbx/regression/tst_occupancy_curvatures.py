"""
Tests for the occupancy refinement parameterization and its curvature
preconditioning (mmtbx/refinement/occupancies.py).
"""
from __future__ import absolute_import, division, print_function
import iotbx.pdb
import mmtbx.model
import mmtbx.f_model
import mmtbx
import mmtbx.refinement.occupancies as occupancies
import mmtbx.refinement.refinement_flags
from mmtbx.regression.tst_minimization_curvatures import pdb_str
from cctbx.array_family import flex
from libtbx.test_utils import approx_equal
from libtbx.utils import null_out
from libtbx import group_args
from mmtbx.refinement import minimization, weights
import mmtbx.refinement.adp_refinement
from cctbx import xray
import scitbx.lbfgs
import random

def exercise_packing():
  # groups: one unconstrained, a pair and a triple constrained to sum to 1
  table = [[0], [1, 2], [3, 4, 5]]
  c = flex.double([2.0, 3.0, 5.0, 1.0, 4.0, 7.0])
  a = flex.double([0.3, 0.2, 0.9, 0.1, 0.4, 0.6])
  # separable quadratic target of the group occupancies
  def f_par(par):
    return 0.5 * flex.sum(c * (par - a)**2)
  def f_x(x):
    return f_par(occupancies.pack_unpack(x=x, table=table))
  x = flex.double([0.5, 0.6, 0.4, 0.2, 0.3, 0.5])
  par = occupancies.pack_unpack(x=x, table=table)
  g_par = c * (par - a)
  g_x = occupancies.pack_gradients(x=g_par, table=table)
  c_x = occupancies.pack_curvatures(x=c, table=table)
  eps = 1.e-4
  for i in range(x.size()):
    xp = x.deep_copy(); xp[i] += eps
    xm = x.deep_copy(); xm[i] -= eps
    g_fd = (f_x(xp) - f_x(xm)) / (2*eps)
    c_fd = (f_x(xp) - 2*f_x(x) + f_x(xm)) / eps**2
    assert approx_equal(g_x[i], g_fd, eps=1.e-6), (i, g_x[i], g_fd)
    # exact for a separable quadratic target
    assert approx_equal(c_x[i], c_fd, eps=1.e-4), (i, c_x[i], c_fd)
  # the last parameters of constrained groups do not affect the target
  assert g_x[2] == 0 and g_x[5] == 0
  assert c_x[2] == 0 and c_x[5] == 0

def exercise_bounded_parameterization():
  # one unconstrained group, a pair and a triple constrained to sum to 1
  table = [[0], [1, 2], [3, 4, 5]]
  bp = occupancies.bounded_parameterization(table=table,
    occupancy_min=0.1, occupancy_max=0.9)
  q0 = flex.double([0.3, 0.2, 0.8, 0.1, 0.4, 0.5])
  x = bp.parameters(q0)
  assert approx_equal(bp.occupancies(x), q0)
  # occupancies at or beyond the bounds are moved inside
  q = bp.occupancies(bp.parameters(flex.double([0.95, 0, 1, -0.2, 0, 1.2])))
  assert q.all_gt(0) and q.all_lt(1)
  assert 0.1 < q[0] < 0.9
  assert approx_equal(q[1]+q[2], 1) and approx_equal(q[3]+q[4]+q[5], 1)
  # whatever the parameters, the occupancies are in range
  for x_rand in [flex.random_double(6)*40-20, flex.double(6, 1.e3),
                 flex.double(6, -1.e3)]:
    q = bp.occupancies(x_rand)
    assert 0.1 <= q[0] <= 0.9
    assert q.all_ge(0) and q.all_le(1)
    assert approx_equal(q[1]+q[2], 1) and approx_equal(q[3]+q[4]+q[5], 1)
  # gradients and Gauss-Newton curvatures vs. finite differences
  c = flex.double([2.0, 3.0, 5.0, 1.0, 4.0, 7.0])
  a = flex.double([0.7, 0.9, 0.1, 0.0, 0.6, 0.4])
  def f(x):
    return 0.5 * flex.sum(c * (bp.occupancies(x) - a)**2)
  x = flex.double([0.4, 0.3, -0.5, 1.0, -0.2, 0.1])
  g_x = bp.gradients(x=x, g=c * (bp.occupancies(x) - a))
  c_x = bp.curvatures(x=x, c=c)
  eps = 1.e-5
  for i in range(x.size()):
    xp = x.deep_copy(); xp[i] += eps
    xm = x.deep_copy(); xm[i] -= eps
    assert approx_equal(g_x[i], (f(xp)-f(xm))/(2*eps), eps=1.e-6)
    dq = (bp.occupancies(xp) - bp.occupancies(xm)) / (2*eps)
    assert approx_equal(c_x[i], flex.sum(c * dq**2), eps=1.e-6)
  # no gradient along the redundant direction of a constrained group
  assert approx_equal(g_x[1]+g_x[2], 0)
  assert approx_equal(g_x[3]+g_x[4]+g_x[5], 0)

def alt_conf_pdb_str(occ_a, occ_b, shift=0.7):
  """Residue 3 of pdb_str in two conformations."""
  lines = []
  for line in pdb_str.splitlines():
    if (line.startswith("ATOM") and int(line[22:26]) == 3):
      x = float(line[30:38])
      for alt, occ, dx in [("A", occ_a, 0.0), ("B", occ_b, shift)]:
        lines.append(line[:16] + alt + line[17:30] + "%8.3f" % (x+dx)
          + line[38:54] + "%6.2f" % occ + line[60:])
    else:
      lines.append(line)
  return "\n".join(lines) + "\n"

def get_model(pdb_text):
  model = mmtbx.model.manager(
    model_input=iotbx.pdb.input(source_info=None, lines=pdb_text),
    log=null_out())
  model.process(make_restraints=True)
  return model

def exercise_refinement():
  true_model = get_model(alt_conf_pdb_str(0.7, 0.3))
  f_obs = abs(true_model.get_xray_structure().structure_factors(
    d_min=1.5, algorithm="direct").f_calc())
  flags = f_obs.generate_r_free_flags(fraction=0.1)
  for use_curvatures in [False, True]:
    model = get_model(alt_conf_pdb_str(0.5, 0.5))
    groups = occupancies.occupancy_selections(model=model)
    assert len([g for g in groups if len(g) == 2]) == 1
    model.set_refinement_flags(mmtbx.refinement.refinement_flags.manager(
      occupancies=True, s_occupancies=groups))
    fmodel = mmtbx.f_model.manager(f_obs=f_obs, r_free_flags=flags,
      xray_structure=model.get_xray_structure(), target_name="ml")
    fmodel.update_all_scales(remove_outliers=False)
    r_start = fmodel.r_work()
    occupancies.manager(
      fmodels=mmtbx.fmodels(fmodel_xray=fmodel),
      model=model,
      max_number_of_iterations=50,
      number_of_macro_cycles=3,
      occupancy_max=1,
      occupancy_min=0,
      log=None,
      use_curvatures=use_curvatures)
    occ = model.get_xray_structure().scatterers().extract_occupancies()
    pair = [g for g in groups if len(g) == 2][0]
    occ_a = flex.mean(occ.select(pair[0]))
    occ_b = flex.mean(occ.select(pair[1]))
    assert approx_equal(occ_a + occ_b, 1)
    assert approx_equal(occ_a, 0.7, eps=0.05), (use_curvatures, occ_a)
    assert fmodel.r_work() < r_start

def joint_minimizer(model, fmodel, groups, use_curvatures, max_iterations):
  xrs = model.get_xray_structure()
  xray.set_scatterer_grad_flags(scatterers=xrs.scatterers(), u_iso=True)
  model.set_refinement_flags(mmtbx.refinement.refinement_flags.manager(
    individual_adp=True, adp_individual_iso=xrs.use_u_iso(),
    occupancies=True, s_occupancies=groups))
  w = weights.weights(wx=2, wx_scale=1, w=1)
  return minimization.lbfgs(
    fmodels=mmtbx.fmodels(fmodel_xray=fmodel),
    restraints_manager=model.restraints_manager,
    model=model,
    target_weights=group_args(xyz_weights_result=w, adp_weights_result=w),
    refine_adp=True,
    iso_restraints=mmtbx.refinement.adp_refinement.\
      adp_restraints_master_params.extract().iso,
    h_params=group_args(refine="riding"),
    lbfgs_termination_params=scitbx.lbfgs.termination_parameters(
      max_iterations=max_iterations),
    collect_monitor=False,
    log=null_out(),
    use_curvatures=use_curvatures,
    occupancy_groups=groups)

def exercise_joint_adp_occupancies():
  true_model = get_model(alt_conf_pdb_str(0.7, 0.3))
  f_obs = abs(true_model.get_xray_structure().structure_factors(
    d_min=1.5, algorithm="direct").f_calc())
  flags = f_obs.generate_r_free_flags(fraction=0.1)
  # direct summation: finite differences of FFT structure factors are not
  # accurate enough for this test
  sfg = mmtbx.f_model.sf_and_grads_accuracy_master_params.extract()
  sfg.algorithm = "direct"
  def setup():
    model = get_model(alt_conf_pdb_str(0.5, 0.5))
    xrs = model.get_xray_structure()
    xrs.shake_adp(spread=5, keep_anisotropic=True)
    model.set_xray_structure(xrs)
    groups = occupancies.occupancy_selections(model=model)
    fmodel = mmtbx.f_model.manager(f_obs=f_obs, r_free_flags=flags,
      xray_structure=model.get_xray_structure(), target_name="ls_wunit_k1",
      sf_and_grads_accuracy_params=sfg)
    fmodel.update_all_scales(remove_outliers=False)
    return model, fmodel, groups
  # gradients with respect to the parameters vs. finite differences
  model, fmodel, groups = setup()
  m = joint_minimizer(model, fmodel, groups, False, 1)
  n_other = m.x.size() - sum(len(g) for g in groups)
  assert n_other == model.get_xray_structure().scatterers().size()
  m._scatterers_start = m.xray_structure.scatterers().deep_copy()
  x0 = flex.double(n_other, 0)
  x0.extend(m._occ.bounded.parameters(flex.double([0.55, 0.45])))
  x0[3] = 0.01 # some u_iso shift
  m.x = x0.deep_copy()
  f0, g0 = m.compute_functional_and_gradients()
  g0 = g0.deep_copy()
  # (softmax: no gradient along the redundant direction)
  assert approx_equal(g0[-1] + g0[-2], 0)
  eps = 1.e-5
  for i in [0, 3, 10, n_other, n_other+1]:
    fs = []
    for e in [eps, -eps]:
      m.x = x0.deep_copy(); m.x[i] += e
      fs.append(m.compute_functional_and_gradients()[0])
    g_fd = (fs[0]-fs[1])/(2*eps)
    assert approx_equal(g0[i], g_fd, eps=1.e-4*max(1, abs(g_fd))), (
      i, g0[i], g_fd)
  # joint refinement: occupancies in range and summing to 1
  for use_curvatures in [False, True]:
    model, fmodel, groups = setup()
    r_start = fmodel.r_work()
    m = joint_minimizer(model, fmodel, groups, use_curvatures, 50)
    occ = m.xray_structure.scatterers().extract_occupancies()
    pair = [g for g in groups if len(g) == 2][0]
    occ_a = flex.mean(occ.select(pair[0]))
    occ_b = flex.mean(occ.select(pair[1]))
    assert approx_equal(occ_a + occ_b, 1)
    assert 0 <= occ_a <= 1
    assert fmodel.r_work() < r_start
    if (use_curvatures):
      assert m.curvatures.size() == m.x.size()
      assert m.curvatures.all_gt(0)

def run():
  random.seed(0)
  flex.set_random_seed(0)
  exercise_packing()
  exercise_bounded_parameterization()
  exercise_refinement()
  exercise_joint_adp_occupancies()
  print("OK")

if (__name__ == "__main__"):
  run()
