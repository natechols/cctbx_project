"""
Tests for curvature preconditioning in mmtbx.refinement.minimization.lbfgs
(use_curvatures=True) and target_result.curvatures_work() in
mmtbx.refinement.targets.
"""
from __future__ import absolute_import, division, print_function
import iotbx.pdb
import mmtbx.model
import mmtbx.f_model
import mmtbx
from mmtbx.refinement import minimization, weights
from cctbx import xray, adptbx
from cctbx.array_family import flex
from libtbx.test_utils import approx_equal
from libtbx.utils import null_out
from libtbx import group_args
import mmtbx.refinement.adp_refinement
import mmtbx.refinement.refinement_flags
import scitbx.lbfgs
import random

pdb_str = """\
CRYST1   50.840   42.770   28.950  90.00  90.00  90.00 P 21 21 21    4          
ATOM      1  N   MET A   1      27.340  24.430   2.614  1.00  9.67           N  
ATOM      2  CA  MET A   1      26.266  25.413   2.842  1.00 10.38           C  
ATOM      3  C   MET A   1      26.913  26.639   3.531  1.00  9.62           C  
ATOM      4  O   MET A   1      27.886  26.463   4.263  1.00  9.62           O  
ATOM      5  CB  MET A   1      25.112  24.880   3.649  1.00 13.77           C  
ATOM      6  CG  MET A   1      25.353  24.860   5.134  1.00 16.29           C  
ATOM      7  SD  MET A   1      23.930  23.959   5.904  1.00 17.17           S  
ATOM      8  CE  MET A   1      24.447  23.984   7.620  1.00 16.11           C  
ATOM      9  N   GLN A   2      26.335  27.770   3.258  1.00  9.27           N  
ATOM     10  CA  GLN A   2      26.850  29.021   3.898  1.00  9.07           C  
ATOM     11  C   GLN A   2      26.100  29.253   5.202  1.00  8.72           C  
ATOM     12  O   GLN A   2      24.865  29.024   5.330  1.00  8.22           O  
ATOM     13  CB  GLN A   2      26.733  30.148   2.905  1.00 14.46           C  
ATOM     14  CG  GLN A   2      26.882  31.546   3.409  1.00 17.01           C  
ATOM     15  CD  GLN A   2      26.786  32.562   2.270  1.00 20.10           C  
ATOM     16  OE1 GLN A   2      27.783  33.160   1.870  1.00 21.89           O  
ATOM     17  NE2 GLN A   2      25.562  32.733   1.806  1.00 19.49           N  
ATOM     18  N   ILE A   3      26.849  29.656   6.217  1.00  5.87           N  
ATOM     19  CA  ILE A   3      26.235  30.058   7.497  1.00  5.07           C  
ATOM     20  C   ILE A   3      26.882  31.428   7.862  1.00  4.01           C  
ATOM     21  O   ILE A   3      27.906  31.711   7.264  1.00  4.61           O  
ATOM     22  CB  ILE A   3      26.344  29.050   8.645  1.00  6.55           C  
ATOM     23  CG1 ILE A   3      27.810  28.748   8.999  1.00  4.72           C  
ATOM     24  CG2 ILE A   3      25.491  27.771   8.287  1.00  5.58           C  
ATOM     25  CD1 ILE A   3      27.967  28.087  10.417  1.00 10.83           C  
ATOM     26  N   PHE A   4      26.214  32.097   8.771  1.00  4.55           N  
ATOM     27  CA  PHE A   4      26.772  33.436   9.197  1.00  4.68           C  
ATOM     28  C   PHE A   4      27.151  33.362  10.650  1.00  5.30           C  
ATOM     29  O   PHE A   4      26.350  32.778  11.395  1.00  5.58           O  
ATOM     30  CB  PHE A   4      25.695  34.498   8.946  1.00  4.83           C  
ATOM     31  CG  PHE A   4      25.288  34.609   7.499  1.00  7.97           C  
ATOM     32  CD1 PHE A   4      24.147  33.966   7.038  1.00  6.69           C  
ATOM     33  CD2 PHE A   4      26.136  35.346   6.640  1.00  8.34           C  
ATOM     34  CE1 PHE A   4      23.812  34.031   5.677  1.00  9.10           C  
ATOM     35  CE2 PHE A   4      25.810  35.392   5.267  1.00 10.61           C  
ATOM     36  CZ  PHE A   4      24.620  34.778   4.853  1.00  8.90           C  
ATOM     37  N   VAL A   5      28.260  33.943  11.096  1.00  4.44           N  
ATOM     38  CA  VAL A   5      28.605  33.965  12.503  1.00  3.87           C  
ATOM     39  C   VAL A   5      28.638  35.461  12.900  1.00  4.93           C  
ATOM     40  O   VAL A   5      29.522  36.103  12.320  1.00  6.84           O  
ATOM     41  CB  VAL A   5      29.963  33.317  12.814  1.00  2.99           C  
ATOM     42  CG1 VAL A   5      30.211  33.394  14.304  1.00  5.28           C  
ATOM     43  CG2 VAL A   5      29.957  31.838  12.352  1.00  9.13           C  
ATOM     44  N   LYS A   6      27.751  35.867  13.740  1.00  6.04           N  
ATOM     45  CA  LYS A   6      27.691  37.315  14.143  1.00  6.12           C  
ATOM     46  C   LYS A   6      28.469  37.475  15.420  1.00  6.57           C  
ATOM     47  O   LYS A   6      28.213  36.753  16.411  1.00  5.76           O  
ATOM     48  CB  LYS A   6      26.219  37.684  14.307  1.00  7.45           C  
ATOM     49  CG  LYS A   6      25.884  39.139  14.615  1.00 11.12           C  
ATOM     50  CD  LYS A   6      24.348  39.296  14.642  1.00 14.54           C  
ATOM     51  CE  LYS A   6      23.865  40.723  14.749  1.00 18.84           C  
ATOM     52  NZ  LYS A   6      22.375  40.720  14.907  1.00 20.55           N  
ATOM     53  N   THR A   7      29.426  38.430  15.446  1.00  7.41           N  
ATOM     54  CA  THR A   7      30.225  38.643  16.662  1.00  7.48           C  
ATOM     55  C   THR A   7      29.664  39.839  17.434  1.00  8.75           C  
ATOM     56  O   THR A   7      28.850  40.565  16.859  1.00  8.58           O  
ATOM     57  CB  THR A   7      31.744  38.879  16.299  1.00  9.61           C  
ATOM     58  OG1 THR A   7      31.737  40.257  15.824  1.00 11.78           O  
ATOM     59  CG2 THR A   7      32.260  37.969  15.171  1.00  9.17           C  
ATOM     60  N   LEU A   8      30.132  40.069  18.642  1.00  9.84           N  
ATOM     61  CA  LEU A   8      29.607  41.180  19.467  1.00 14.15           C  
ATOM     62  C   LEU A   8      30.075  42.538  18.984  1.00 17.37           C  
ATOM     63  O   LEU A   8      29.586  43.570  19.483  1.00 17.01           O  
ATOM     64  CB  LEU A   8      29.919  40.890  20.938  1.00 16.63           C  
ATOM     65  CG  LEU A   8      29.183  39.722  21.581  1.00 18.88           C  
ATOM     66  CD1 LEU A   8      29.308  39.750  23.095  1.00 19.31           C  
ATOM     67  CD2 LEU A   8      27.700  39.721  21.228  1.00 18.59           C  
END
"""

class target_weights(object):
  def __init__(self, w):
    self.xyz_weights_result = w
    self.adp_weights_result = w

def get_model():
  model = mmtbx.model.manager(
    model_input=iotbx.pdb.input(source_info=None, lines=pdb_str),
    log=null_out())
  model.process(make_restraints=True)
  return model

def get_data(model):
  xrs = model.get_xray_structure()
  f_obs = abs(xrs.structure_factors(d_min=2.0, algorithm="direct").f_calc())
  flags = f_obs.generate_r_free_flags(fraction=0.1)
  return f_obs, flags

def exercise_curvatures_work(model, f_obs, flags):
  for target_name in ["ml", "ls_wunit_k1"]:
    fmodel = mmtbx.f_model.manager(f_obs=f_obs, r_free_flags=flags,
      xray_structure=model.get_xray_structure(), target_name=target_name)
    fmodel.update_all_scales(remove_outliers=False)
    tr = fmodel.target_functor()(compute_gradients=False)
    c = tr.curvatures_work()
    assert c.size() == fmodel.f_obs_work().size()
    assert c.all_ge(0)
    assert flex.max(c) > 0
    k = fmodel.k_anisotropic_work() * fmodel.k_isotropic_work()
    if (target_name == "ml"):
      core = tr.core
      expected = xray.targets_mlf_expected_curvatures_work(
        f_obs=core.f_obs.data(), r_free_flags=core.r_free_flags.data(),
        f_calc=fmodel.f_model().data(), alpha=core.alpha_beta[0].data(),
        beta=core.alpha_beta[1].data(), scale_factor=core.scale_factor,
        epsilons=core.epsilons, centric_flags=core.centric_flags)
    else:
      expected = xray.targets_least_squares_curvatures_work(
        obs_type="F", obs=f_obs.data(), weights=flex.double(f_obs.size(), 1),
        r_free_flags=flags.data(), f_calc=fmodel.f_model().data(),
        scale_factor=tr.core_result.scale_factor())
    assert approx_equal(c, expected*k*k)

def run_minimization(model, f_obs, flags, refine, use_curvatures, w=None):
  model = model.deep_copy()
  xrs = model.get_xray_structure()
  iso_restraints = None
  if (refine == "xyz"):
    xray.set_scatterer_grad_flags(scatterers=xrs.scatterers(), site=True)
    if (w is None): w = 1
  else:
    xray.set_scatterer_grad_flags(scatterers=xrs.scatterers(), u_iso=True)
    if (w is None): w = 0
    iso_restraints = mmtbx.refinement.adp_refinement.\
      adp_restraints_master_params.extract().iso
    model.set_refinement_flags(mmtbx.refinement.refinement_flags.manager(
      individual_adp=True,
      adp_individual_iso=flex.bool(xrs.scatterers().size(), True)))
  w = weights.weights(wx=1, wx_scale=1, w=w)
  fmodel = mmtbx.f_model.manager(f_obs=f_obs, r_free_flags=flags,
    xray_structure=xrs, target_name="ml")
  fmodel.update_all_scales(remove_outliers=False)
  r_start = fmodel.r_work()
  m = minimization.lbfgs(
    fmodels=mmtbx.fmodels(fmodel_xray=fmodel),
    restraints_manager=model.restraints_manager,
    model=model,
    target_weights=target_weights(w),
    refine_xyz=(refine == "xyz"),
    refine_adp=(refine == "adp"),
    iso_restraints=iso_restraints,
    h_params=group_args(refine="riding"),
    lbfgs_termination_params=scitbx.lbfgs.termination_parameters(
      max_iterations=50),
    collect_monitor=False,
    log=null_out(),
    use_curvatures=use_curvatures)
  if (use_curvatures):
    assert m.curvatures.size() == xrs.n_parameters()
    assert m.curvatures.all_gt(0)
  else:
    assert m.curvatures is None
  return r_start, fmodel.r_work(), m

def run():
  random.seed(0)
  flex.set_random_seed(0)
  model = get_model()
  f_obs, flags = get_data(model)
  exercise_curvatures_work(model, f_obs, flags)
  xrs = model.get_xray_structure()
  xrs.shake_sites_in_place(rms_difference=0.3)
  model.set_xray_structure(xrs)
  results = {}
  for use_curvatures in [False, True]:
    r_start, r_final, m = run_minimization(
      model, f_obs, flags, "xyz", use_curvatures)
    assert r_final < 0.5 * r_start, (r_start, r_final)
    results[use_curvatures] = r_final
  assert results[True] < 1.5 * results[False]
  model = get_model()
  xrs = model.get_xray_structure()
  xrs.set_b_iso(value=20)
  model.set_xray_structure(xrs)
  for use_curvatures in [False, True]:
    r_start, r_final, m = run_minimization(
      model, f_obs, flags, "adp", use_curvatures)
    assert r_final < 0.5 * r_start, (r_start, r_final)
  # with ADP restraints: their curvatures are added to the X-ray curvatures
  for use_curvatures in [False, True]:
    r_start, r_final, m = run_minimization(
      model, f_obs, flags, "adp", use_curvatures, w=1)
    assert r_final < 0.5 * r_start, (r_start, r_final)
  restraints_part = m.curvatures - m.xray_curvatures()
  assert restraints_part.all_ge(0)
  assert flex.max(restraints_part) > 0
  exercise_simultaneous(f_obs, flags)
  exercise_smooth_target(f_obs, flags)
  exercise_aniso_gradients()
  print("OK")

def exercise_aniso_gradients():
  """
  Gradients of the target (X-ray + anisotropic ADP restraints) with respect
  to the u_cart parameters vs. finite differences (direct summation, since
  finite differences of FFT structure factors are not accurate enough).
  """
  from cctbx import adptbx
  model = get_model()
  f_obs, flags = get_data(model)
  xrs = model.get_xray_structure()
  xrs.convert_to_anisotropic()
  uc = xrs.unit_cell()
  for sc in xrs.scatterers():
    u = list(adptbx.u_star_as_u_cart(uc, sc.u_star))
    u = [u[0]*random.uniform(0.7,1.3), u[1]*random.uniform(0.7,1.3),
         u[2]*random.uniform(0.7,1.3), u[3]+random.uniform(-0.02,0.02),
         u[4]+random.uniform(-0.02,0.02), u[5]+random.uniform(-0.02,0.02)]
    sc.u_star = adptbx.u_cart_as_u_star(uc, u)
  model.set_xray_structure(xrs)
  sfg = mmtbx.f_model.sf_and_grads_accuracy_master_params.extract()
  sfg.algorithm = "direct"
  fmodel = mmtbx.f_model.manager(f_obs=f_obs, r_free_flags=flags,
    xray_structure=model.get_xray_structure(), target_name="ls_wunit_k1",
    sf_and_grads_accuracy_params=sfg)
  fmodel.update_all_scales(remove_outliers=False)
  xrs = fmodel.xray_structure
  xray.set_scatterer_grad_flags(scatterers=xrs.scatterers(), u_aniso=True)
  n = xrs.scatterers().size()
  model.set_refinement_flags(mmtbx.refinement.refinement_flags.manager(
    individual_adp=True, adp_individual_aniso=flex.bool(n, True)))
  w = weights.weights(wx=1, wx_scale=1, w=1)
  for use_curvatures in [False, True]:
    m = minimization.lbfgs(
      fmodels=mmtbx.fmodels(fmodel_xray=fmodel),
      restraints_manager=model.restraints_manager,
      model=model,
      target_weights=group_args(xyz_weights_result=w, adp_weights_result=w),
      refine_adp=True,
      iso_restraints=mmtbx.refinement.adp_refinement.\
        adp_restraints_master_params.extract().iso,
      h_params=group_args(refine="riding"),
      lbfgs_termination_params=scitbx.lbfgs.termination_parameters(
        max_iterations=1),
      collect_monitor=False,
      log=null_out(),
      use_curvatures=use_curvatures)
    if (use_curvatures):
      assert m.curvatures.size() == 6*n
      assert m.curvatures.all_gt(0)
  m._scatterers_start = m.xray_structure.scatterers().deep_copy()
  x0 = flex.double(m.x.size(), 0)
  m.x = x0.deep_copy()
  f0, g0 = m.compute_functional_and_gradients()
  g0 = g0.deep_copy()
  eps = 1.e-5
  for i in [0, 1, 2, 3, 4, 5, 6, 9]:
    fs = []
    for e in [eps, -eps]:
      m.x = x0.deep_copy(); m.x[i] += e
      fs.append(m.compute_functional_and_gradients()[0])
    g_fd = (fs[0]-fs[1])/(2*eps)
    assert approx_equal(g0[i], g_fd, eps=1.e-4*max(1.e-4, abs(g_fd))), (
      i, g0[i], g_fd)

def exercise_simultaneous(f_obs, flags):
  # simultaneous refinement of coordinates and isotropic ADPs
  model0 = get_model()
  xrs = model0.get_xray_structure()
  xrs.shake_sites_in_place(rms_difference=0.3)
  xrs.set_b_iso(value=20)
  model0.set_xray_structure(xrs)
  w_xyz = weights.weights(wx=2, wx_scale=1, w=1)
  w_adp = weights.weights(wx=0.5, wx_scale=1, w=1)
  tw = group_args(xyz_weights_result=w_xyz, adp_weights_result=w_adp)
  for use_curvatures in [False, True]:
    model = model0.deep_copy()
    xrs = model.get_xray_structure()
    sites_start = xrs.sites_cart()
    xray.set_scatterer_grad_flags(scatterers=xrs.scatterers(), site=True,
      u_iso=True)
    model.set_refinement_flags(mmtbx.refinement.refinement_flags.manager(
      individual_sites=True, individual_adp=True,
      sites_individual=flex.bool(xrs.scatterers().size(), True),
      adp_individual_iso=flex.bool(xrs.scatterers().size(), True)))
    fmodel = mmtbx.f_model.manager(f_obs=f_obs, r_free_flags=flags,
      xray_structure=xrs, target_name="ml")
    fmodel.update_all_scales(remove_outliers=False)
    r_start = fmodel.r_work()
    m = minimization.lbfgs(
      fmodels=mmtbx.fmodels(fmodel_xray=fmodel),
      restraints_manager=model.restraints_manager,
      model=model,
      target_weights=tw,
      refine_xyz=True,
      refine_adp=True,
      iso_restraints=mmtbx.refinement.adp_refinement.\
        adp_restraints_master_params.extract().iso,
      h_params=group_args(refine="riding"),
      lbfgs_termination_params=scitbx.lbfgs.termination_parameters(
        max_iterations=50),
      collect_monitor=False,
      log=null_out(),
      use_curvatures=use_curvatures)
    # ADP restraints on the scale of the coordinate refinement target
    assert approx_equal(m.adp_restraints_weight, 2/0.5)
    assert fmodel.r_work() < 0.5 * r_start, (r_start, fmodel.r_work())
    # coordinate shifts are not truncated like ADP shifts
    shifts = flex.sqrt((fmodel.xray_structure.sites_cart()-sites_start).dot())
    assert flex.max(shifts) > 0.1
    b = fmodel.xray_structure.extract_u_iso_or_u_equiv()
    assert flex.max(b) - flex.min(b) > 0

class probing_lbfgs(minimization.lbfgs):
  """
  Records the state of the restraints during the minimization, and the
  effect of apply_shifts() on large negative u_iso shifts.
  """
  def compute_functional_and_gradients(self, compute_curvatures=False):
    base = minimization.lbfgs.compute_functional_and_gradients
    if (not hasattr(self, "probe")):
      self.probe = group_args(
        plain_pairs_radius = self.restraints_manager.geometry.plain_pairs_radius)
      # u_iso shift far below u_min (B -5): truncated unless smooth_target
      x0 = self.x.deep_copy()
      all_ = flex.bool(x0.size(), True)
      i_u = self.u_iso_parameter_selection().iselection()[0]
      self.x[i_u] = adptbx.b_as_u(-20)
      self.apply_shifts()
      self.probe.u_shift = self.x[i_u]
      self.x.set_selected(all_, x0)
    return base(self, compute_curvatures=compute_curvatures)

def exercise_smooth_target(f_obs, flags):
  """smooth_target=True: see mmtbx.refinement.minimization.lbfgs"""
  model0 = get_model()
  xrs = model0.get_xray_structure()
  xrs.shake_sites_in_place(rms_difference=0.3)
  xrs.set_b_iso(value=40)
  model0.set_xray_structure(xrs)
  w = weights.weights(wx=1, wx_scale=1, w=1)
  tw = group_args(xyz_weights_result=w, adp_weights_result=w)
  for smooth_target in [False, True]:
    model = model0.deep_copy()
    xrs = model.get_xray_structure()
    xray.set_scatterer_grad_flags(scatterers=xrs.scatterers(), site=True,
      u_iso=True)
    model.set_refinement_flags(mmtbx.refinement.refinement_flags.manager(
      individual_sites=True, individual_adp=True,
      sites_individual=flex.bool(xrs.scatterers().size(), True),
      adp_individual_iso=flex.bool(xrs.scatterers().size(), True)))
    fmodel = mmtbx.f_model.manager(f_obs=f_obs, r_free_flags=flags,
      xray_structure=xrs, target_name="ml")
    fmodel.update_all_scales(remove_outliers=False)
    geometry = model.restraints_manager.geometry
    plain_pairs_radius = geometry.plain_pairs_radius
    assert plain_pairs_radius is not None
    r_start = fmodel.r_work()
    m = probing_lbfgs(
      fmodels=mmtbx.fmodels(fmodel_xray=fmodel),
      restraints_manager=model.restraints_manager,
      model=model,
      target_weights=tw,
      refine_xyz=True,
      refine_adp=True,
      iso_restraints=mmtbx.refinement.adp_refinement.\
        adp_restraints_master_params.extract().iso,
      h_params=group_args(refine="riding"),
      lbfgs_termination_params=scitbx.lbfgs.termination_parameters(
        max_iterations=20),
      collect_monitor=False,
      log=null_out(),
      use_curvatures=True,
      smooth_target=smooth_target)
    assert fmodel.r_work() < r_start
    # restored after the minimization
    assert geometry.plain_pairs_radius == plain_pairs_radius
    if (smooth_target):
      # the pair table of the ADP restraints is not updated during the
      # minimization; the normalization is fixed; no truncation of shifts
      assert m.probe.plain_pairs_radius is None
      assert sorted(m._fixed_normalization.keys()) == ["adp", "geometry"]
      # (the u_iso shifts are truncated in both cases)
      assert approx_equal(m.probe.u_shift, m.u_min)
      e = group_args(normalization_factor =
        m._fixed_normalization["geometry"] * 2)
      assert approx_equal(m._normalization_scale("geometry", e), 0.5)
    else:
      assert m.probe.plain_pairs_radius == plain_pairs_radius
      assert approx_equal(m.probe.u_shift, m.u_min)
      assert m._normalization_scale("geometry",
        group_args(normalization_factor=1)) == 1

if (__name__ == "__main__"):
  run()
