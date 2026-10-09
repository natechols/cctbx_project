"""
Tests for mmtbx.refinement.group_curvatures: approximate curvatures of the
X-ray target with respect to group parameters (group ADP, group occupancy,
rigid-body translation and rotation) vs. exact Gauss-Newton curvatures from
finite differences of directly summed structure factors, and curvature
preconditioning in mmtbx.refinement.group.
"""
from __future__ import absolute_import, division, print_function
from cctbx import xray, adptbx
from cctbx.development import random_structure
from cctbx.array_family import flex
from scitbx import matrix
from libtbx.test_utils import approx_equal
from libtbx.utils import null_out
import mmtbx.f_model
import mmtbx.refinement.group
from mmtbx.refinement.group_curvatures import group_curvatures
import math, random

def make_structure(space_group_symbol, n_atoms=120, n_aniso=0):
  xs = random_structure.xray_structure(
    space_group_symbol=space_group_symbol,
    elements=(["C"]*6+["N"]*2+["O"]*2)*(n_atoms//10),
    volume_per_atom=20, min_distance=1.3,
    general_positions_only=True, random_u_iso=False)
  for i, sc in enumerate(xs.scatterers()):
    sc.u_iso = adptbx.b_as_u(random.uniform(10, 40))
    if (i < n_aniso):
      u_cart = adptbx.random_u_cart(u_scale=sc.u_iso, u_min=0.5*sc.u_iso)
      sc.u_star = xs.space_group().average_u_star(
        adptbx.u_cart_as_u_star(xs.unit_cell(), u_cart))
      sc.set_use_u_aniso_only()
  # plain xray.structure (random_structure.xray_structure has an attribute
  # use_u_aniso)
  return xray.structure(special_position_settings=xs,
    scatterers=xs.scatterers())

def get_fmodel(xs, d_min=2.0):
  # no bulk solvent, unit scales: f_model = f_calc
  f_obs = abs(xs.structure_factors(d_min=d_min, algorithm="direct").f_calc())
  f_obs = f_obs.customized_copy(data=f_obs.data() *
    flex.double([random.uniform(0.7, 1.3) for i in range(f_obs.size())]))
  flags = f_obs.generate_r_free_flags(fraction=0.05)
  sfg = mmtbx.f_model.sf_and_grads_accuracy_master_params.extract()
  sfg.algorithm = "direct"
  return mmtbx.f_model.manager(f_obs=f_obs, r_free_flags=flags,
    xray_structure=xs, target_name="ls_wunit_k1",
    sf_and_grads_accuracy_params=sfg)

class exact_gauss_newton(object):
  """sum_h c_h (d|F_calc|/dp)^2 for shifted structures."""
  def __init__(self, fmodel):
    self.ms = fmodel.f_obs_work()
    self.c = fmodel.target_functor()(compute_gradients=False).\
      curvatures_work()
    self.xs = fmodel.xray_structure
    self.f0 = self.f_calc(self.xs)
    self.a0 = flex.abs(self.f0)

  def f_calc(self, xs):
    return self.ms.structure_factors_from_scatterers(xs,
      algorithm="direct").f_calc().data()

  def __call__(self, shift, eps):
    """shift(xs, e) returns a copy of xs with the parameter shifted by e"""
    df = (self.f_calc(shift(self.xs, eps)) - self.f_calc(shift(self.xs, -eps)))\
       / (2*eps)
    da = (flex.real(df)*flex.real(self.f0) + flex.imag(df)*flex.imag(self.f0))\
       / self.a0
    return flex.sum(self.c * da * da)

def compact_groups(xs):
  """groups of the atoms closest to atom 0 (5% to 100% of the atoms)"""
  sites = xs.sites_cart()
  perm = flex.sort_permutation((sites - sites[0]).norms())
  n = sites.size()
  return [perm[:max(3, int(f*n))] for f in [0.05, 0.3, 0.6, 1.0]]

verbose = False
# |log(approx/exact)|: the approximations are within a factor of ~1.6
# (e.g. low-resolution intensities of the random structures deviate from
# Wilson statistics; small, compact groups are partly coherent)
TOLERANCE = 0.5
def check(label, approx, exact, tolerance):
  ratio = approx / exact
  if (verbose): print("  %-12s approx/exact %.2f" % (label, ratio))
  assert abs(math.log(ratio)) < tolerance, (label, approx, exact, ratio)

def exercise_curvatures(space_group_symbol, n_aniso=0):
  xs = make_structure(space_group_symbol, n_aniso=n_aniso)
  fmodel = get_fmodel(xs)
  gc = group_curvatures(fmodel=fmodel, reflection_stride=1)
  exact = exact_gauss_newton(fmodel)
  groups = compact_groups(xs)
  eps = 1.e-4
  c_b = gc.group_u_iso(groups)
  c_occ = gc.group_occupancy(groups)
  for i_group, sel in enumerate(groups):
    def shift_b(x, e):
      x = x.deep_copy_scatterers()
      for i in sel:
        sc = x.scatterers()[i]
        if (sc.flags.use_u_aniso()):
          sc.u_star = adptbx.u_cart_as_u_star(x.unit_cell(), [u + d for u, d
            in zip(adptbx.u_star_as_u_cart(x.unit_cell(), sc.u_star),
              (e,e,e,0,0,0))])
        else:
          sc.u_iso += e
      return x
    check("B", c_b[i_group], exact(shift_b, eps), TOLERANCE)
    def shift_occ(x, e):
      x = x.deep_copy_scatterers()
      for i in sel: x.scatterers()[i].occupancy += e
      return x
    check("occupancy", c_occ[i_group], exact(shift_occ, eps), TOLERANCE)
    c_t = gc.group_translation(sel)
    for k in range(3):
      def shift_t(x, e):
        x = x.deep_copy_scatterers()
        s = x.sites_cart()
        d = [0,0,0]; d[k] = e
        s.set_selected(sel, s.select(sel) + d)
        x.set_sites_cart(s)
        return x
      e_t = exact(shift_t, eps)
      if (e_t < 1.e-6 * flex.sum(gc.atoms.sites_cart.as_double())):
        # translation of everything in P1: phases only
        assert c_t[k] < 1.e-6 * abs(e_t + 1) + 1.e-9 * flex.sum(
          gc.atoms.sites_cart.as_double()), (k, c_t[k], e_t)
      else:
        check("translation", c_t[k], e_t, TOLERANCE)
    center = xs.sites_cart().select(sel).mean()
    c_r = gc.group_rotation(sel, center)
    for k in range(3):
      axis = [0,0,0]; axis[k] = 1
      def shift_r(x, e):
        x = x.deep_copy_scatterers()
        s = x.sites_cart()
        r = matrix.col(axis).axis_and_angle_as_r3_rotation_matrix(e)
        c = matrix.col(center)
        for i in sel:
          s[i] = r * (matrix.col(s[i]) - c) + c
        x.set_sites_cart(s)
        return x
      check("rotation", c_r[k], exact(shift_r, eps), TOLERANCE)

def exercise_group_refinement():
  xs = make_structure("P212121", n_atoms=150)
  fmodel = get_fmodel(xs, d_min=2.5)
  # groups of different sizes (e.g. side chains and whole residues)
  groups = []
  i = 0
  for size in [1, 2, 4, 8, 16, 32, 3, 6, 12, 24, 42]:
    groups.append(flex.size_t(range(i, i+size)))
    i += size
  assert i == 150
  results = {}
  for refine_adp, refine_occ in [(True, False), (False, True)]:
    shifts = [random.uniform(-0.1, 0.1) for sel in groups]
    for use_curvatures in [False, True]:
      xs_start = xs.deep_copy_scatterers()
      for sel, d in zip(groups, shifts):
        for i in sel:
          if (refine_adp): xs_start.scatterers()[i].u_iso += d
          else: xs_start.scatterers()[i].occupancy += 2*d
      fm = fmodel.deep_copy()
      fm.update_xray_structure(xray_structure=xs_start, update_f_calc=True)
      r_start = fm.r_work()
      m = mmtbx.refinement.group.manager(
        fmodel                   = fm,
        selections               = groups,
        max_number_of_iterations = 50,
        number_of_macro_cycles   = 2,
        refine_adp               = refine_adp,
        refine_occ               = refine_occ,
        log                      = null_out(),
        occupancy_max            = 2,
        occupancy_min            = 0,
        use_curvatures           = use_curvatures)
      assert fm.r_work() < r_start
      if (refine_adp): p = fm.xray_structure.extract_u_iso_or_u_equiv()
      else: p = fm.xray_structure.scatterers().extract_occupancies()
      results[(refine_adp, use_curvatures)] = (fm.r_work(), p)
    # both minimizations converge to the same parameters
    (r0, p0), (r1, p1) = results[(refine_adp, False)], \
      results[(refine_adp, True)]
    assert approx_equal(r0, r1, eps=1.e-3), (refine_adp, r0, r1)
    # the same group ADPs (the group occupancies of this noisy data set
    # are poorly determined: the target is flat along some directions)
    if (refine_adp):
      assert flex.max(flex.abs(p0 - p1)) < 1.e-3, flex.max(flex.abs(p0 - p1))

def rigid_body_minimizer_stub(fmodel, selections, convention):
  """rigid_body_minimizer at zero shifts, without minimization"""
  import mmtbx.refinement.rigid_body as rb
  m = rb.rigid_body_minimizer.__new__(rb.rigid_body_minimizer)
  m.fmodel = fmodel
  m.fmodel_copy = fmodel.deep_copy()
  m.selections = selections
  m.refine_r = m.refine_t = True
  m.euler_angle_convention = convention
  m.atomic_weights = fmodel.xray_structure.atomic_weights()
  m.sites_cart = fmodel.xray_structure.sites_cart()
  m.sites_frac = fmodel.xray_structure.sites_frac()
  m.n_groups = len(selections)
  m.dim_r = m.dim_t = 3
  m.r_min = [(0,0,0)] * len(selections)
  m.t_min = [(0,0,0)] * len(selections)
  return m

def exercise_rigid_body_curvatures(space_group_symbol, convention):
  """chain rule for the Euler angles and translations"""
  import mmtbx.refinement.rigid_body as rb
  import scitbx.rigid_body
  xs = make_structure(space_group_symbol, n_atoms=150)
  fmodel = get_fmodel(xs)
  sites = xs.sites_cart()
  perm = flex.sort_permutation((sites - sites[0]).norms())
  selections = [perm[:20], perm[20:70], perm[70:]]
  m = rigid_body_minimizer_stub(fmodel, selections, convention)
  c = m.get_curvatures()
  assert c.size() == 18
  exact = exact_gauss_newton(fmodel)
  x0 = m.pack(m.r_min, m.t_min)
  for i_par in range(18):
    def shift(x, e):
      xp = x0.deep_copy(); xp[i_par] += e
      m.x = xp
      m.unpack_x()
      rots = [scitbx.rigid_body.euler(phi=r[0], psi=r[1], the=r[2],
        convention=convention).rot_mat() for r in m.r_min]
      new_sites_frac = rb.apply_transformation_(
        xray_structure=x, sites_cart=m.sites_cart, sites_frac=m.sites_frac,
        rotation_matrices=rots, translation_vectors=m.t_min,
        selections=selections, atomic_weights=m.atomic_weights)[0]
      result = x.deep_copy_scatterers()
      result.set_sites_frac(new_sites_frac)
      return result
    # Euler angles in degrees
    check("rigid body %d" % i_par, c[i_par], exact(shift, 1.e-3), TOLERANCE)

def exercise_rigid_body_refinement(number_of_zones=None, n_atoms=150,
      max_iterations=100):
  import mmtbx.refinement.rigid_body as rb
  import scitbx.rigid_body
  # no floating origin: the distance to the true model is meaningful
  xs = make_structure("P212121", n_atoms=n_atoms)
  fmodel = get_fmodel(xs, d_min=2.0)
  sites = xs.sites_cart()
  perm = flex.sort_permutation((sites - sites[0]).norms())
  selections = [perm[:30], perm[30:80], perm[80:]]
  # shifted start: rotations of up to 3 degrees, translations up to 0.4 A
  xs_start = xs.deep_copy_scatterers()
  rots, trans = [], []
  for sel in selections:
    rots.append(scitbx.rigid_body.euler(
      phi=random.uniform(-3,3), psi=random.uniform(-3,3),
      the=random.uniform(-3,3), convention="xyz").rot_mat())
    trans.append([random.uniform(-0.4, 0.4) for k in range(3)])
  xs_start = rb.apply_transformation(xray_structure=xs_start,
    rotation_matrices=rots, translation_vectors=trans,
    selections=selections)
  params = rb.master_params.extract()
  params.target = "ls_wunit_k1"
  params.bulk_solvent_and_scale = False
  if (number_of_zones is not None):
    params.number_of_zones = number_of_zones
  if (max_iterations is not None):
    params.max_iterations = max_iterations
  results = {}
  for use_curvatures in [False, True]:
    fm = fmodel.deep_copy()
    fm.update_xray_structure(xray_structure=xs_start.deep_copy_scatterers(),
      update_f_calc=True)
    r_start = fm.r_work()
    n_eval = [0]
    orig = rb.rigid_body_minimizer.compute_functional_and_gradients
    def counting(self, *args, **kwds):
      n_eval[0] += 1
      return orig(self, *args, **kwds)
    rb.rigid_body_minimizer.compute_functional_and_gradients = counting
    try:
      rb.manager(fmodel=fm, selections=selections, params=params,
        log=null_out(), use_curvatures=use_curvatures)
    finally:
      rb.rigid_body_minimizer.compute_functional_and_gradients = orig
    results[use_curvatures] = (fm.r_work(), n_eval[0],
      fm.xray_structure.sites_cart().rms_difference(xs.sites_cart()))
    assert fm.r_work() < 0.5 * r_start, (use_curvatures, r_start, fm.r_work())
  if (verbose): print("rigid body (r_work, evaluations, rmsd)", results)
  # both converge to the same model; with curvatures in fewer evaluations
  # (rotations in degrees and translations in A are poorly scaled)
  assert approx_equal(results[False][0], results[True][0], eps=0.002)
  assert approx_equal(results[False][2], results[True][2], eps=0.01)
  assert results[True][1] < results[False][1], results
  return results

def tls_structure(space_group_symbol, n_atoms=150):
  """random structure with ADPs from TLS (two groups)"""
  from mmtbx.tls import tools
  xs = make_structure(space_group_symbol, n_atoms=n_atoms)
  sites = xs.sites_cart()
  perm = flex.sort_permutation((sites - sites[0]).norms())
  selections = [perm[:50], perm[50:]]
  T = [[0.15, 0.20, 0.25, 0.02, -0.01, 0.03], [0.30, 0.25, 0.20, 0, 0.02, 0]]
  L = [[2.0, 3.0, 1.5, 0.3, 0, -0.2], [1.0, 1.5, 2.5, 0, 0.2, 0.1]]
  S = [[0.02, 0.01, 0, 0.01, -0.03, 0.02, 0, 0.01, 0.01],
       [0, 0.02, 0.01, -0.01, 0.02, 0, 0.01, 0, -0.02]]
  tlsos = tools.generate_tlsos(selections=selections, xray_structure=xs,
    T=T, L=L, S=S)
  xs.convert_to_anisotropic()
  xs.set_u_cart(u_cart=tools.u_cart_from_tls(sites_cart=sites,
    selections=selections, tlsos=tlsos))
  return xs, selections, tlsos

def tls_minimizer_stub(fmodel, selections, tlsos):
  """tls_xray_target_minimizer at the given TLS, without minimization"""
  from mmtbx.tls import tools
  m = tools.tls_xray_target_minimizer.__new__(tools.tls_xray_target_minimizer)
  m.fmodel = fmodel
  m.fmodel_copy = fmodel.deep_copy()
  m.selections = selections
  m.selections_1d = flex.bool(fmodel.xray_structure.scatterers().size(),
    True).iselection()
  m.correct_adp = False
  m.refine_T = m.refine_L = m.refine_S = True
  m.T_min = [t.t for t in tlsos]
  m.L_min = [t.l for t in tlsos]
  m.S_min = [t.s for t in tlsos]
  m.n_groups = len(tlsos)
  m.dim_T, m.dim_L, m.dim_S = 6, 6, 9
  return m

def exercise_tls_curvatures(space_group_symbol):
  from mmtbx.tls import tools
  xs, selections, tlsos = tls_structure(space_group_symbol)
  fmodel = get_fmodel(xs)
  m = tls_minimizer_stub(fmodel, selections, tlsos)
  c = m.get_curvatures()
  assert c.size() == 42
  exact = exact_gauss_newton(fmodel)
  x0 = m.pack(m.T_min, m.L_min, m.S_min)
  sites = xs.sites_cart()
  ratios = []
  for i_par in range(42):
    def shift(x, e):
      m.x = x0.deep_copy(); m.x[i_par] += e
      m.unpack_x()
      t = tools.generate_tlsos(selections=selections, xray_structure=x,
        T=m.T_min, L=m.L_min, S=m.S_min)
      result = x.deep_copy_scatterers()
      result.set_u_cart(u_cart=tools.u_cart_from_tls(sites_cart=sites,
        selections=selections, tlsos=t))
      return result
    e = exact(shift, 1.e-4)
    if (i_par % 21 in [12, 16, 20]):
      # S11, S22, S33: only differences matter (trace(S) = 0)
      continue
    # the TLS approximations are within a factor of ~2
    check("tls %d" % i_par, c[i_par], e, 0.8)
    ratios.append(c[i_par] / e)
  if (verbose):
    print("TLS approx/exact:", " ".join("%.2f" % r for r in ratios))

def exercise_tls_refinement(max_iterations=25):
  from mmtbx.tls import tools
  xs, selections, tlsos = tls_structure("P212121")
  fmodel = get_fmodel(xs, d_min=2.0)
  # start: isotropic T and L, S = 0
  T0 = [[0.2]*3 + [0]*3 for t in tlsos]
  L0 = [[1.0]*3 + [0]*3 for t in tlsos]
  S0 = [[0]*9 for t in tlsos]
  results = {}
  for use_curvatures in [False, True]:
    fm = fmodel.deep_copy()
    tlsos_start = tools.generate_tlsos(selections=selections,
      xray_structure=fm.xray_structure, T=[list(t) for t in T0],
      L=[list(l) for l in L0], S=[list(s) for s in S0])
    tools.update_xray_structure_with_tls(xray_structure=fm.xray_structure,
      selections=selections, tlsos=tlsos_start)
    fm.update_xray_structure(update_f_calc=True)
    r_start = fm.r_work()
    n_eval = [0]
    orig = tools.tls_xray_target_minimizer.compute_functional_and_gradients
    def counting(self):
      n_eval[0] += 1
      return orig(self)
    tools.tls_xray_target_minimizer.compute_functional_and_gradients = counting
    try:
      m = tools.tls_xray_target_minimizer(fmodel=fm,
        tlsos_initial=tlsos_start, refine_T=True, refine_L=True,
        refine_S=True, selections=selections,
        selections_1d=flex.bool(xs.scatterers().size(), True).iselection(),
        max_iterations=max_iterations, use_curvatures=use_curvatures)
    finally:
      tools.tls_xray_target_minimizer.compute_functional_and_gradients = orig
    results[use_curvatures] = (m.fmodel_copy.r_work(), n_eval[0],
      flex.max(flex.abs(m.fmodel_copy.xray_structure.extract_u_iso_or_u_equiv()
        - xs.extract_u_iso_or_u_equiv())))
    assert m.fmodel_copy.r_work() < r_start
  if (verbose): print("TLS (r_work, evaluations, max |u_eq error|)", results)
  # with the same number of evaluations (the minimizer does not stop
  # earlier without curvatures), a lower target with curvatures
  assert results[True][0] < results[False][0], results
  return results

def run():
  random.seed(0)
  flex.set_random_seed(0)
  for sg in ["P1", "P-1", "P21", "P212121"]:
    exercise_curvatures(sg)
  exercise_curvatures("P21", n_aniso=60)
  exercise_group_refinement()
  for sg, convention in [("P1", "xyz"), ("P21", "zyz"), ("P212121", "xyz")]:
    exercise_rigid_body_curvatures(sg, convention)
  exercise_rigid_body_refinement()
  for sg in ["P1", "P21", "P212121"]:
    exercise_tls_curvatures(sg)
  exercise_tls_refinement()
  print("OK")

if (__name__ == "__main__"):
  run()
