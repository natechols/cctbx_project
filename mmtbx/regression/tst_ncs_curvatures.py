"""
Tests for the curvatures (diagonal 2nd derivatives) of the Cartesian and
torsion NCS restraints, which are used for preconditioning in
mmtbx.refinement.minimization.lbfgs (use_curvatures=True).
"""
from __future__ import absolute_import, division, print_function
import iotbx.pdb
import mmtbx.model
from mmtbx.regression.tst_minimization_curvatures import pdb_str
from mmtbx.ncs.ncs_params import global_ncs_params
import cctbx.geometry_restraints
from cctbx import crystal
from cctbx.array_family import flex
from scitbx import matrix
from libtbx.test_utils import approx_equal
from libtbx.utils import null_out
import math, random

def ncs_pdb_str(n_copies=3, shake=0.2, b_spread=5.0):
  """Copies of the chain of pdb_str related by a rotation about an axis
  through the center of the chain, with random perturbations."""
  ph = iotbx.pdb.input(source_info=None, lines=pdb_str).construct_hierarchy()
  chain = ph.only_chain()
  center = matrix.col(chain.atoms().extract_xyz().mean())
  model = ph.only_model()
  for i_copy in range(1, n_copies):
    c = chain.detached_copy()
    c.id = "ABCDEFGH"[i_copy]
    r = matrix.col((0,0,1)).axis_and_angle_as_r3_rotation_matrix(
      angle=360/n_copies*i_copy, deg=True)
    t = matrix.col((0, 0, 0)) + 12 * matrix.col((math.cos(i_copy),
      math.sin(i_copy), 0))
    for atom in c.atoms():
      site = r * (matrix.col(atom.xyz) - center) + center + t
      atom.xyz = site + matrix.col([random.uniform(-shake, shake)
        for k in range(3)])
      atom.b = max(2, atom.b + random.uniform(-b_spread, b_spread))
    model.append_chain(c)
  return ph.as_pdb_string(crystal_symmetry=crystal.symmetry(
    unit_cell=(80, 80, 60, 90, 90, 90), space_group_symbol="P1"))

def get_model(pdb_text):
  model = mmtbx.model.manager(
    model_input=iotbx.pdb.input(source_info=None, lines=pdb_text),
    log=null_out())
  model.process(make_restraints=True)
  model.search_for_ncs()
  return model

def ncs_params(b_factor_weight=10):
  p = global_ncs_params.extract().ncs
  p.b_factor_weight = b_factor_weight
  return p

def exercise_cartesian_sites(model):
  model.setup_cartesian_ncs_groups(ncs_params=ncs_params())
  ncs_manager = model.get_restraints_manager().cartesian_ncs_manager
  assert ncs_manager is not None and ncs_manager.get_n_groups() == 1
  sites_cart = model.get_sites_cart()
  e = ncs_manager.energies_sites(sites_cart=sites_cart,
    compute_curvatures=True)
  assert e.curvatures.size() == sites_cart.size()
  # the NCS operators are recomputed from the sites in energies_sites;
  # the curvatures are for fixed operators
  def target(sites):
    return sum(g.energies_sites(sites_cart=sites,
      coordinate_sigma=ncs_manager.ncs_params.coordinate_sigma,
      compute_gradients=False).residual_sum
      for g in ncs_manager.groups_list)
  eps = 1.e-3
  f0 = target(sites_cart)
  n_checked = 0
  for i_seq in range(0, sites_cart.size(), 7):
    for k in range(3):
      fs = []
      for e_k in [eps, -eps]:
        s = sites_cart.deep_copy()
        x = list(s[i_seq]); x[k] += e_k; s[i_seq] = x
        fs.append(target(s))
      c_fd = (fs[0] - 2*f0 + fs[1]) / eps**2
      # exact: the residual is quadratic in the sites
      assert approx_equal(e.curvatures[i_seq][k], c_fd,
        eps=1.e-4*max(1, c_fd)), (i_seq, k, e.curvatures[i_seq][k], c_fd)
      n_checked += 1
  assert n_checked > 20
  # every atom is in all 3 copies: 2*w*(1-1/3)
  w = 1/ncs_manager.ncs_params.coordinate_sigma**2
  assert approx_equal(e.curvatures.as_double(),
    flex.double(sites_cart.size()*3, 2*w*2/3))
  # plumbing: mmtbx.restraints.manager adds the NCS curvatures
  rm = model.get_restraints_manager()
  rm.normalization = False
  e_all = rm.energies_sites(sites_cart=sites_cart, compute_gradients=True,
    compute_curvatures=True)
  rm.cartesian_ncs_manager = None
  e_geo = rm.energies_sites(sites_cart=sites_cart, compute_gradients=True,
    compute_curvatures=True)
  rm.cartesian_ncs_manager = ncs_manager
  assert approx_equal(e_all.curvatures.as_double(),
    e_geo.curvatures.as_double() + e.curvatures.as_double())

def gauss_newton_adp(groups_i_seqs, u_isos, weight, average_power, eps=1.e-6):
  """2 sum_k (dr_k/du_j)^2 for r_k = sqrt(w) (u_k - u_ave) / u_ave^(p/2),
  with the derivatives by finite differences."""
  result = flex.double(u_isos.size(), 0)
  def residuals(u):
    u_ave = flex.mean(u)
    return math.sqrt(weight) * (u - u_ave) / u_ave**(average_power/2)
  for i_seqs in groups_i_seqs:
    u = u_isos.select(i_seqs)
    for j in range(u.size()):
      up = u.deep_copy(); up[j] += eps
      um = u.deep_copy(); um[j] -= eps
      d = (residuals(up) - residuals(um)) / (2*eps)
      result[i_seqs[j]] += 2 * flex.sum(d*d)
  return result

def exercise_cartesian_adp(model):
  model.setup_cartesian_ncs_groups(ncs_params=ncs_params(b_factor_weight=10))
  ncs_manager = model.get_restraints_manager().cartesian_ncs_manager
  u_isos = model.get_xray_structure().extract_u_iso_or_u_equiv()
  group = ncs_manager.groups_list[0]
  pairs = group.selection_pairs
  groups_i_seqs = []
  for i in range(pairs[0][0].size()):
    i_seqs = flex.size_t([pairs[0][0][i]])
    for pair in pairs:
      assert pair[0][i] == i_seqs[0]
      i_seqs.append(pair[1][i])
    groups_i_seqs.append(i_seqs)
  for average_power in [0, 1, 2]:
    e = ncs_manager.energies_adp_iso(u_isos=u_isos,
      average_power=average_power, compute_curvatures=True)
    expected = gauss_newton_adp(groups_i_seqs, u_isos,
      weight=ncs_manager.ncs_params.b_factor_weight,
      average_power=average_power)
    assert approx_equal(e.curvatures, expected,
      eps=1.e-5*flex.max(expected)), average_power
    assert e.curvatures.all_gt(0)
    # the curvatures do not change the target and gradients
    e0 = ncs_manager.energies_adp_iso(u_isos=u_isos,
      average_power=average_power)
    assert e0.curvatures is None
    assert approx_equal(e0.target, e.target)
    assert approx_equal(e0.gradients, e.gradients)
  # plumbing: mmtbx.restraints.manager.energies_adp_iso
  import mmtbx.refinement.adp_refinement
  params = mmtbx.refinement.adp_refinement.adp_restraints_master_params.\
    extract().iso
  rm = model.get_restraints_manager()
  rm.normalization = False
  xrs = model.get_xray_structure()
  kwds = dict(xray_structure=xrs, parameters=params, use_u_local_only=False,
    use_hd=False, compute_gradients=True, compute_curvatures=True)
  e_all = rm.energies_adp_iso(**kwds)
  rm.cartesian_ncs_manager = None
  e_geo = rm.energies_adp_iso(**kwds)
  rm.cartesian_ncs_manager = ncs_manager
  e_ncs = ncs_manager.energies_adp_iso(u_isos=u_isos,
    average_power=params.average_power, compute_curvatures=True)
  assert approx_equal(e_all.curvatures, e_geo.curvatures + e_ncs.curvatures)

def exercise_torsion(model):
  from mmtbx.geometry_restraints.torsion_restraints.torsion_ncs import \
    torsion_ncs
  geometry = model.get_restraints_manager().geometry
  geometry.ncs_dihedral_manager = torsion_ncs(
    model=model, fmodel=None, params=ncs_params().torsion, log=null_out())
  manager = geometry.ncs_dihedral_manager
  assert manager.get_n_proxies() > 0
  sites_cart = model.get_sites_cart()
  e_all = geometry.energies_sites(sites_cart=sites_cart,
    compute_gradients=True, compute_curvatures=True)
  geometry.ncs_dihedral_manager = None
  e_geo = geometry.energies_sites(sites_cart=sites_cart,
    compute_gradients=True, compute_curvatures=True)
  geometry.ncs_dihedral_manager = manager
  c = flex.vec3_double(sites_cart.size(), (0,0,0))
  g = flex.vec3_double(sites_cart.size(), (0,0,0))
  cctbx.geometry_restraints.dihedral_residual_sum(
    unit_cell=model.crystal_symmetry().unit_cell(),
    sites_cart=sites_cart,
    proxies=manager.ncs_dihedral_proxies,
    gradient_array=g,
    curvature_array=c)
  assert flex.max(c.as_double()) > 0
  assert approx_equal(e_all.curvatures.as_double(),
    e_geo.curvatures.as_double() + c.as_double())
  assert approx_equal(e_all.gradients.as_double(),
    e_geo.gradients.as_double() + g.as_double())

def run():
  random.seed(0)
  flex.set_random_seed(0)
  model = get_model(ncs_pdb_str())
  exercise_cartesian_sites(model)
  exercise_cartesian_adp(model)
  exercise_torsion(get_model(ncs_pdb_str()))
  print("OK")

if (__name__ == "__main__"):
  run()
