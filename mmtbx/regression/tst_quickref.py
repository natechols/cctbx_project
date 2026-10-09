"""Tests for mmtbx.quickref (mmtbx/programs/quickref.py)."""
from __future__ import absolute_import, division, print_function
import os
import random
import iotbx.pdb
from iotbx.cli_parser import run_program
from iotbx.data_manager import DataManager
from mmtbx.programs import quickref
from mmtbx.regression.tst_minimization_curvatures import pdb_str
from mmtbx.regression.tst_occupancy_curvatures import alt_conf_pdb_str
from cctbx.array_family import flex
from libtbx.utils import null_out, Sorry
from libtbx.test_utils import approx_equal

def make_inputs(prefix):
  pdb_inp = iotbx.pdb.input(source_info=None, lines=pdb_str)
  xrs = pdb_inp.xray_structure_simple()
  f_obs = abs(xrs.structure_factors(d_min=2.0, algorithm="direct").f_calc())
  flags = f_obs.generate_r_free_flags(fraction=0.1)
  mtz = f_obs.as_mtz_dataset(column_root_label="F")
  mtz.add_miller_array(flags, column_root_label="R-free-flags")
  data_file = prefix + "_data.mtz"
  mtz.mtz_object().write(data_file)
  # starting model: shaken coordinates, flat B
  hierarchy = pdb_inp.construct_hierarchy()
  xrs.shake_sites_in_place(rms_difference=0.3)
  xrs.set_b_iso(value=20)
  hierarchy.adopt_xray_structure(xrs)
  model_file = prefix + "_start.pdb"
  with open(model_file, "w") as f:
    f.write(hierarchy.as_pdb_string(crystal_symmetry=xrs.crystal_symmetry()))
  return model_file, data_file

def run_quickref(model_file, data_file, prefix, extra_args=[]):
  args = [model_file, data_file, "output.prefix=%s" % prefix,
    "output.overwrite=True"] + extra_args
  return run_program(program_class=quickref.Program, args=args,
    logger=null_out())

def exercise(model_file, data_file):
  for i, extra_args in enumerate([
        [],
        ["simultaneous=False"],
        ["use_curvatures=False", "number_of_macro_cycles=2"],
        ["refinement.adp=False", "write_map_coefficients=False"]]):
    prefix = "tst_quickref_%d" % i
    r = run_quickref(model_file, data_file, prefix, extra_args)
    assert r.r_free < r.r_free_start, (extra_args, r.r_free_start, r.r_free)
    assert r.r_work < 0.7 * r.r_work_start, (extra_args, r.r_work_start,
      r.r_work)
    assert os.path.isfile(r.model_file_name)
    model = iotbx.pdb.input(file_name=r.model_file_name)
    assert model.atoms().size() == 67
    if ("write_map_coefficients=False" in extra_args):
      assert r.mtz_file_name is None
    else:
      assert os.path.isfile(r.mtz_file_name)
      dm = DataManager(["miller_array"])
      dm.process_miller_array_file(r.mtz_file_name)
      labels = dm.get_miller_array_labels(r.mtz_file_name)
      assert "2FOFCWT,PH2FOFCWT" in labels, labels
      assert "FOFCWT,PHFOFCWT" in labels, labels

def exercise_hydrogens(data_file):
  # models with hydrogens are refused unless they are removed
  lines = pdb_str.splitlines()
  i = [k for k, l in enumerate(lines) if l.startswith("ATOM")][0]
  h = lines[i][:12] + " H  " + lines[i][16:76] + " H"
  model_file = "tst_quickref_h.pdb"
  with open(model_file, "w") as f:
    f.write("\n".join(lines[:i+1] + [h] + lines[i+1:]) + "\n")
  try:
    run_quickref(model_file, data_file, "tst_quickref_h",
      ["number_of_macro_cycles=1"])
  except Sorry as e:
    assert "hydrogen" in str(e)
  else:
    raise AssertionError("Sorry expected")
  r = run_quickref(model_file, data_file, "tst_quickref_h",
    ["number_of_macro_cycles=1", "remove_hydrogens=True"])
  assert iotbx.pdb.input(file_name=r.model_file_name).atoms().size() == 67

def exercise_occupancies():
  # data from a model with two conformations (0.7/0.3) of residue 3,
  # refinement starting from 0.5/0.5
  true_inp = iotbx.pdb.input(source_info=None,
    lines=alt_conf_pdb_str(0.7, 0.3))
  xrs = true_inp.xray_structure_simple()
  f_obs = abs(xrs.structure_factors(d_min=1.5, algorithm="direct").f_calc())
  flags = f_obs.generate_r_free_flags(fraction=0.1)
  mtz = f_obs.as_mtz_dataset(column_root_label="F")
  mtz.add_miller_array(flags, column_root_label="R-free-flags")
  data_file = "tst_quickref_occ_data.mtz"
  mtz.mtz_object().write(data_file)
  model_file = "tst_quickref_occ_start.pdb"
  with open(model_file, "w") as f:
    f.write(alt_conf_pdb_str(0.5, 0.5))
  occupancy_only = ["refinement.sites=False", "refinement.adp=False"]
  for extra_args in [occupancy_only,
                     occupancy_only + ["use_curvatures=False"],
                     [],
                     ["occupancies_simultaneous=False"],
                     ["simultaneous=False"],
                     ["refinement.occupancies=False"]]:
    r = run_quickref(model_file, data_file, "tst_quickref_occ",
      ["number_of_macro_cycles=3"] + extra_args)
    assert r.r_work < r.r_work_start
    h = iotbx.pdb.input(file_name=r.model_file_name).construct_hierarchy()
    occ = {}
    for atom in h.atoms():
      label = atom.pdb_label_columns()
      if (label[4] in "AB" and label[5:8] != "   "):
        occ.setdefault(label[4], []).append(atom.occ)
    occ_a = sum(occ["A"]) / len(occ["A"])
    occ_b = sum(occ["B"]) / len(occ["B"])
    if ("refinement.occupancies=False" in extra_args):
      assert occ_a == 0.5 and occ_b == 0.5
    else:
      assert abs(occ_a + occ_b - 1) < 0.01, (occ_a, occ_b)
    if ("refinement.sites=False" in extra_args):
      # with coordinates and ADPs fixed the occupancies are determined;
      # otherwise ADPs partly absorb the occupancy error
      assert abs(occ_a - 0.7) < 0.05, (extra_args, occ_a)

def exercise_reproducibility(model_file, data_file):
  # the random numbers used for the weights are seeded (random_seed)
  results = []
  for seed in [None, None, 1]:
    random.seed(); flex.set_random_seed(int(1.e6*random.random()))
    extra_args = ["number_of_macro_cycles=2"]
    if (seed is not None): extra_args.append("random_seed=%d" % seed)
    results.append(run_quickref(model_file, data_file, "tst_quickref_seed",
      extra_args))
  assert results[0].r_work == results[1].r_work
  assert results[0].r_free == results[1].r_free
  assert results[2].r_free < results[2].r_free_start

def exercise_anisotropic(data_file):
  pdb_inp = iotbx.pdb.input(file_name="tst_quickref_start.pdb")
  xrs = pdb_inp.xray_structure_simple()
  xrs.convert_to_anisotropic()
  hierarchy = pdb_inp.construct_hierarchy()
  hierarchy.adopt_xray_structure(xrs)
  model_file = "tst_quickref_aniso_start.pdb"
  with open(model_file, "w") as f:
    f.write(hierarchy.as_pdb_string(crystal_symmetry=xrs.crystal_symmetry()))
  # start values as read back (ANISOU records are rounded)
  u_start = iotbx.pdb.input(file_name=model_file).xray_structure_simple().\
    scatterers().extract_u_star()
  for adp_type in ["anisotropic", "fixed_anisotropic"]:
    r = run_quickref(model_file, data_file, "tst_quickref_aniso",
      ["number_of_macro_cycles=2", "adp_type=%s" % adp_type])
    assert r.r_free < r.r_free_start
    xrs_final = r.model.get_xray_structure()
    assert xrs_final.use_u_aniso().all_eq(True)
    u_final = xrs_final.scatterers().extract_u_star()
    changed = flex.max(flex.abs(u_final.as_double() - u_start.as_double()))
    # (round-off from u_star <-> u_cart conversions is ~1.e-11)
    if (adp_type == "anisotropic"): assert changed > 1.e-6, changed
    else:                           assert changed < 1.e-8, changed
  # isotropic, and auto at 2 A (> anisotropic_d_min): converted to isotropic
  for args in [["adp_type=isotropic"], []]:
    r = run_quickref(model_file, data_file, "tst_quickref_aniso",
      ["number_of_macro_cycles=1"] + args)
    xrs_final = r.model.get_xray_structure()
    assert xrs_final.use_u_aniso().count(True) == 0, args
    assert xrs_final.use_u_iso().all_eq(True)
  # auto with a high anisotropic_d_min: anisotropic
  r = run_quickref(model_file, data_file, "tst_quickref_aniso",
    ["number_of_macro_cycles=1", "anisotropic_d_min=2.5"])
  assert r.model.get_xray_structure().use_u_aniso().all_eq(True)

def exercise_reference_coordinate_restraints():
  import io, re
  # start from a model with good geometry; the data (from shifted
  # coordinates) pull the atoms away from it
  # (processed, so that symmetric side chains have the canonical naming
  # and atoms can be compared with the refined model)
  from mmtbx.regression.tst_minimization_curvatures import get_model
  model = get_model()
  model_file = "tst_quickref_ref_start.pdb"
  with open(model_file, "w") as f:
    f.write(model.model_as_pdb())
  sites_start = model.get_sites_cart()
  xrs = model.get_xray_structure().deep_copy_scatterers()
  xrs.shake_sites_in_place(rms_difference=0.3)
  f_obs = abs(xrs.structure_factors(d_min=2.0, algorithm="direct").f_calc())
  flags = f_obs.generate_r_free_flags(fraction=0.1)
  mtz = f_obs.as_mtz_dataset(column_root_label="F")
  mtz.add_miller_array(flags, column_root_label="R-free-flags")
  data_file = "tst_quickref_ref_data.mtz"
  mtz.mtz_object().write(data_file)
  rmsd = {}
  for key, args in [("none", []), ("tight", [
      "reference_coordinate_restraints.enabled=True",
      "reference_coordinate_restraints.sigma=0.02",
      "reference_coordinate_restraints.exclude_outliers=False"]),
      ("ca", ["reference_coordinate_restraints.enabled=True",
       "reference_coordinate_restraints.top_out=True",
       "reference_coordinate_restraints.selection=name CA"])]:
    log = io.StringIO()
    r = run_program(program_class=quickref.Program, args=[model_file,
      data_file, "output.prefix=tst_quickref_ref", "output.overwrite=True",
      "number_of_macro_cycles=2", "write_map_coefficients=False",
      "refinement.adp=False"] + args, logger=log)
    rmsd[key] = r.model.get_sites_cart().rms_difference(sites_start)
    if (key != "none"):
      assert "Reference coordinate restraints" in log.getvalue()
    assert r.r_work < r.r_work_start
  # 8 CA atoms, without those of Ramachandran or rotamer outliers
  n = int(re.search("Reference coordinate restraints: ([0-9]+) atoms "
    r"\(sigma=0.20 top-out limit=1.00\)", log.getvalue()).group(1))
  assert 0 < n <= 8, n
  # tight restraints keep the model close to the start
  assert rmsd["tight"] < 0.5 * rmsd["none"], rmsd

def exercise_input_tls():
  """TLS records in the input model, with residual B-factors"""
  import io, re
  from mmtbx.tls import tools
  from mmtbx.regression.tst_minimization_curvatures import get_model
  from cctbx import adptbx
  model = get_model()
  xrs = model.get_xray_structure()
  n = xrs.scatterers().size()
  sel = flex.size_t(range(n))
  tlsos = tools.generate_tlsos(selections=[sel], xray_structure=xrs,
    T=[[0.10, 0.15, 0.12, 0.01, 0, 0.02]], L=[[2.0, 1.5, 3.0, 0.2, 0, 0.1]],
    S=[[0, 0.01, 0, 0.02, 0, -0.01, 0, 0.01, 0]])
  u_tls = tools.u_cart_from_tls(sites_cart=xrs.sites_cart(), selections=[sel],
    tlsos=tlsos)
  b_residual = flex.double([random.uniform(2, 15) for i in range(n)])
  # data from U_tls + residual
  xrs_true = xrs.deep_copy_scatterers()
  xrs_true.convert_to_anisotropic()
  xrs_true.set_u_cart(u_cart=flex.sym_mat3_double([
    (u[0]+r, u[1]+r, u[2]+r, u[3], u[4], u[5]) for u, r in
    zip(u_tls, b_residual * adptbx.b_as_u(1))]))
  f_obs = abs(xrs_true.structure_factors(d_min=2.0,
    algorithm="direct").f_calc())
  flags = f_obs.generate_r_free_flags(fraction=0.1)
  mtz = f_obs.as_mtz_dataset(column_root_label="F")
  mtz.add_miller_array(flags, column_root_label="R-free-flags")
  data_file = "tst_quickref_tls_data.mtz"
  mtz.mtz_object().write(data_file)
  # model: residual B-factors (no ANISOU) and TLS records
  h = model.get_hierarchy().deep_copy()
  h.atoms().set_b(b_residual)
  model.determine_tls_groups(selection_strings=["chain A"],
    generate_tlsos=[sel])
  model.tls_groups.tlsos = tlsos
  model_file = "tst_quickref_tls_start.pdb"
  with open(model_file, "w") as f:
    f.write(model.tls_groups_as_pdb())
    f.write(h.as_pdb_string(crystal_symmetry=model.crystal_symmetry()))
  def run(args):
    log = io.StringIO()
    r = run_program(program_class=quickref.Program, args=[model_file,
      data_file, "output.prefix=tst_quickref_tls", "output.overwrite=True",
      "number_of_macro_cycles=1", "write_map_coefficients=False"] + args,
      logger=log)
    return r, log.getvalue()
  r_ignore, text = run(["input_tls.mode=ignore"])
  assert "Input TLS" not in text
  r_auto, text = run([])
  assert re.search("Input TLS: 1 groups.*TLS applied", text), text
  # starting model with the TLS contribution: lower R
  assert r_auto.r_work_start < 0.7 * r_ignore.r_work_start, (
    r_auto.r_work_start, r_ignore.r_work_start)
  # TLS refinement of the input TLS group, starting from the input TLS
  r_tls, text = run(["tls.enabled=True", "refinement.sites=False"])
  assert "TLS refinement: 1 groups (%d atoms) from the input model" % n \
    in text, text
  # exact split U_tls + residual: the starting model reproduces the data
  # (without TLS refinement, the atoms are made isotropic at this
  # resolution, adp_type=auto)
  assert r_tls.r_work_start < 0.01, r_tls.r_work_start
  assert r_tls.r_work_start < r_auto.r_work_start
  assert r_tls.r_work <= r_tls.r_work_start + 1.e-4
  pdb_inp = iotbx.pdb.input(file_name=r_tls.model_file_name)
  t = pdb_inp.extract_tls_params(pdb_inp.construct_hierarchy()).tls_params
  assert len(t) == 1
  assert approx_equal(t[0].t, tlsos[0].t, eps=0.03)

def exercise_groups():
  """rigid-body, group ADP and TLS refinement"""
  import io
  from mmtbx.regression.tst_ncs_curvatures import ncs_pdb_str
  from mmtbx.regression.tst_minimization_curvatures import get_model
  import mmtbx.model
  from scitbx import matrix
  # true model: 3 chains, B varying per residue
  pdb_inp = iotbx.pdb.input(source_info=None, lines=ncs_pdb_str())
  true_model = mmtbx.model.manager(model_input=pdb_inp, log=null_out())
  true_model.process(make_restraints=True)
  hierarchy = true_model.get_hierarchy()
  for rg in hierarchy.residue_groups():
    b = random.uniform(10, 40)
    for atom in rg.atoms(): atom.b = b
  true_model = mmtbx.model.manager(model_input=None,
    pdb_hierarchy=hierarchy, crystal_symmetry=true_model.crystal_symmetry(),
    log=null_out())
  xrs = true_model.get_xray_structure()
  f_obs = abs(xrs.structure_factors(d_min=2.5, algorithm="fft").f_calc())
  flags = f_obs.generate_r_free_flags(fraction=0.1)
  mtz = f_obs.as_mtz_dataset(column_root_label="F")
  mtz.add_miller_array(flags, column_root_label="R-free-flags")
  data_file = "tst_quickref_groups_data.mtz"
  mtz.mtz_object().write(data_file)
  # start: chain B rotated and translated, flat B
  h = true_model.get_hierarchy().deep_copy()
  sel_b = h.atom_selection_cache().selection("chain B")
  sites = h.atoms().extract_xyz()
  center = matrix.col(sites.select(sel_b).mean())
  r = matrix.col((1,1,0)).axis_and_angle_as_r3_rotation_matrix(3, deg=True)
  for i in sel_b.iselection():
    sites[i] = r * (matrix.col(sites[i]) - center) + center + \
      matrix.col((0.4, -0.3, 0.2))
  h.atoms().set_xyz(sites)
  h.atoms().set_b(flex.double(sites.size(), 20))
  model_file = "tst_quickref_groups_start.pdb"
  with open(model_file, "w") as f:
    f.write(h.as_pdb_string(crystal_symmetry=true_model.crystal_symmetry()))
  sites_true = true_model.get_sites_cart()
  def run(args):
    log = io.StringIO()
    r = run_program(program_class=quickref.Program, args=[model_file,
      data_file, "output.prefix=tst_quickref_groups", "output.overwrite=True",
      "number_of_macro_cycles=3", "write_map_coefficients=False"] + args,
      logger=log)
    return r, log.getvalue()
  # rigid body (with individual ADPs): chain B back in place, rigidly
  r, text = run(["rigid_body.enabled=True"])
  assert "Rigid-body refinement: 3 groups" in text, text
  assert r.r_work < 0.05, r.r_work
  sites = r.model.get_sites_cart()
  # atoms that model processing may rename (symmetric side chains) are
  # not compared
  flip = h.atom_selection_cache().selection(
    "((resname PHE or resname TYR) and (name CD1 or name CD2 or "
    "name CE1 or name CE2)) or (resname ASP and (name OD1 or name OD2)) or "
    "(resname GLU and (name OE1 or name OE2)) or "
    "(resname ARG and (name NH1 or name NH2))")
  isel_b = (sel_b & ~flip).iselection()
  # P1: the overall translation is arbitrary (from chains A and C)
  isel_ac = (~sel_b & ~flip).iselection()
  shift = (sites.select(isel_ac) - sites_true.select(isel_ac)).mean()
  d_start = sites_true.select(isel_b).rms_difference(
    flex.vec3_double(h.atoms().extract_xyz()).select(isel_b))
  d_final = sites_true.select(isel_b).rms_difference(
    sites.select(isel_b) - shift)
  assert d_final < 0.1 * d_start, (d_start, d_final)
  # rigid: interatomic distances within chain B unchanged (up to the
  # rounding of the coordinates in the PDB file)
  s0 = flex.vec3_double(h.atoms().extract_xyz()).select(isel_b)
  s1 = sites.select(isel_b)
  for i, j in [(0, 10), (5, 50), (20, 60)]:
    assert approx_equal(abs(matrix.col(s0[i]) - matrix.col(s0[j])),
      abs(matrix.col(s1[i]) - matrix.col(s1[j])), eps=2.e-3)
  # group ADP (ADPs only): one B per residue
  r, text = run(["group_adp.enabled=True", "refinement.sites=False"])
  assert "Group ADP refinement: 24 groups" in text, text
  assert r.r_work < r.r_work_start
  b = r.model.get_hierarchy().atoms().extract_b()
  for rg in r.model.get_hierarchy().residue_groups():
    b_rg = rg.atoms().extract_b()
    assert flex.max(b_rg) - flex.min(b_rg) < 1.e-3
  # (chain B is misplaced: its B-factors are inflated)
  b_true = true_model.get_hierarchy().atoms().extract_b()
  cc = flex.linear_correlation(b.select(isel_ac),
    b_true.select(isel_ac)).coefficient()
  assert cc > 0.9, cc
  # two groups per residue
  r, text = run(["group_adp.enabled=True", "group_adp.mode=two_per_residue",
    "refinement.sites=False"])
  assert "Group ADP refinement: 48 groups" in text, text
  # TLS, with individual isotropic ADPs; TLS records in the output
  r, text = run(["tls.enabled=True", "refinement.sites=False"])
  assert "TLS refinement: 3 groups" in text, text
  assert r.r_work < r.r_work_start
  pdb_inp = iotbx.pdb.input(file_name=r.model_file_name)
  tls = pdb_inp.extract_tls_params(pdb_inp.construct_hierarchy())
  assert len(tls.tls_params) == 3
  assert pdb_inp.atoms().extract_uij().size() > 0
  # none of these by default
  r, text = run(["number_of_macro_cycles=1"])
  for label in ["Rigid-body", "Group ADP", "TLS"]:
    assert label not in text, label

def exercise_ncs():
  from mmtbx.regression.tst_ncs_curvatures import ncs_pdb_str
  import io
  pdb_inp = iotbx.pdb.input(source_info=None, lines=ncs_pdb_str())
  xrs = pdb_inp.xray_structure_simple()
  f_obs = abs(xrs.structure_factors(d_min=2.5, algorithm="fft").f_calc())
  flags = f_obs.generate_r_free_flags(fraction=0.1)
  mtz = f_obs.as_mtz_dataset(column_root_label="F")
  mtz.add_miller_array(flags, column_root_label="R-free-flags")
  data_file = "tst_quickref_ncs_data.mtz"
  mtz.mtz_object().write(data_file)
  hierarchy = pdb_inp.construct_hierarchy()
  xrs.shake_sites_in_place(rms_difference=0.3)
  xrs.set_b_iso(value=20)
  hierarchy.adopt_xray_structure(xrs)
  model_file = "tst_quickref_ncs_start.pdb"
  with open(model_file, "w") as f:
    f.write(hierarchy.as_pdb_string(crystal_symmetry=xrs.crystal_symmetry()))
  # the default (ncs.type=none) does not use NCS
  log = io.StringIO()
  run_program(program_class=quickref.Program, args=[model_file, data_file,
    "output.prefix=tst_quickref_ncs", "output.overwrite=True",
    "number_of_macro_cycles=1", "write_map_coefficients=False"], logger=log)
  assert "NCS" not in log.getvalue(), log.getvalue()
  for ncs_type, expected in [
      ("cartesian", "Cartesian NCS restraints"),
      ("torsion", "Torsion NCS restraints")]:
    for use_curvatures in [True, False]:
      log = io.StringIO()
      r = run_program(program_class=quickref.Program, args=[model_file,
        data_file, "output.prefix=tst_quickref_ncs", "output.overwrite=True",
        "ncs.type=%s" % ncs_type, "use_curvatures=%s" % use_curvatures,
        "number_of_macro_cycles=3", "write_map_coefficients=False"],
        logger=log)
      text = log.getvalue()
      assert "NCS groups: 1 (3 copies)" in text, text
      assert expected in text, text
      assert r.r_free < r.r_free_start, (ncs_type, r.r_free_start, r.r_free)
      assert r.r_work < 0.7 * r.r_work_start, (ncs_type, r.r_work_start,
        r.r_work)

def run():
  random.seed(0)
  flex.set_random_seed(0)
  model_file, data_file = make_inputs("tst_quickref")
  exercise_anisotropic(data_file)
  exercise_reproducibility(model_file, data_file)
  exercise(model_file, data_file)
  exercise_hydrogens(data_file)
  exercise_occupancies()
  exercise_ncs()
  exercise_reference_coordinate_restraints()
  exercise_groups()
  exercise_input_tls()
  print("OK")

if (__name__ == "__main__"):
  run()
