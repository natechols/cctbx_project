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
    # (with riding hydrogens added)
    atoms = iotbx.pdb.input(file_name=r.model_file_name).atoms()
    assert atoms.extract_element().count(" H") > 40
    assert (atoms.extract_element() != " H").count(True) == 67
    if ("write_map_coefficients=False" in extra_args):
      assert r.mtz_file_name is None
    else:
      assert os.path.isfile(r.mtz_file_name)
      dm = DataManager(["miller_array"])
      dm.process_miller_array_file(r.mtz_file_name)
      labels = dm.get_miller_array_labels(r.mtz_file_name)
      assert "2FOFCWT,PH2FOFCWT" in labels, labels
      assert "FOFCWT,PHFOFCWT" in labels, labels

def exercise_hydrogens(model_file, data_file):
  """riding hydrogens: added, kept (riding) or removed"""
  import io, re
  log = io.StringIO()
  r = run_program(program_class=quickref.Program, args=[model_file,
    data_file, "output.prefix=tst_quickref_h", "output.overwrite=True",
    "number_of_macro_cycles=2", "add_hydrogens=True"], logger=log)
  text = log.getvalue()
  n_added = int(re.search("Hydrogens added: ([0-9]+)", text).group(1))
  assert n_added > 40, text
  assert "Riding hydrogens: %d" % n_added in text, text
  assert r.r_work < r.r_work_start
  # output: hydrogens with the B-factors of their parent atoms
  model_h = r.model
  xrs = model_h.get_xray_structure()
  b = xrs.extract_u_iso_or_u_equiv()
  for rc in model_h.riding_h_manager.parameterization_cpp:
    assert approx_equal(b[rc.ih], b[rc.a0])
  pdb_inp = iotbx.pdb.input(file_name=r.model_file_name)
  h_file = r.model_file_name
  assert pdb_inp.atoms().size() == 67 + n_added
  # input with hydrogens: riding
  log = io.StringIO()
  r = run_program(program_class=quickref.Program, args=[h_file, data_file,
    "output.prefix=tst_quickref_h2", "output.overwrite=True",
    "number_of_macro_cycles=1"], logger=log)
  assert "Riding hydrogens: %d" % n_added in log.getvalue(), log.getvalue()
  # removed
  r = run_quickref(h_file, data_file, "tst_quickref_h3",
    ["number_of_macro_cycles=1", "remove_hydrogens=True",
     "add_hydrogens=False"])
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

def exercise_convergence(model_file, data_file):
  """macro-cycles stop when r_work changes by less than convergence_r_work"""
  import io, re
  def run(args):
    log = io.StringIO()
    r = run_program(program_class=quickref.Program, args=[model_file,
      data_file, "output.prefix=tst_quickref_conv", "output.overwrite=True",
      "number_of_macro_cycles=6", "write_map_coefficients=False",
      "tls.enabled=False"] + args, logger=log)
    text = log.getvalue()
    r_work = [float(v) for v in re.findall(r"macro-cycle \d+: r_work=(\S+)",
      text)]
    return r, text, r_work
  r, text, r_work = run(["convergence_r_work=0.002"])
  assert "Converged" in text, text
  assert 2 <= len(r_work) < 6, r_work
  assert abs(r_work[-1] - r_work[-2]) < 0.002, r_work
  # final R-factors after the update of the mask and scales
  r_work_final = float(re.search(r"Final: r_work=(\S+)", text).group(1))
  assert abs(r.r_work - r_work_final) < 1.e-4, (r.r_work, r_work_final)
  r, text, r_work = run(["convergence_r_work=0"])
  assert "Converged" not in text
  assert len(r_work) == 6, r_work

def exercise_find_tls_groups(model_file, data_file):
  """tls.find_groups: TLS groups from mmtbx.command_line.find_tls_groups"""
  import io, re
  def run(model_file):
    log = io.StringIO()
    r = run_program(program_class=quickref.Program, args=[model_file,
      data_file, "output.prefix=tst_quickref_find_tls",
      "output.overwrite=True", "number_of_macro_cycles=1",
      "write_map_coefficients=False", "tls.enabled=True",
      "tls.find_groups=True"], logger=log)
    return r, log.getvalue()
  # identical B-factors: no groups found, the default group is used
  r, text = run(model_file)
  assert "TLS groups not found" in text, text
  assert "TLS refinement: 1 groups" in text, text
  # B-factors increasing along the chain
  pdb_inp = iotbx.pdb.input(file_name=model_file)
  h = pdb_inp.construct_hierarchy()
  atoms = h.atoms()
  atoms.set_b(flex.double([10 + 0.5*i + random.uniform(0, 2)
    for i in range(atoms.size())]))
  model_file_b = "tst_quickref_find_tls.pdb"
  with open(model_file_b, "w") as f:
    f.write(h.as_pdb_string(crystal_symmetry=pdb_inp.crystal_symmetry()))
  r, text = run(model_file_b)
  n = int(re.search(r"TLS groups found \(find_tls_groups\): (\d+)",
    text).group(1))
  assert n >= 1, text
  assert re.search(r"TLS refinement: %d groups" % n, text), text
  assert r.r_free < r.r_free_start

def exercise_anomalous():
  """anomalous data (F+, F-): used as separate observations, anomalous
  difference map; optional f' and f'' from the wavelength"""
  import io, re
  import iotbx.mtz
  from mmtbx.regression.tst_minimization_curvatures import get_model
  model = get_model()
  xrs = model.get_xray_structure().deep_copy_scatterers()
  wavelength = 1.75
  xrs.set_inelastic_form_factors(photon=wavelength, table="sasaki")
  f_obs = abs(xrs.structure_factors(d_min=1.6, algorithm="direct",
    anomalous_flag=True).f_calc())
  flags = f_obs.average_bijvoet_mates().generate_r_free_flags(fraction=0.1)
  mtz = f_obs.as_mtz_dataset(column_root_label="F", wavelength=wavelength)
  mtz.add_miller_array(flags, column_root_label="R-free-flags")
  data_file = "tst_quickref_anom_data.mtz"
  mtz.mtz_object().write(data_file)
  model_file = "tst_quickref_anom_start.pdb"
  with open(model_file, "w") as f:
    f.write(model.model_as_pdb())
  s_sites = [sc.site for sc in xrs.scatterers() if sc.scattering_type == "S"]
  assert len(s_sites) > 0
  def run(args):
    log = io.StringIO()
    r = run_program(program_class=quickref.Program, args=[model_file,
      data_file, "output.prefix=tst_quickref_anom", "output.overwrite=True",
      "number_of_macro_cycles=3", "convergence_r_work=0"] + args, logger=log)
    labels = iotbx.mtz.object(r.mtz_file_name).column_labels()
    return r, log.getvalue(), labels
  # default: Friedel mates as separate observations, anomalous map
  r0, text, labels = run([])
  assert "Anomalous data:" in text, text
  assert "f' and f''" not in text
  assert r0.fmodel.f_obs().anomalous_flag()
  for label in ["2FOFCWT", "FOFCWT", "ANOM", "PHANOM"]:
    assert label in labels, labels
  assert not [l for l in labels if "(+)" in l or "(-)" in l], labels
  anom = [a for a in iotbx.mtz.object(r0.mtz_file_name).as_miller_arrays()
    if a.info().labels[0] == "ANOM"][0]
  fft_map = anom.fft_map(resolution_factor=0.25)
  fft_map.apply_sigma_scaling()
  for site in s_sites:
    assert fft_map.real_map_unpadded().eight_point_interpolation(site) > 10
  # f' and f'' at the wavelength of the reflection file
  r1, text, labels = run(["anomalous.set_inelastic_form_factors=True"])
  assert re.search(r"S +f'= *0\.35\d f''= *0\.70\d", text), text
  assert r1.r_free < r0.r_free, (r1.r_free, r0.r_free)
  # merged
  r2, text, labels = run(["anomalous.use_anomalous_data=False"])
  assert "Anomalous data:" not in text
  assert not r2.fmodel.f_obs().anomalous_flag()
  assert "ANOM" not in labels

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
    # (riding hydrogens are isotropic, with U_eq of the parent atoms)
    hd = r.model.get_hd_selection()
    assert hd.count(True) > 40
    assert xrs_final.use_u_aniso().select(~hd).all_eq(True)
    assert xrs_final.use_u_aniso().select(hd).count(True) == 0
    u_eq = xrs_final.extract_u_iso_or_u_equiv()
    for rc in r.model.riding_h_manager.parameterization_cpp:
      assert approx_equal(u_eq[rc.ih], u_eq[rc.a0])
    # (riding hydrogens are added: non-H atoms only)
    u_final = xrs_final.scatterers().extract_u_star().select(
      ~r.model.get_hd_selection())
    changed = flex.max(flex.abs(u_final.as_double() - u_start.as_double()))
    # (round-off from u_star <-> u_cart conversions is ~1.e-11)
    if (adp_type == "anisotropic"): assert changed > 1.e-6, changed
    else:                           assert changed < 1.e-8, changed
  # isotropic, and auto at 2 A (> anisotropic_d_min): converted to isotropic
  for args in [["adp_type=isotropic"], []]:
    r = run_quickref(model_file, data_file, "tst_quickref_aniso",
      ["number_of_macro_cycles=1", "tls.enabled=False"] + args)
    xrs_final = r.model.get_xray_structure()
    assert xrs_final.use_u_aniso().count(True) == 0, args
    assert xrs_final.use_u_iso().all_eq(True)
  # with isotropic ADPs, TLS is refined by default (tls.enabled=Auto): the
  # ADPs are the TLS contribution plus individual isotropic ADPs
  import io
  log = io.StringIO()
  r = run_program(program_class=quickref.Program, args=[model_file,
    data_file, "output.prefix=tst_quickref_aniso", "output.overwrite=True",
    "number_of_macro_cycles=1"], logger=log)
  assert "TLS refinement enabled (isotropic ADPs at d_min=2.00)" in \
    log.getvalue(), log.getvalue()
  # (riding hydrogens are isotropic, not in TLS groups)
  xrs_final = r.model.get_xray_structure()
  hd = r.model.get_hd_selection()
  assert xrs_final.use_u_aniso().select(~hd).all_eq(True)
  assert xrs_final.use_u_aniso().select(hd).count(True) == 0
  assert xrs_final.use_u_iso().all_eq(True)
  # auto with a high anisotropic_d_min: anisotropic
  r = run_quickref(model_file, data_file, "tst_quickref_aniso",
    ["number_of_macro_cycles=1", "anisotropic_d_min=2.5"])
  hd = r.model.get_hd_selection()
  assert r.model.get_xray_structure().use_u_aniso().select(~hd).all_eq(True)

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
    # (non-H atoms: riding hydrogens are added)
    rmsd[key] = r.model.get_sites_cart().select(
      ~r.model.get_hd_selection()).rms_difference(sites_start)
    if (key != "none"):
      assert "Reference coordinate restraints" in log.getvalue()
    assert r.r_work < r.r_work_start
  # 8 CA atoms, without those of Ramachandran or rotamer outliers
  n = int(re.search("Reference coordinate restraints: ([0-9]+) atoms "
    r"\(sigma=0.20 top-out limit=1.00\)", log.getvalue()).group(1))
  assert 0 < n <= 8, n
  # tight restraints keep the model close to the start
  assert rmsd["tight"] < 0.5 * rmsd["none"], rmsd

def exercise_unrestrained():
  import io
  # a ligand without restraint definitions: its coordinates are fixed
  # (with a warning), the rest of the model is refined
  ligand = """\
HETATM   68  C1  ZQX B   1      27.700  39.700  25.500  1.00 20.00           C
HETATM   69  C2  ZQX B   1      28.900  39.700  26.300  1.00 20.00           C
HETATM   70  O1  ZQX B   1      26.500  39.700  26.300  1.00 20.00           O
"""
  lines = [l for l in pdb_str.splitlines() if not l.startswith("END")]
  pdb_inp = iotbx.pdb.input(source_info=None,
    lines="\n".join(lines) + "\n" + ligand)
  xrs = pdb_inp.xray_structure_simple()
  f_obs = abs(xrs.structure_factors(d_min=2.0, algorithm="direct").f_calc())
  flags = f_obs.generate_r_free_flags(fraction=0.1)
  mtz = f_obs.as_mtz_dataset(column_root_label="F")
  mtz.add_miller_array(flags, column_root_label="R-free-flags")
  data_file = "tst_quickref_unk_data.mtz"
  mtz.mtz_object().write(data_file)
  hierarchy = pdb_inp.construct_hierarchy()
  xrs.shake_sites_in_place(rms_difference=0.3)
  xrs.set_b_iso(value=20)
  hierarchy.adopt_xray_structure(xrs)
  model_file = "tst_quickref_unk_start.pdb"
  with open(model_file, "w") as f:
    f.write(hierarchy.as_pdb_string(crystal_symmetry=xrs.crystal_symmetry()))
  ligand_start = xrs.sites_cart()[-3:]
  for extra_args in [[], ["add_hydrogens=False"]]:
    log = io.StringIO()
    r = run_program(program_class=quickref.Program, args=[model_file,
      data_file, "output.prefix=tst_quickref_unk", "output.overwrite=True",
      "write_map_coefficients=False"] + extra_args, logger=log)
    assert "WARNING: 3 atoms have no geometry restraints" in log.getvalue()
    assert "ZQX B 1 (3 of 3 atoms)" in log.getvalue()
    assert r.r_work < 0.7 * r.r_work_start, (extra_args, r.r_work_start,
      r.r_work)
    sel = r.model.selection("resname ZQX")
    assert sel.count(True) == 3
    assert approx_equal(r.model.get_sites_cart().select(sel), ligand_start,
      eps=1.e-3)
    # (the other atoms move)
    assert r.model.get_sites_cart().select(~sel & ~r.model.get_hd_selection(
      )).rms_difference(xrs.sites_cart()[:-3]) > 0.05

def exercise_user_bulk_solvent():
  # data with a "solvent" contribution (atoms not in the model): with the
  # solvent structure factors supplied, R is much lower than with the flat
  # mask, and the user F_mask is kept (also after outlier rejection)
  pdb_inp = iotbx.pdb.input(source_info=None, lines=pdb_str)
  xrs = pdb_inp.xray_structure_simple()
  solvent = iotbx.pdb.input(source_info=None, lines="""\
CRYST1   50.840   42.770   28.950  90.00  90.00  90.00 P 21 21 21
HETATM    1  O   HOH S   1      10.000  10.000  10.000  1.00 30.00           O
HETATM    2  O   HOH S   2      15.000   5.000  20.000  1.00 30.00           O
HETATM    3  O   HOH S   3       5.000  30.000   3.000  1.00 30.00           O
HETATM    4  O   HOH S   4      40.000  15.000  12.000  1.00 30.00           O
HETATM    5  O   HOH S   5      45.000  35.000  25.000  1.00 30.00           O
HETATM    6  O   HOH S   6      20.000  20.000   5.000  1.00 30.00           O
""").xray_structure_simple()
  f_calc = xrs.structure_factors(d_min=2.0, algorithm="direct").f_calc()
  f_solvent = f_calc.structure_factors_from_scatterers(solvent,
    algorithm="direct").f_calc()
  f_obs = abs(f_calc.customized_copy(data=f_calc.data()+f_solvent.data()))
  flags = f_obs.generate_r_free_flags(fraction=0.1)
  mtz = f_obs.as_mtz_dataset(column_root_label="F")
  mtz.add_miller_array(flags, column_root_label="R-free-flags")
  data_file = "tst_quickref_solv_data.mtz"
  mtz.mtz_object().write(data_file)
  mtz = f_solvent.as_mtz_dataset(column_root_label="FPART")
  solvent_file = "tst_quickref_solv_fpart.mtz"
  mtz.mtz_object().write(solvent_file)
  model_file = "tst_quickref_solv_start.pdb"
  with open(model_file, "w") as f:
    f.write(pdb_inp.construct_hierarchy().as_pdb_string(
      crystal_symmetry=xrs.crystal_symmetry()))
  args = [model_file, data_file, "output.prefix=tst_quickref_solv",
    "output.overwrite=True", "write_map_coefficients=False",
    "number_of_macro_cycles=1", "add_hydrogens=False"]
  r_flat = run_program(program_class=quickref.Program, args=args,
    logger=null_out())
  r_user = run_program(program_class=quickref.Program, args=args + [
    "bulk_solvent_map.file_name=%s" % solvent_file,
    "bulk_solvent_map.phases_label=PHIFPART"], logger=null_out())
  assert r_user.r_work_start < 0.5 * r_flat.r_work_start, (
    r_user.r_work_start, r_flat.r_work_start)
  assert r_user.r_work < 0.5 * r_flat.r_work, (r_user.r_work, r_flat.r_work)
  fm = r_user.fmodel
  assert fm._user_f_masks is not None
  assert fm._user_f_masks[0].indices().all_eq(fm.f_obs().indices())
  assert fm.f_masks()[0].data().all_eq(fm._user_f_masks[0].data())
  # (propagated through select)
  sel = flex.bool(fm.f_obs().size(), True)
  sel[0] = False
  assert fm.select(sel)._user_f_masks[0].size() == sel.count(True)
  # missing column
  try:
    run_program(program_class=quickref.Program, args=args + [
      "bulk_solvent_map.file_name=%s" % solvent_file], logger=null_out())
  except Sorry as e:
    assert "Column PHIPART not found" in str(e), str(e)
  else: raise AssertionError("Sorry expected")

def exercise_twinning_check():
  import io
  # the test model in P4 (twin law k,h,-l); data twinned with a twin
  # fraction of 0.4: the twinning check warns (with the R-factors), the
  # refinement is not changed; no warning for untwinned data
  from cctbx import crystal
  pdb_inp = iotbx.pdb.input(source_info=None, lines=pdb_str)
  hierarchy = pdb_inp.construct_hierarchy()
  cs = crystal.symmetry(unit_cell=(80, 80, 40, 90, 90, 90),
    space_group_symbol="P 4")
  xrs = hierarchy.extract_xray_structure(crystal_symmetry=cs)
  f_calc = xrs.structure_factors(d_min=2.0, algorithm="direct").f_calc()
  i_calc = abs(f_calc).set_observation_type_xray_amplitude().f_as_f_sq()
  model_file = "tst_quickref_twin_start.pdb"
  with open(model_file, "w") as f:
    f.write(hierarchy.as_pdb_string(crystal_symmetry=cs))
  flags = i_calc.generate_r_free_flags(fraction=0.1)
  results = {}
  for alpha in [0.4, 0]:
    i_obs = i_calc.twin_data(twin_law="k,h,-l", alpha=alpha) if alpha > 0 \
      else i_calc
    f_obs = i_obs.f_sq_as_f()
    f_obs = f_obs.customized_copy(sigmas=f_obs.data()*0.01+0.1)
    mtz = f_obs.as_mtz_dataset(column_root_label="F")
    mtz.add_miller_array(flags, column_root_label="R-free-flags")
    data_file = "tst_quickref_twin_data_%g.mtz" % alpha
    mtz.mtz_object().write(data_file)
    log = io.StringIO()
    r = run_program(program_class=quickref.Program, args=[model_file,
      data_file, "output.prefix=tst_quickref_twin", "output.overwrite=True",
      "write_map_coefficients=False", "number_of_macro_cycles=1",
      "add_hydrogens=False"], logger=log)
    results[alpha] = (r, log.getvalue())
  r, text = results[0.4]
  assert r.twinning is not None and r.twinning.confirmed
  assert "WARNING: the data are probably twinned" in text, text
  # (one twin law: k,h,-l, or the equivalent h,-k,-l)
  assert len(r.twinning.twin_laws) == 1
  law, h_alpha, britton_alpha, r_work, r_free, fraction = \
    r.twinning.twin_laws[0]
  assert law in ["k,h,-l", "h,-k,-l"], law
  assert approx_equal(fraction, 0.4, eps=0.05), fraction
  assert approx_equal(h_alpha, 0.4, eps=0.05), h_alpha
  assert r_free < 0.5 * r.twinning.r_no_twin[1], (r_free, r.twinning.r_no_twin)
  r, text = results[0]
  assert r.twinning is None or not r.twinning.confirmed
  assert "WARNING: the data are probably twinned" not in text
  # twinning.check=False
  r = run_program(program_class=quickref.Program, args=[model_file,
    "tst_quickref_twin_data_0.4.mtz", "output.prefix=tst_quickref_twin",
    "output.overwrite=True", "write_map_coefficients=False",
    "number_of_macro_cycles=1", "add_hydrogens=False",
    "twinning.check=False"], logger=null_out())
  assert r.twinning is None

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
      "number_of_macro_cycles=1", "write_map_coefficients=False",
      # (the data are calculated without hydrogens)
      "add_hydrogens=False"] + args,
      logger=log)
    return r, log.getvalue()
  # (without TLS refinement)
  r_ignore, text = run(["input_tls.mode=ignore", "tls.enabled=False"])
  assert "Input TLS" not in text
  r_iso, text = run(["tls.enabled=False"])
  assert re.search("Input TLS: 1 groups.*TLS applied", text), text
  # starting model with the TLS contribution (made isotropic at this
  # resolution, adp_type=auto): lower R
  assert r_iso.r_work_start < 0.7 * r_ignore.r_work_start, (
    r_iso.r_work_start, r_ignore.r_work_start)
  # TLS refinement (by default with isotropic ADPs) of the input TLS group,
  # starting from the input TLS
  r_tls, text = run(["refinement.sites=False"])
  assert "TLS refinement enabled" in text, text
  assert "TLS refinement: 1 groups (%d atoms) from the input model" % n \
    in text, text
  # exact split U_tls + residual: the starting model reproduces the data
  assert r_tls.r_work_start < 0.01, r_tls.r_work_start
  assert r_tls.r_work_start < r_iso.r_work_start
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
      "number_of_macro_cycles=3", "write_map_coefficients=False",
      # (the data are calculated without hydrogens)
      "add_hydrogens=False"] + args,
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
  r, text = run(["group_adp.enabled=True", "refinement.sites=False",
    "tls.enabled=False"])
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
    "refinement.sites=False", "tls.enabled=False"])
  assert "Group ADP refinement: 48 groups" in text, text
  # TLS, with individual isotropic ADPs; TLS records in the output
  r, text = run(["tls.enabled=True", "refinement.sites=False"])
  assert "TLS refinement: 3 groups" in text, text
  assert r.r_work < r.r_work_start
  pdb_inp = iotbx.pdb.input(file_name=r.model_file_name)
  tls = pdb_inp.extract_tls_params(pdb_inp.construct_hierarchy())
  assert len(tls.tls_params) == 3
  assert pdb_inp.atoms().extract_uij().size() > 0
  # rigid body and group ADP are not used by default; TLS is used with
  # isotropic ADPs (here at 2.5 A, tls.enabled=Auto)
  r, text = run(["number_of_macro_cycles=1"])
  for label in ["Rigid-body", "Group ADP"]:
    assert label not in text, label
  assert "TLS refinement enabled" in text
  assert "TLS refinement: 3 groups" in text
  r, text = run(["number_of_macro_cycles=1", "tls.enabled=False"])
  assert "TLS" not in text
  # chains smaller than tls.min_group_size are not TLS groups
  r, text = run(["number_of_macro_cycles=1", "tls.min_group_size=100"])
  assert "No TLS groups with at least 100 atoms" in text, text

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
  # the default (ncs.type=auto) uses torsion NCS if NCS is found;
  # ncs.type=none does not use NCS
  for args, expected in [([], True), (["ncs.type=none"], False)]:
    log = io.StringIO()
    run_program(program_class=quickref.Program, args=[model_file, data_file,
      "output.prefix=tst_quickref_ncs", "output.overwrite=True",
      "number_of_macro_cycles=1", "write_map_coefficients=False"] + args,
      logger=log)
    text = log.getvalue()
    assert ("Torsion NCS restraints" in text) == expected, (args, text)
    assert ("NCS groups: 1 (3 copies)" in text) == expected, (args, text)
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

def exercise_weight_optimization(model_file, data_file):
  """weights.optimize: trial refinements against a subset of the work set"""
  import io, re
  def run(args):
    log = io.StringIO()
    r = run_program(program_class=quickref.Program, args=[model_file,
      data_file, "output.prefix=tst_quickref_w", "output.overwrite=True",
      "number_of_macro_cycles=2", "write_map_coefficients=False",
      "tls.enabled=False"] + args, logger=log)
    return r, log.getvalue()
  r0, text = run([])
  assert "Weight optimization" not in text
  # wxu_scale trials 0.5, 1, 2 (wxc_scale=0.5), then wxc_scale trials 0.25
  # and 1 (0.5 is done); grids are extended if the best value is at an end
  r, text = run(["weights.optimize=True", "weights.wxu_scale_grid=0.5 2",
    "weights.wxc_scale_grid=0.25 1", "weights.min_test_reflections=100",
    "weights.max_bond_rmsd=1"])
  trials = re.findall(r"  wxc_scale=(\S+) +wxu_scale=(\S+) +r_work=(\S+) "
    r"r_test=(\S+)", text)
  assert 5 <= len(trials) <= 9, text
  assert [t[1] for t in trials[:3]] == ["0.5", "1", "2"], trials
  n_wxu = [i for i, t in enumerate(trials) if t[0] != "0.5"][0]
  # the R-free set is not used: trials against the work set only
  m = re.search(r"trial refinements against (\d+) of (\d+) work reflections "
    r"\(test subset: (\d+)\)", text)
  n_train, n_work, n_test = [int(v) for v in m.groups()]
  assert n_train + n_test == n_work
  assert n_test == max(100, int(0.05 * n_work)), (n_test, n_work)
  assert r.fmodel.r_free_flags().data().count(False) == n_work
  # the selected scales are those with the lowest r_test
  best_wxu = min(trials[:n_wxu], key = lambda t: float(t[3]))[1]
  best_wxc = min([t for t in trials if t[1] == best_wxu],
    key = lambda t: float(t[3]))[0]
  assert "Weight optimization: wxc_scale=%s wxu_scale=%s" % (
    best_wxc, best_wxu) in text, text
  assert r.r_free_start == r0.r_free_start
  assert r.r_free < r.r_free_start
  # all trials over max_bond_rmsd: the trials with the smallest rmsd(bonds)
  r, text = run(["weights.optimize=True", "weights.wxu_scale_grid=0.5 2",
    "weights.wxc_scale_grid=0.25 1", "weights.min_test_reflections=100",
    "weights.max_bond_rmsd=0.00001"])
  trials = re.findall(r"  wxc_scale=(\S+) +wxu_scale=(\S+) +r_work=\S+ "
    r"r_test=\S+ rmsd\(bonds\)=(\S+) .*\(rmsd\(bonds\) > 1e-05\)", text)
  assert len(trials) == len(re.findall("  wxc_scale=", text)), text
  n_wxu = [i for i, t in enumerate(trials) if t[0] != "0.5"][0]
  best_wxu = min(trials[:n_wxu], key = lambda t: float(t[2]))[1]
  best_wxc = min([t for t in trials if t[1] == best_wxu],
    key = lambda t: float(t[2]))[0]
  assert "Weight optimization: wxc_scale=%s wxu_scale=%s" % (
    best_wxc, best_wxu) in text, text
  # the trials do not change the starting model: with only the default
  # scales, the result is that of the refinement without trials
  r1, text = run(["weights.optimize=True", "weights.wxu_scale_grid=1",
    "weights.wxc_scale_grid=0.5", "weights.min_test_reflections=100"])
  assert len(re.findall("  wxc_scale=", text)) == 1, text
  assert abs(r1.r_free - r0.r_free) < 1.e-3, (r1.r_free, r0.r_free)
  # with TLS refinement (the TLS parameters are reset after each trial)
  r, text = run(["weights.optimize=True", "weights.wxu_scale_grid=1 2",
    "weights.wxc_scale_grid=0.5", "weights.min_test_reflections=100",
    "tls.enabled=True", "tls.min_group_size=10"])
  assert "TLS refinement: 1 groups" in text, text
  assert "Weight optimization: wxc_scale=" in text, text
  assert r.r_free < r.r_free_start

def run():
  random.seed(0)
  flex.set_random_seed(0)
  model_file, data_file = make_inputs("tst_quickref")
  exercise_anisotropic(data_file)
  exercise_reproducibility(model_file, data_file)
  exercise_convergence(model_file, data_file)
  exercise_find_tls_groups(model_file, data_file)
  exercise_anomalous()
  exercise(model_file, data_file)
  exercise_hydrogens(model_file, data_file)
  exercise_weight_optimization(model_file, data_file)
  exercise_occupancies()
  exercise_ncs()
  exercise_reference_coordinate_restraints()
  exercise_groups()
  exercise_input_tls()
  exercise_unrestrained()
  exercise_user_bulk_solvent()
  exercise_twinning_check()
  print("OK")

if (__name__ == "__main__"):
  run()
