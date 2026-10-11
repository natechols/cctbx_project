"""Quick reciprocal-space refinement of coordinates, ADPs and occupancies"""
from __future__ import absolute_import, division, print_function
from libtbx.program_template import ProgramTemplate
from libtbx.utils import Sorry, null_out
from libtbx import group_args, Auto
from cctbx import xray, adptbx
from cctbx.array_family import flex
import iotbx.pdb
import mmtbx
import mmtbx.refinement.adp_refinement
import mmtbx.refinement.refinement_flags
import mmtbx.refinement.occupancies
from mmtbx.refinement import minimization
from mmtbx.refinement import weights as weights_module
import scitbx.lbfgs
import os
import random
import time

master_phil_str = '''
scattering_table = *n_gaussian wk1995 it1992 electron
  .type = choice
  .short_caption = Scattering table
  .help = Scattering table for structure factors calculations
refinement
  .short_caption = Refinement
{
  number_of_macro_cycles = 5
    .type = int(value_min=1)
    .help = Maximum number of macro-cycles. Each macro-cycle updates the \
            bulk-solvent and scale parameters, then minimizes the target.
  convergence_r_work = 0.001
    .type = float(value_min=0)
    .help = Stop when r_work changes by less than this in a macro-cycle \
            (after at least min_macro_cycles). With 0, all \
            number_of_macro_cycles are done. r_free is not used.
  min_macro_cycles = 2
    .type = int(value_min=1)
  max_iterations = 50
    .type = int(value_min=1)
    .help = Maximum number of LBFGS iterations per macro-cycle
  sites = True
    .type = bool
    .help = Refine coordinates
  adp = True
    .type = bool
    .help = Refine ADPs
  adp_type = *auto isotropic anisotropic fixed_anisotropic
    .type = choice
    .help = ADPs of atoms that are anisotropic in the input model: refined \
            anisotropically (anisotropic), converted to isotropic and refined \
            isotropically (isotropic), or kept fixed (fixed_anisotropic). \
            auto: anisotropic if the resolution is anisotropic_d_min or \
            better, otherwise isotropic (with tls.enabled=True, the TLS \
            contribution models the anisotropy). Atoms with isotropic ADPs \
            remain isotropic.
  anisotropic_d_min = 1.7
    .type = float(value_min=0)
    .help = Resolution limit for adp_type=auto
  simultaneous = True
    .type = bool
    .help = Refine coordinates and ADPs simultaneously (otherwise they are \
            refined in alternation in each macro-cycle)
  occupancies = True
    .type = bool
    .help = Refine occupancies of alternative conformations (constrained \
            to sum to 1) and of other atoms with partial occupancies
  occupancies_simultaneous = True
    .type = bool
    .help = Refine occupancies simultaneously with the ADPs (and with the \
            coordinates if simultaneous=True). If False, occupancies are \
            refined in a separate step after the coordinates and ADPs in \
            each macro-cycle.
  occupancy_min = 0
    .type = float
  occupancy_max = 1
    .type = float
  occupancy_max_iterations = 25
    .type = int(value_min=1)
    .help = Maximum number of LBFGS iterations for occupancies per macro-cycle
  use_curvatures = True
    .type = bool
    .help = Precondition the minimizer with approximate curvatures (diagonal \
            of the second derivatives) of the target
  smooth_target = True
    .type = bool
    .expert_level = 2
    .help = Keep the target smooth during each minimization (fixed ADP \
            restraint pairs and restraint normalization), see \
            mmtbx.refinement.minimization.lbfgs
  curvature_floor_fraction = 0.1
    .type = float(value_min=0)
    .help = Parameters without curvature information get this fraction of \
            the mean curvature
  target = *ml ls
    .type = choice
    .help = Maximum-likelihood (ml) or least-squares (ls) X-ray target
}
weights
  .short_caption = Weights
{
  wxc_scale = 0.5
    .type = float(value_min=0)
    .help = Scale for the X-ray weight in coordinate refinement \
            (wxc = wxc_scale * |g_geometry| / |g_xray|)
  wxu_scale = 1.0
    .type = float(value_min=0)
    .help = Scale for the X-ray weight in ADP refinement \
            (wxu = wxu_scale * |g_adp_restraints| / |g_xray|)
  recompute_each_macro_cycle = False
    .type = bool
    .help = Recompute the weights at the start of each macro-cycle \
            (otherwise once, from the input model)
  optimize = False
    .type = bool
    .help = Choose wxu_scale and wxc_scale by trial refinements. The trials \
            do not use the R-free set: a test subset of the work set \
            (test_fraction or min_test_reflections, whichever is larger) is \
            left out of the trial refinements, and the scales giving the \
            lowest R-factor for this subset are used in the refinement \
            against the full work set.
  test_fraction = 0.05
    .type = float(value_min=0, value_max=0.5)
  min_test_reflections = 2000
    .type = int(value_min=0)
  wxu_scale_grid = 0.25 0.5 1 2 4 8
    .type = floats(value_min=0)
    .help = wxu_scale values of the trials (with the default wxc_scale), \
            extended by factors of 2 (at most twice) if the best value is \
            at the end of the grid
  wxc_scale_grid = 0.125 0.25 0.5 1 2
    .type = floats(value_min=0)
    .help = wxc_scale values of the trials (with the best wxu_scale), \
            extended as wxu_scale_grid
  trial_macro_cycles = 1
    .type = int(value_min=1)
  max_bond_rmsd = Auto
    .type = float(value_min=0)
    .help = Trials with a larger rmsd(bonds) are not used (if all are \
            larger, the trial with the smallest rmsd(bonds) is used). \
            Auto: 0.015 A, or 0.02 A at atomic resolution (d_min <= 1.0 A).
  trial_shake_sites = 0
    .type = float(value_min=0)
    .help = RMS shift (A) of the coordinates of the starting model of the \
            trials (with distorted geometry, the trials favour tight \
            geometry restraints)
  trial_shake_adp = 10
    .type = float(value_min=0)
    .help = Spread of the B-factors of the starting model of the trials \
            (anisotropic ADPs are not changed). The input model was refined \
            against the test subset, and without the perturbation the \
            trials favour the weights that change the model least.
}
rigid_body
  .short_caption = Rigid-body refinement
{
  enabled = False
    .type = bool
    .help = Refine the coordinates as rigid bodies instead of individually \
            (if refinement.sites=True)
  group = None
    .type = atom_selection
    .multiple = True
    .help = Rigid groups (default: one group per chain, without water)
  max_iterations = 25
    .type = int(value_min=1)
  number_of_zones = 5
    .type = int(value_min=1)
    .help = Number of resolution zones (from low to high resolution)
}
group_adp
  .short_caption = Group ADP refinement
{
  enabled = False
    .type = bool
    .help = Refine one isotropic ADP shift per group instead of individual \
            ADPs (if refinement.adp=True)
  mode = *one_per_residue two_per_residue
    .type = choice
    .help = Groups: one per residue, or two per residue (main chain and \
            side chain of amino acids)
  max_iterations = 50
    .type = int(value_min=1)
}
tls
  .short_caption = TLS refinement
{
  enabled = Auto
    .type = bool
    .help = Refine TLS parameters in a separate step in each macro-cycle. \
            The ADPs of the atoms in TLS groups are the TLS contribution \
            plus an individual isotropic (or group) contribution. Auto: \
            enabled if the ADPs are refined isotropically (see \
            refinement.adp_type, i.e. at d_min > anisotropic_d_min with \
            adp_type=auto).
  group = None
    .type = atom_selection
    .multiple = True
    .help = TLS groups (default: the TLS groups of the input model, or one \
            group per chain, without water)
  min_group_size = 50
    .type = int(value_min=1)
    .help = Default groups (one per chain) with fewer atoms are not used
  max_iterations = 25
    .type = int(value_min=1)
  joint = False
    .type = bool
    .help = Experimental: refine the TLS parameters simultaneously with \
            the coordinates and individual ADPs (in the minimizer of the \
            individual parameters, mmtbx.refinement.minimization \
            tls_groups, with the target of the minimizer) instead of in a \
            separate step (least-squares target). In a benchmark of 16 \
            structures, R-free was 0.0023 higher on average (worse at low \
            resolution) for 3% less time.
  joint_ls_first_cycle = False
    .type = bool
    .help = With tls.joint: refine the TLS parameters in a separate step \
            (least-squares target) in the first macro-cycle, and jointly \
            in the following macro-cycles
  joint_refine_trace_t = True
    .type = bool
    .help = With tls.joint: refine the trace of T (equivalent to a change \
            of the u_iso of all atoms of the group, but the shifts of u_iso \
            are limited in each minimization)
}
reference_coordinate_restraints
  .short_caption = Reference coordinate restraints
{
  enabled = False
    .type = bool
    .help = Restrain the selected atoms to their starting coordinates \
            (harmonic, or top-out with top_out=True)
  selection = all
    .type = atom_selection
  exclude_outliers = True
    .type = bool
    .help = Do not restrain residues that are Ramachandran or rotamer \
            outliers in the starting model
  sigma = 0.2
    .type = float(value_min=0, allow_none=False)
    .help = Restraint sigma (A)
  limit = 1.0
    .type = float(value_min=0)
    .help = Top-out potential: deviation (A) beyond which the restraint \
            levels off
  top_out = False
    .type = bool
}
input_tls
  .short_caption = TLS in the input model
{
  mode = *auto apply ignore
    .type = choice
    .help = TLS groups and parameters of the input model (PDB REMARK 3 or \
            mmCIF). apply: the ADPs are the TLS contribution plus the \
            B-factors of the atom records (residual B-factors). ignore: the \
            ADPs as read (ANISOU records and B-factors). auto: apply if this \
            gives the lower R-work (as PDB-REDO). With tls.enabled=True, the \
            input TLS groups are refined unless tls.group is given.
}
ncs
  .short_caption = NCS restraints
{
  type = *auto none torsion cartesian
    .type = choice
    .help = NCS restraints: auto (torsion-angle NCS restraints if NCS is \
            found), none, torsion-angle NCS restraints, or Cartesian NCS \
            restraints (coordinates restrained to the NCS average, and \
            B-factors if b_factor_weight > 0). NCS groups are found \
            automatically.
  coordinate_sigma = 0.05
    .type = float(value_min=0, allow_none=False)
    .help = Cartesian NCS: sigma of the coordinate restraints (A)
  b_factor_weight = 10
    .type = float(value_min=0)
    .help = Cartesian NCS: weight of the restraints of B-factors to the NCS \
            average (0 disables them). Not applied to atoms with anisotropic \
            ADPs, nor to any atom while anisotropic ADPs are refined.
  torsion
    .short_caption = Torsion-angle NCS
  {
    include scope mmtbx.geometry_restraints.torsion_restraints.torsion_ncs.torsion_ncs_params
  }
}
outlier_rejection = True
  .type = bool
  .help = Reject outlier reflections once, in the initial bulk-solvent and \
          scaling (the data set is then kept fixed during refinement)
random_seed = 2679941
  .type = int
  .help = Seed for the random numbers used in the weight determination \
          (ADPs are perturbed before the ADP weight is computed)
remove_hydrogens = False
  .type = bool
  .help = Remove hydrogen atoms of the input model before refinement \
          (with add_hydrogens=True, all hydrogens are placed anew). \
          Hydrogens ride on their parent atoms, with the ADPs of the \
          parent atoms.
add_hydrogens = True
  .type = bool
  .help = Add riding hydrogens with mmtbx.hydrogens.reduce_hydrogen \
          (existing hydrogens are kept unless remove_hydrogens=True)
output
{
  write_map_coefficients = True
    .type = bool
    .help = Write 2mFo-DFc and mFo-DFc map coefficients (MTZ)
}
'''

# =============================================================================

class Program(ProgramTemplate):

  description = '''
Quick reciprocal-space refinement of coordinates, ADPs and occupancies.

In each macro-cycle the bulk-solvent and scale parameters are updated, then
coordinates and isotropic ADPs are refined against the X-ray target together
with geometry and ADP restraints. By default coordinates and ADPs are refined
simultaneously, and the LBFGS minimizer is preconditioned with approximate
curvatures of the target (X-ray: random-phase approximation; restraints:
Gauss-Newton approximation).

Occupancies of alternative conformations (constrained to sum to 1) and of
other atoms with partial occupancies are refined together with the
coordinates and ADPs (occupancies_simultaneous=False: in a separate step).

Optionally, torsion-angle or Cartesian NCS restraints are used
(ncs.type=torsion or ncs.type=cartesian), with automatically found groups.

Anisotropic atoms are refined anisotropically at a resolution of 1.7 A or
better, otherwise isotropically, with TLS refinement as a separate step
(refinement.adp_type, tls.enabled).

Alternatives (off by default): rigid-body refinement instead of individual
coordinates (rigid_body.enabled=True) and group ADP refinement instead of
individual ADPs (group_adp.enabled=True).

Inputs:
  - Model file (PDB or mmCIF)
  - Reflection file with R-free flags

Usage examples:
  mmtbx.quickref model.pdb data.mtz
  mmtbx.quickref model.cif data.mtz number_of_macro_cycles=10
  mmtbx.quickref model.pdb data.mtz simultaneous=False use_curvatures=False
'''

  datatypes = ['model', 'miller_array', 'restraint', 'phil']

  master_phil_str = master_phil_str

  # ---------------------------------------------------------------------------

  def validate(self):
    self.data_manager.has_models(
      raise_sorry = True, expected_n = 1, exact_count = True)
    self.data_manager.has_miller_arrays(raise_sorry = True)
    p = self.params.refinement
    if (not p.sites and not p.adp and not p.occupancies):
      raise Sorry("Nothing to refine: sites, adp and occupancies are False.")

  # ---------------------------------------------------------------------------

  def run(self):
    t0 = time.time()
    p = self.params.refinement
    log = self.logger
    random.seed(self.params.random_seed)
    flex.set_random_seed(self.params.random_seed)
    model_name = self.data_manager.get_default_model_name()
    model = self.data_manager.get_model(model_name)
    # (the model input, e.g. with TLS records, is not kept by get_fmodel)
    self.model_input = model.get_model_input()
    if (model.crystal_symmetry() is None):
      raise Sorry("The model has no crystal symmetry.")
    if (model.get_hd_selection().count(True) > 0
        and self.params.remove_hydrogens):
      model = model.select(~model.get_hd_selection())
      self.data_manager.add_model(model_name, model)
      print("Hydrogen atoms removed.", file=log)
    # get_fmodel() may reset the model (crystal symmetry), so the model
    # is processed afterwards
    fmodel = self.data_manager.get_fmodel(
      scattering_table = self.params.scattering_table,
      model_filename   = model_name)
    model.set_log(null_out()) # restraints processing is verbose
    model.process(make_restraints=True)
    if (self.params.add_hydrogens):
      model = self._add_hydrogens(model)
    if (model.get_hd_selection().count(True) > 0):
      model.setup_riding_h_manager(idealize = True)
      n_h = model.get_hd_selection().count(True)
      n_riding = len(model.riding_h_manager.parameterization_cpp)
      print("Riding hydrogens: %d%s" % (n_riding, "" if n_riding == n_h else
        " (%d hydrogens without riding parameterization are fixed)" % (
        n_h - n_riding)), file=log)
      self._riding_h_adps_from_parents(model)
    model.set_log(log)
    self.model = model
    fmodel.update_xray_structure(
      xray_structure = model.get_xray_structure(),
      update_f_calc  = True,
      update_f_mask  = True)
    assert fmodel.xray_structure is model.get_xray_structure()
    target_name = {"ml": "ml", "ls": "ls_wunit_k1"}[p.target]
    fmodel.update(target_name = target_name)
    self.fmodel = fmodel
    self._setup_reference_coordinate_restraints()
    self._setup_ncs()
    self.iso_restraints = mmtbx.refinement.adp_refinement.\
      adp_restraints_master_params.extract().iso
    self._resolve_tls_enabled()
    self._setup_input_tls()
    self._setup_adp_type()
    xrs = model.get_xray_structure()
    # (riding hydrogens are isotropic, with the ADPs of the parent atoms)
    not_h = ~model.get_hd_selection()
    n_aniso = (xrs.use_u_aniso() & not_h).count(True)
    n_iso = (xrs.use_u_iso() & not_h).count(True)
    if (p.adp and n_iso == 0 and not (n_aniso > 0 and self.refine_anisotropic)):
      print("No refinable ADPs (all atoms are anisotropic and "
        "adp_type=fixed_anisotropic): ADPs are not refined.", file=log)
      p.adp = False
    if (p.adp and n_aniso > 0):
      if (self.refine_anisotropic):
        print("%d atoms with anisotropic ADPs (refined anisotropically)." %
          n_aniso, file=log)
      else:
        print("%d atoms with anisotropic ADPs: their ADPs are kept fixed." %
          n_aniso, file=log)
    self._setup_groups()
    fmodel.update_all_scales(remove_outliers = self.params.outlier_rejection)
    print("\nStart: r_work=%.4f r_free=%.4f %s" % (
      fmodel.r_work(), fmodel.r_free(), self._geometry_summary()), file=log)
    self.r_work_start = fmodel.r_work()
    self.r_free_start = fmodel.r_free()
    self.occupancy_groups = None
    if (p.occupancies):
      self.occupancy_groups = mmtbx.refinement.occupancies.\
        occupancy_selections(model = model)
      if (self.occupancy_groups is not None
          and len(self.occupancy_groups) == 0):
        self.occupancy_groups = None
      if (self.occupancy_groups is None):
        print("No alternative conformations or partial occupancies: "
          "occupancies are not refined.", file=log)
      else:
        n_constrained = len([g for g in self.occupancy_groups if len(g) > 1])
        print("Occupancy groups: %d (%d constrained to sum to 1)" % (
          len(self.occupancy_groups), n_constrained), file=log)
    w = None
    if (self.params.weights.optimize and
        (self.individual_sites or self.individual_adp)):
      self._optimize_weights()
    r_work_previous = fmodel.r_work()
    for cycle in range(p.number_of_macro_cycles):
      if (cycle > 0):
        # outliers are rejected only once, at the start
        fmodel.update_all_scales(remove_outliers = False)
        if (model.torsion_NCS_present()):
          # torsion NCS targets (and the outlier filtering) follow the model
          model.torsion_ncs_restraints_update(log=null_out())
      if ((self.individual_sites or self.individual_adp) and
          (w is None or self.params.weights.recompute_each_macro_cycle)):
        w = self._weights()
      n_iter = self._macro_cycle(first = (cycle == 0), w = w)
      print("macro-cycle %d: r_work=%.4f r_free=%.4f %s iterations=%d" % (
        cycle+1, fmodel.r_work(), fmodel.r_free(), self._geometry_summary(),
        n_iter), file=log)
      r_work_change = abs(fmodel.r_work() - r_work_previous)
      r_work_previous = fmodel.r_work()
      if (cycle + 1 >= p.min_macro_cycles
          and cycle + 1 < p.number_of_macro_cycles
          and r_work_change < p.convergence_r_work):
        print("Converged (r_work changed by %.4f < %g): stopping after %d "
          "macro-cycles." % (r_work_change, p.convergence_r_work, cycle+1),
          file=log)
        break
    self.r_work = fmodel.r_work()
    self.r_free = fmodel.r_free()
    self._write_output()
    print("\nTime: %.1fs" % (time.time() - t0), file=log)

  def _macro_cycle(self, first, w):
    """One macro-cycle of refinement (self.fmodel, self.model); returns the
    number of LBFGS iterations of the individual refinement."""
    p = self.params.refinement
    self._first_macro_cycle = first
    if (self.rigid_body_selections is not None):
      self._refine_rigid_body()
    if (self.tls_selections is not None):
      if (self._tls_joint(first)):
        self._tls_start(first = first)
      else:
        self._refine_tls(first = first)
    sites, adp = self.individual_sites, self.individual_adp
    if (not sites and not adp):
      steps = []
    elif (sites and adp and p.simultaneous):
      steps = [(True, True)]
    else:
      steps = []
      if (sites): steps.append((True, False))
      if (adp):   steps.append((False, True))
    n_iter = 0
    occupancies_done = False
    for refine_sites, refine_adp in steps:
      groups = None
      if (self.occupancy_groups is not None and p.occupancies_simultaneous
          and refine_adp):
        groups = self.occupancy_groups
        occupancies_done = True
      n_iter += self._minimize(refine_sites, refine_adp, w,
        occupancy_groups = groups)
    if (self.group_adp_selections is not None):
      self._refine_group_adp()
    if (self.occupancy_groups is not None and not occupancies_done):
      self._refine_occupancies()
    self.model.set_xray_structure(self.fmodel.xray_structure)
    return n_iter

  def _optimize_weights(self):
    """
    wxu_scale and wxc_scale from trial refinements against a subset of the
    work set; the R-free set is not used. The test subset is excluded from
    the trial refinements, and the scales with the lowest R-factor for the
    subset are kept (parameters weights.wxu_scale and weights.wxc_scale).
    """
    from mmtbx.tls import tools
    pw = self.params.weights
    log = self.logger
    fmodel = self.fmodel
    work = ~fmodel.r_free_flags().data()
    n_work = work.count(True)
    n_test = max(pw.min_test_reflections, int(pw.test_fraction * n_work))
    if (n_test > n_work // 2):
      print("Weight optimization: too few work reflections (%d): the weights "
        "are not optimized." % n_work, file=log)
      return
    # test subset: random reflections of the work set
    fmodel_work = fmodel.select(work)
    perm = flex.random_permutation(n_work)
    test = flex.bool(n_work, False)
    test.set_selected(perm[:n_test], True)
    fmodel_work.update(r_free_flags =
      fmodel_work.r_free_flags().customized_copy(data = test))
    fmodel_work.update_all_scales(remove_outliers = False)
    max_bond_rmsd = pw.max_bond_rmsd
    if (max_bond_rmsd is None or max_bond_rmsd is Auto):
      max_bond_rmsd = [0.015, 0.02][int(fmodel.f_obs().d_min() <= 1.0)]
    print("Weight optimization: trial refinements against %d of %d work "
      "reflections (test subset: %d), max. rmsd(bonds) %g" % (
      n_work - n_test, n_work, n_test, max_bond_rmsd), file=log)
    model = self.model
    xrs_start = model.get_xray_structure().deep_copy_scatterers()
    # starting model of the trials, optionally perturbed: the test subset
    # was part of the work set the input model was refined against
    xrs_trial = xrs_start.deep_copy_scatterers()
    if (pw.trial_shake_sites > 0 or pw.trial_shake_adp > 0):
      if (pw.trial_shake_sites > 0):
        xrs_trial.shake_sites_in_place(rms_difference = pw.trial_shake_sites)
      if (pw.trial_shake_adp > 0):
        xrs_trial.shake_adp(spread = pw.trial_shake_adp,
          keep_anisotropic = True)
      if (model.riding_h_manager is not None):
        sites = xrs_trial.sites_cart()
        model.riding_h_manager.idealize_riding_h_positions(sites_cart = sites)
        xrs_trial.set_sites_cart(sites)
        self._riding_h_adps_from_parents(model, xrs_trial)
    def copy_tlsos(tlsos):
      return [tools.tlso(t = o.t, l = o.l, s = o.s, origin = o.origin)
        for o in tlsos]
    tlsos_start = None
    if (model.tls_groups is not None):
      tlsos_start = copy_tlsos(model.tls_groups.tlsos)
    w_base = self._weights()
    results = {}
    def trial(wxc_scale, wxu_scale):
      key = (wxc_scale, wxu_scale)
      if (key in results): return results[key]
      w = group_args(
        xyz_weights_result = weights_module.weights(
          wx = w_base.xyz_weights_result.wx, wx_scale = wxc_scale, w = 1),
        adp_weights_result = weights_module.weights(
          wx = w_base.adp_weights_result.wx, wx_scale = wxu_scale, w = 1))
      xrs = xrs_trial.deep_copy_scatterers()
      model.set_xray_structure(xrs)
      if (tlsos_start is not None):
        model.tls_groups.tlsos = copy_tlsos(tlsos_start)
      self.fmodel = fmodel_work.deep_copy()
      self.fmodel.update_xray_structure(xray_structure = xrs,
        update_f_calc = True, update_f_mask = True)
      try:
        for cycle in range(pw.trial_macro_cycles):
          if (cycle > 0):
            self.fmodel.update_all_scales(remove_outliers = False)
          self._macro_cycle(first = (cycle == 0), w = w)
        es = self.model.restraints_manager_energies_sites()
        r = group_args(r_work = self.fmodel.r_work(),
          r_test = self.fmodel.r_free(),
          bonds = es.geometry.bond_deviations()[2],
          geometry = self._geometry_summary())
      finally:
        self.fmodel = fmodel
      print("  wxc_scale=%-6g wxu_scale=%-6g r_work=%.4f r_test=%.4f %s%s" % (
        wxc_scale, wxu_scale, r.r_work, r.r_test, r.geometry,
        ["", " (rmsd(bonds) > %g)" % max_bond_rmsd][int(r.bonds >
          max_bond_rmsd)]), file=log)
      results[key] = r
      return r
    def scan(grid, result):
      # lowest r_test among the trials with rmsd(bonds) <= max_bond_rmsd,
      # otherwise the lowest rmsd(bonds). If the best value is at the end
      # of the grid, the grid is extended by factors of 2, at most twice.
      def key(v):
        r = result(v)
        if (r.bonds <= max_bond_rmsd): return (0, r.r_test)
        return (1, r.bonds)
      grid = sorted(set(grid))
      for extension in range(3):
        best = min(grid, key = key)
        if (extension == 2 or len(grid) < 2): break
        if   (best == grid[-1]): grid.append(best * 2)
        elif (best == grid[0]):  grid.insert(0, best / 2)
        else: break
      return best
    try:
      wxc, wxu = pw.wxc_scale, pw.wxu_scale
      if (self.individual_adp):
        wxu = scan(pw.wxu_scale_grid + [wxu], lambda v: trial(wxc, v))
      if (self.individual_sites):
        wxc = scan(pw.wxc_scale_grid + [wxc], lambda v: trial(v, wxu))
    finally:
      # the starting model
      model.set_xray_structure(xrs_start)
      if (tlsos_start is not None):
        model.tls_groups.tlsos = tlsos_start
      fmodel.update_xray_structure(xray_structure = xrs_start,
        update_f_calc = True, update_f_mask = True)
    pw.wxc_scale, pw.wxu_scale = wxc, wxu
    print("Weight optimization: wxc_scale=%g wxu_scale=%g" % (wxc, wxu),
      file=log)

  # ---------------------------------------------------------------------------

  def _setup_input_tls(self):
    """
    TLS groups and parameters of the input model. If applied, the ADPs of
    the atoms in TLS groups are U_tls + B (B: the residual B-factors of the
    atom records).
    """
    from mmtbx.tls import tools
    self.input_tls = None
    mode = self.params.input_tls.mode
    if (mode is None or mode == "ignore"): return
    log = self.logger
    model = self.model
    model_input = self.model_input
    if (model_input is None): return
    try:
      tls_params = model_input.extract_tls_params(
        model.get_hierarchy()).tls_params
    except Exception as e:
      print("TLS records of the input model not interpreted: %s" % str(e),
        file=log)
      return
    if (len(tls_params) == 0): return
    selections, strings, tlsos = [], [], []
    for t in tls_params:
      try:
        # (riding hydrogens are isotropic)
        sel = (model.selection(t.selection_string) &
               ~model.get_hd_selection())
      except Exception as e:
        print("Input TLS group not interpreted (%s): %s: TLS is not used." % (
          t.selection_string, str(e)), file=log)
        return
      if (sel.count(True) == 0): continue
      selections.append(sel.iselection())
      strings.append(t.selection_string)
      tlsos.append(tools.tlso(t=t.t, l=t.l, s=t.s, origin=t.origin))
    if (len(selections) == 0): return
    xrs = model.get_xray_structure()
    in_tls = flex.bool(xrs.scatterers().size(), False)
    for sel in selections:
      if ((in_tls.select(sel)).count(True) > 0):
        print("Input TLS groups overlap: TLS is not used.", file=log)
        return
      in_tls.set_selected(sel, True)
    tls_isel = in_tls.iselection()
    u_tls = tools.u_cart_from_tls(sites_cart = xrs.sites_cart(),
      selections = selections, tlsos = tlsos)
    u_residual = model.get_hierarchy().atoms().extract_b() * adptbx.b_as_u(1)
    xrs_tls = xrs.deep_copy_scatterers()
    xrs_tls.convert_to_anisotropic(selection = in_tls)
    u_total = xrs_tls.scatterers().extract_u_cart(xrs_tls.unit_cell())
    for i_seq in tls_isel:
      u = u_tls[i_seq]
      r = u_residual[i_seq]
      u_total[i_seq] = (u[0]+r, u[1]+r, u[2]+r, u[3], u[4], u[5])
    xrs_tls.set_u_cart(u_cart = u_total, selection = tls_isel)
    self._riding_h_adps_from_parents(model, xrs_tls)
    apply = (mode == "apply")
    if (mode == "auto"):
      def r_work(x):
        fmodel = self.fmodel.deep_copy()
        fmodel.update_xray_structure(xray_structure = x, update_f_calc = True,
          update_f_mask = True)
        fmodel.update_all_scales(remove_outliers = False)
        return fmodel.r_work()
      r_read, r_tls = r_work(xrs), r_work(xrs_tls)
      apply = (r_tls < r_read)
      print("Input TLS: %d groups; r_work with ADPs as read %.4f, with "
        "TLS + residual B %.4f: %s" % (len(selections), r_read, r_tls,
        ["ADPs as read", "TLS applied"][int(apply)]), file=log)
    else:
      print("Input TLS: %d groups, applied (TLS + residual B)" %
        len(selections), file=log)
    if (apply):
      model.set_xray_structure(xrs_tls)
      self.fmodel.update_xray_structure(xray_structure = xrs_tls,
        update_f_calc = True, update_f_mask = True)
    self.input_tls = group_args(
      applied    = apply,
      selections = selections,
      strings    = strings,
      tlsos      = tlsos,
      u_tls      = u_tls,
      u_residual = u_residual)

  def _riding_h_adps_from_parents(self, model, xrs = None):
    """Riding hydrogens are isotropic, with u_iso = U_eq of the parent atom
    (as in the minimizer, riding_h=True), from the start: the hydrogens do
    not affect the choice of refinable ADPs (adp_type). Without xrs, the
    ADPs of the model are changed."""
    if (model.riding_h_manager is None): return
    set_model = (xrs is None)
    if (set_model):
      xrs = model.get_xray_structure()
    uc = xrs.unit_cell()
    scatterers = xrs.scatterers()
    u_eq = xrs.extract_u_iso_or_u_equiv()
    for rc in model.riding_h_manager.parameterization_cpp:
      h = scatterers[rc.ih]
      if (h.flags.use_u_aniso()):
        h.convert_to_isotropic(unit_cell = uc)
      h.flags.set_use_u_iso(True)
      h.u_iso = u_eq[rc.a0]
    if (set_model):
      model.set_xray_structure(xrs)

  def _add_hydrogens(self, model):
    """Riding hydrogens (B of the parent atoms) with reduce_hydrogen."""
    from mmtbx.hydrogens import reduce_hydrogen
    n_before = model.get_hd_selection().count(True)
    result = None
    # the charged N-terminus requires the chemical components library
    for n_terminal_charge in ["residue_one", "no_charge"]:
      try:
        rh = reduce_hydrogen.place_hydrogens(model = model,
          n_terminal_charge = n_terminal_charge, keep_existing_H = True)
        rh.run()
        result = rh.get_model()
        break
      except Exception as e:
        error = e
    if (result is None):
      raise Sorry("Hydrogens could not be added: %s" % str(error))
    result.set_log(null_out())
    result.process(make_restraints = True)
    print("Hydrogens added: %d" % (
      result.get_hd_selection().count(True) - n_before), file=self.logger)
    return result

  def _resolved_adp_type(self):
    p = self.params.refinement
    adp_type = p.adp_type
    if (adp_type is None or adp_type == "auto"):
      d_min = self.fmodel.f_obs().d_min()
      adp_type = ["isotropic", "anisotropic"][int(d_min <= p.anisotropic_d_min)]
    return adp_type

  def _resolve_tls_enabled(self):
    """tls.enabled=Auto: TLS refinement if the ADPs are refined
    isotropically."""
    from libtbx import Auto
    self.tls_auto = (self.params.tls.enabled is Auto)
    if (self.tls_auto):
      self.params.tls.enabled = (self._resolved_adp_type() == "isotropic")
      if (self.params.tls.enabled):
        print("TLS refinement enabled (isotropic ADPs at d_min=%.2f)" %
          self.fmodel.f_obs().d_min(), file=self.logger)

  def _setup_adp_type(self):
    """Anisotropic or isotropic refinement of anisotropic atoms."""
    p = self.params.refinement
    log = self.logger
    d_min = self.fmodel.f_obs().d_min()
    adp_type = self._resolved_adp_type()
    self.refine_anisotropic = (adp_type == "anisotropic")
    xrs = self.model.get_xray_structure()
    aniso = xrs.use_u_aniso()
    if (adp_type == "isotropic" and aniso.count(True) > 0):
      # atoms of TLS groups that are refined with the input TLS keep the
      # TLS contribution (see _setup_groups)
      if (self.params.tls.enabled and self.input_tls is not None
          and self.input_tls.applied
          and len([s for s in self.params.tls.group if s is not None]) == 0):
        for sel in self.input_tls.selections:
          aniso.set_selected(sel, False)
      n = aniso.count(True)
      if (n > 0):
        xrs.convert_to_isotropic(selection = aniso.iselection())
        self.fmodel.update_xray_structure(update_f_calc = True)
        print("%d anisotropic atoms converted to isotropic (adp_type=%s, "
          "d_min=%.2f)." % (n, adp_type, d_min), file=log)
        if (self.input_tls is not None and not self.params.tls.enabled):
          print("  (the input model has TLS groups: consider "
            "tls.enabled=True)", file=log)

  def _chain_selections(self, selection_strings, label, min_size=1,
        allow_none=False, exclude_hydrogens=False):
    """
    Selections from selection strings, or one per chain (without water;
    chains with fewer than min_size atoms are not used), optionally without
    hydrogens.
    Returns the selections (flex.size_t) and selection strings.
    """
    model = self.model
    selection_strings = [s for s in selection_strings if s is not None]
    default = (len(selection_strings) == 0)
    if (default):
      chain_ids = []
      for chain in model.get_hierarchy().only_model().chains():
        if (chain.id not in chain_ids): chain_ids.append(chain.id)
      selection_strings = ["chain '%s' and not water" % c for c in chain_ids]
    selections, strings = [], []
    used = flex.bool(model.get_number_of_atoms(), False)
    if (exclude_hydrogens and model.get_hd_selection().count(True) > 0):
      selection_strings = ["(%s) and not (element H or element D)" % s
        for s in selection_strings]
    for s in selection_strings:
      sel = model.selection(s)
      if (sel.count(True) == 0):
        if (" and not water" in s): continue # e.g. a water chain
        raise Sorry("%s: empty selection: %s" % (label, s))
      if (default and sel.count(True) < min_size): continue
      if ((used & sel).count(True) > 0):
        raise Sorry("%s: overlapping selections (%s)" % (label, s))
      used |= sel
      selections.append(sel.iselection())
      strings.append(s)
    if (len(selections) == 0):
      if (allow_none): return None, None
      raise Sorry("%s: no groups." % label)
    return selections, strings

  def _residue_selections(self, two_per_residue):
    """Group ADP: one or two (amino acids: main chain, side chain) groups
    per residue."""
    main_chain = set([" N  ", " CA ", " C  ", " O  ", " OXT"])
    get_class = iotbx.pdb.common_residue_names_get_class
    result = []
    for rg in self.model.get_hierarchy().residue_groups():
      i_seqs = rg.atoms().extract_i_seq()
      if (two_per_residue and
          get_class(rg.atom_groups()[0].resname) == "common_amino_acid"):
        mc, sc = flex.size_t(), flex.size_t()
        for atom in rg.atoms():
          if (atom.name in main_chain): mc.append(atom.i_seq)
          else: sc.append(atom.i_seq)
        result.extend([g for g in [mc, sc] if g.size() > 0])
      else:
        result.append(i_seqs)
    return result

  def _setup_groups(self):
    """Rigid-body, group ADP and TLS groups."""
    log = self.logger
    p = self.params.refinement
    self.model.get_hierarchy().atoms().reset_i_seq()
    self.rigid_body_selections = None
    self.group_adp_selections = None
    self.tls_selections = None
    self.tls_selection_bool = None
    if (p.sites and self.params.rigid_body.enabled):
      self.rigid_body_selections = self._chain_selections(
        self.params.rigid_body.group, "Rigid body groups")[0]
      print("Rigid-body refinement: %d groups (%d atoms)" % (
        len(self.rigid_body_selections),
        sum(s.size() for s in self.rigid_body_selections)), file=log)
    self.individual_sites = (p.sites and self.rigid_body_selections is None)
    if (p.adp and self.params.group_adp.enabled):
      self.group_adp_selections = self._residue_selections(
        two_per_residue = (self.params.group_adp.mode == "two_per_residue"))
      print("Group ADP refinement: %d groups" % len(self.group_adp_selections),
        file=log)
    self.individual_adp = (p.adp and self.group_adp_selections is None)
    self.tls_presplit = False
    if (self.params.tls.enabled):
      use_input = (self.input_tls is not None and
        len([s for s in self.params.tls.group if s is not None]) == 0)
      if (use_input):
        self.tls_selections = self.input_tls.selections
        strings = self.input_tls.strings
      else:
        self.tls_selections, strings = self._chain_selections(
          self.params.tls.group, "TLS groups",
          min_size = self.params.tls.min_group_size,
          allow_none = self.tls_auto,
          # (riding hydrogens are isotropic)
          exclude_hydrogens = True)
        if (self.tls_selections is None):
          print("No TLS groups with at least %d atoms: TLS is not refined." %
            self.params.tls.min_group_size, file=log)
          self.params.tls.enabled = False
    if (self.params.tls.enabled):
      xrs = self.model.get_xray_structure()
      self.tls_selection_bool = flex.bool(xrs.scatterers().size(), False)
      for sel in self.tls_selections:
        self.tls_selection_bool.set_selected(sel, True)
      n_aniso = (xrs.use_u_aniso() & self.tls_selection_bool).count(True)
      if (use_input and self.input_tls.applied):
        # exact split of the ADPs: U_tls (anisotropic) + residual u_iso
        uc = xrs.unit_cell()
        scatterers = xrs.scatterers()
        for i_seq in self.tls_selection_bool.iselection():
          sc = scatterers[i_seq]
          sc.u_star = adptbx.u_cart_as_u_star(uc, self.input_tls.u_tls[i_seq])
          sc.u_iso = self.input_tls.u_residual[i_seq]
          sc.flags.set_use_u_iso(True)
          sc.flags.set_use_u_aniso(True)
        self.fmodel.update_xray_structure(update_f_calc = True)
        self.tls_presplit = True
        n_aniso = 0
      if (n_aniso > 0):
        # the individual contribution of atoms in TLS groups is isotropic
        xrs.convert_to_isotropic(
          selection = self.tls_selection_bool.iselection())
        self.fmodel.update_xray_structure(update_f_calc = True)
        print("%d anisotropic atoms in TLS groups made isotropic." % n_aniso,
          file=log)
      self.model.determine_tls_groups(selection_strings = strings,
        generate_tlsos = self.tls_selections)
      if (use_input):
        # start from the input TLS parameters
        self.model.tls_groups.tlsos = self.input_tls.tlsos
      print("TLS refinement: %d groups (%d atoms)%s" % (
        len(self.tls_selections), self.tls_selection_bool.count(True),
        [""," from the input model"][int(use_input)]), file=log)

  def _sync_model(self):
    """The group minimizers may replace fmodel.xray_structure."""
    self.model.set_xray_structure(self.fmodel.xray_structure)

  def _refine_rigid_body(self):
    import mmtbx.refinement.rigid_body
    p = self.params.refinement
    params = mmtbx.refinement.rigid_body.master_params.extract()
    params.max_iterations = self.params.rigid_body.max_iterations
    params.number_of_zones = self.params.rigid_body.number_of_zones
    params.bulk_solvent_and_scale = False
    mmtbx.refinement.rigid_body.manager(
      fmodel                   = self.fmodel,
      selections               = self.rigid_body_selections,
      params                   = params,
      log                      = null_out(),
      use_curvatures           = p.use_curvatures,
      curvature_floor_fraction = p.curvature_floor_fraction)
    self._sync_model()

  def _refine_group_adp(self):
    import mmtbx.refinement.group
    p = self.params.refinement
    mmtbx.refinement.group.manager(
      fmodel                   = self.fmodel,
      selections               = self.group_adp_selections,
      max_number_of_iterations = self.params.group_adp.max_iterations,
      number_of_macro_cycles   = 1,
      refine_adp               = True,
      use_restraints           = False,
      log                      = null_out(),
      use_curvatures           = p.use_curvatures,
      curvature_floor_fraction = p.curvature_floor_fraction)
    self._sync_model()

  def _refine_tls(self, first):
    """
    TLS refinement (as in phenix.refine): the ADPs of the atoms in TLS
    groups are split into the TLS contribution (U_aniso) and an individual
    isotropic contribution, which is refined in the ADP step.
    """
    # (with tls.joint=True, the TLS parameters are refined in the joint
    # minimizer instead: _tls_start() and _minimize())
    from mmtbx.tls import tools
    p = self.params.refinement
    fmodel = self.fmodel
    tls_isel = self.tls_selection_bool.iselection()
    self._tls_start(first = first, apply_tls = False)
    start_tls_value = self.model.tls_groups.tlsos
    target_name = fmodel.target_name
    fmodel.update(target_name = "ls_wunit_k1")
    try:
      tools.tls_refinement(
        fmodel                   = fmodel,
        model                    = self.model,
        selections               = self.tls_selections,
        selections_1d            = tls_isel,
        refine_T                 = True,
        refine_L                 = True,
        refine_S                 = True,
        number_of_macro_cycles   = 1,
        max_number_of_iterations = self.params.tls.max_iterations,
        start_tls_value          = start_tls_value,
        out                      = null_out(),
        verbose                  = False,
        use_curvatures           = p.use_curvatures,
        curvature_floor_fraction = p.curvature_floor_fraction)
    finally:
      fmodel.update(target_name = target_name)
    self._sync_model()
    if (self.model.riding_h_manager is not None):
      self._riding_h_adps_from_parents(self.model)
      fmodel.update_xray_structure(xray_structure =
        self.model.get_xray_structure(), update_f_calc = True)

  def _tls_joint(self, first):
    """True if the TLS parameters are refined in the joint minimizer in
    this macro-cycle (tls.joint)."""
    p = self.params.tls
    return p.joint and not (first and p.joint_ls_first_cycle)

  def _tls_start(self, first, apply_tls = True):
    """
    At the start (first macro-cycle, unless the input TLS are applied): the
    ADPs of the atoms of the TLS groups are split into an anisotropic
    (U_tls) and an isotropic contribution, and the TLS parameters are
    fitted to the anisotropic contribution. With apply_tls, U_tls is then
    set from the TLS parameters (for tls.joint=True).
    """
    from mmtbx.tls import tools
    if (not first or self.tls_presplit): return
    fmodel = self.fmodel
    xrs = fmodel.xray_structure
    tls_isel = self.tls_selection_bool.iselection()
    scatterers = xrs.scatterers()
    for i_seq in tls_isel:
      sc = scatterers[i_seq]
      if (not sc.flags.use_u_aniso()):
        sc.u_star = (0,0,0,0,0,0)
      sc.flags.set_use_u_iso(True)
      sc.flags.set_use_u_aniso(True)
    tools.split_u(xrs, self.tls_selections, offset = True)
    fmodel.update_xray_structure(update_f_calc = True)
    # The TLS parameters are fitted to the ADPs only at the start; then the
    # refinement starts from the current TLS parameters (or from those of
    # the input model). (Without start_tls_value, tls_refinement fits them
    # to the ADPs again in each macro-cycle (tls_from_u_cart), which takes
    # about as long as the TLS refinement itself and does not lower R-free;
    # it would fit with 100 macro-cycles of up to 100 iterations for each
    # of three parameter subsets, most of them after convergence; small
    # groups may not converge.)
    self.model.tls_groups.tlsos = tools.tls_from_u_cart(
      xray_structure         = xrs,
      tlsos_initial          = self.model.tls_groups.tlsos,
      tls_selections         = self.tls_selections,
      number_of_macro_cycles = 10,
      max_iterations         = 100,
      convergence_tolerance  = 1.e-6)
    if (apply_tls):
      tools.update_xray_structure_with_tls(
        xray_structure = xrs,
        selections     = self.tls_selections,
        tlsos          = self.model.tls_groups.tlsos,
        selections_1d  = tls_isel,
        correct_adp    = False)
      fmodel.update_xray_structure(update_f_calc = True)
      self._sync_model()
      if (self.model.riding_h_manager is not None):
        self._riding_h_adps_from_parents(self.model)
        fmodel.update_xray_structure(xray_structure =
          self.model.get_xray_structure(), update_f_calc = True)

  def _tls_after_joint_minimization(self):
    """tls.joint=True: U_tls with the refined coordinates; non-positive
    definite ADPs are corrected as in mmtbx.tls.tools.tls_refinement."""
    from mmtbx.tls import tools
    model = self.model
    xrs = model.get_xray_structure()
    tls_isel = self.tls_selection_bool.iselection()
    tools.update_xray_structure_with_tls(
      xray_structure = xrs,
      selections     = self.tls_selections,
      tlsos          = model.tls_groups.tlsos,
      selections_1d  = tls_isel,
      correct_adp    = False)
    if (xrs.is_positive_definite_u().count(False) > 0):
      xrs.tidy_us(u_min = 1.e-6)
      model.tls_groups.tlsos = tools.make_tlso_compatible_with_u_positive_definite(
        tlsos                            = model.tls_groups.tlsos,
        xray_structure                   = xrs.deep_copy_scatterers(),
        selections                       = self.tls_selections,
        max_iterations                   = 10,
        number_of_u_nonpositive_definite = 0,
        eps                              = 1.e-6,
        refine_T                         = True,
        refine_L                         = True,
        refine_S                         = True,
        out                              = null_out(),
        number_of_macro_cycles_for_tls_from_uanisos = 10)
      tools.update_xray_structure_with_tls(
        xray_structure = xrs,
        selections     = self.tls_selections,
        tlsos          = model.tls_groups.tlsos,
        selections_1d  = tls_isel)
    model.set_xray_structure(xrs)
    if (model.riding_h_manager is not None):
      self._riding_h_adps_from_parents(model)
    self.fmodel.update_xray_structure(
      xray_structure = model.get_xray_structure(), update_f_calc = True)

  def _setup_reference_coordinate_restraints(self):
    """Restraints of the selected atoms to the starting coordinates."""
    p = self.params.reference_coordinate_restraints
    if (not p.enabled):
      return
    from mmtbx.geometry_restraints.reference import \
      add_coordinate_restraints, \
      exclude_outliers_from_reference_restraints_selection
    model = self.model
    # (riding hydrogens follow their parent atoms)
    selection = model.selection(p.selection) & ~model.get_hd_selection()
    if (p.exclude_outliers):
      hierarchy = model.get_hierarchy()
      hierarchy.atoms().reset_i_seq()
      selection = exclude_outliers_from_reference_restraints_selection(
        pdb_hierarchy        = hierarchy,
        restraints_selection = selection)
    isel = selection.iselection()
    if (isel.size() == 0):
      raise Sorry("No atoms selected for reference coordinate restraints.")
    proxies = add_coordinate_restraints(
      sites_cart        = model.get_sites_cart().select(isel),
      selection         = isel,
      sigma             = p.sigma,
      limit             = p.limit,
      top_out_potential = p.top_out)
    model.get_restraints_manager().geometry.reference_coordinate_proxies = \
      proxies
    print("Reference coordinate restraints: %d atoms (sigma=%.2f%s)" % (
      isel.size(), p.sigma,
      (" top-out limit=%.2f" % p.limit) if p.top_out else ""), file=self.logger)

  def _setup_ncs(self):
    """NCS restraints with automatically found NCS groups."""
    p = self.params.ncs
    if (p.type is None or p.type == "none"): # phil choice "none" is None
      return
    log = self.logger
    model = self.model
    model.search_for_ncs(log=null_out())
    groups = model.get_ncs_groups()
    if (groups is None or len(groups) == 0):
      if (p.type != "auto"):
        print("No NCS found: NCS restraints are not used.", file=log)
      return
    ncs_type = p.type
    if (ncs_type == "auto"):
      ncs_type = "torsion"
    print("NCS groups: %d (%s)" % (len(groups), ", ".join(
      ["%d copies" % (g.get_number_of_copies()+1) for g in groups])),
      file=log)
    if (ncs_type == "cartesian"):
      from mmtbx.ncs.ncs_params import global_ncs_params
      ncs_params = global_ncs_params.extract().ncs
      ncs_params.type = "cartesian"
      ncs_params.coordinate_sigma = p.coordinate_sigma
      ncs_params.b_factor_weight = p.b_factor_weight
      model.setup_cartesian_ncs_groups(ncs_params=ncs_params, log=null_out())
      print("Cartesian NCS restraints: coordinate_sigma=%.3f "
        "b_factor_weight=%s" % (p.coordinate_sigma, p.b_factor_weight),
        file=log)
    elif (ncs_type == "torsion"):
      model.setup_torsion_ncs_restraints(
        fmodel             = self.fmodel,
        ncs_torsion_params = p.torsion,
        sites_individual   = flex.bool(model.get_number_of_atoms(), True),
        log                = null_out())
      if (model.torsion_NCS_present()):
        print("Torsion NCS restraints: %d dihedrals (sigma=%.1f "
          "limit=%.1f degrees)" % (
            model.get_restraints_manager().geometry.ncs_dihedral_manager.
              get_n_proxies(), p.torsion.sigma, p.torsion.limit), file=log)
      else:
        print("No torsion NCS restraints.", file=log)

  def _set_flags(self, refine_sites, refine_adp):
    model = self.model
    xrs = model.get_xray_structure()
    n = xrs.scatterers().size()
    # u_iso (u_aniso) flags are only set for isotropic (anisotropic) atoms;
    # for atoms in TLS groups only the isotropic contribution is refined
    aniso = self._individual_aniso_selection()
    refine_aniso = (refine_adp and self.refine_anisotropic
                    and aniso.count(True) > 0)
    refine_iso = (refine_adp and
      (xrs.use_u_iso() & ~model.get_hd_selection()).count(True) > 0)
    xray.set_scatterer_grad_flags(
      scatterers = xrs.scatterers(), site = refine_sites, u_iso = refine_adp,
      u_aniso = refine_aniso)
    if (self.tls_selection_bool is not None):
      scatterers = xrs.scatterers()
      for i_seq in self.tls_selection_bool.iselection():
        scatterers[i_seq].flags.set_grad_u_aniso(False)
    model.set_refinement_flags(mmtbx.refinement.refinement_flags.manager(
      individual_sites     = refine_sites,
      individual_adp       = refine_adp,
      sites_individual     = flex.bool(n, refine_sites),
      adp_individual_iso   = (xrs.use_u_iso() if refine_iso else None),
      adp_individual_aniso = (aniso if refine_aniso else None)))

  def _individual_aniso_selection(self):
    """Atoms with individual anisotropic ADPs (not TLS)."""
    result = self.model.get_xray_structure().use_u_aniso()
    if (self.tls_selection_bool is not None):
      result = result & ~self.tls_selection_bool
    return result

  def _weights(self):
    """
    Relative weights as in phenix.refine: the X-ray target is weighted by
    wxc = wxc_scale*|g_geometry|/|g_xray| (coordinates) or
    wxu = wxu_scale*|g_adp_restraints|/|g_xray| (ADPs).
    """
    p = self.params.weights
    result = group_args(
      xyz_weights_result = weights_module.weights(wx=1, wx_scale=1, w=1),
      adp_weights_result = weights_module.weights(wx=1, wx_scale=1, w=1))
    if (self.individual_sites):
      self._set_flags(refine_sites=True, refine_adp=False)
      g_x = self.fmodel.target_functor()(compute_gradients=True).\
        d_target_d_site_cart()
      # geometry restraints only: the (stiff) Cartesian NCS restraints
      # would inflate the weight
      rm = self.model.get_restraints_manager()
      ncs_manager = rm.cartesian_ncs_manager
      rm.cartesian_ncs_manager = None
      try:
        g_c = self.model.restraints_manager_energies_sites(
          compute_gradients=True).gradients
      finally:
        rm.cartesian_ncs_manager = ncs_manager
      # riding hydrogens are not parameters
      not_h = ~self.model.get_hd_selection()
      g_x, g_c = g_x.select(not_h), g_c.select(not_h)
      if (g_c.norm() == 0 or g_x.norm() == 0):
        wxc = 1.0
      else:
        wxc = g_c.norm() / g_x.norm()
      result.xyz_weights_result = weights_module.weights(
        wx=wxc, wx_scale=p.wxc_scale, w=1)
    if (self.individual_adp):
      self._set_flags(refine_sites=False, refine_adp=True)
      # As in mmtbx.refinement.weights: gradients are computed after
      # shaking the ADPs (otherwise the ADP restraints gradients are small
      # for a smooth ADP distribution), using only C, N, O, S, P atoms
      # (heavy atoms may overwhelm the gradients).
      model = self.model.deep_copy()
      fmodel = self.fmodel.deep_copy()
      xrs = fmodel.xray_structure
      xrs.shake_adp_if_all_equal(b_iso_tolerance = 1.e-3)
      xrs.shake_adp(spread=10, keep_anisotropic=True)
      model.set_xray_structure(xrs)
      fmodel.update_xray_structure(xray_structure=xrs, update_f_calc=True)
      tr = fmodel.target_functor()(compute_gradients=True)
      # ADP restraints only, without the NCS B-factor restraints
      model.get_restraints_manager().cartesian_ncs_manager = None
      ea = model.energies_adp(iso_restraints=self.iso_restraints,
        compute_gradients=True, use_hd=False)
      scattering_types = xrs.scatterers().extract_scattering_types()
      sel_use = flex.bool(scattering_types.size(), False)
      for t in ["C", "N", "O", "S", "P"]:
        sel_use |= (scattering_types == t)
      g_x, g_a = flex.double(), flex.double()
      sel = sel_use & xrs.use_u_iso()
      if (sel.count(True) == 0): sel = xrs.use_u_iso()
      if (ea.u_iso_gradients is not None and sel.count(True) > 0):
        g_x.extend(tr.gradients_wrt_atomic_parameters(u_iso=True).select(sel))
        g_a.extend(ea.u_iso_gradients.select(sel))
      if (ea.u_aniso_gradients is not None):
        aniso = self._individual_aniso_selection()
        sel = sel_use & aniso
        if (sel.count(True) == 0): sel = aniso
        g_x.extend(tr.gradients_wrt_atomic_parameters(u_aniso=True).
          select(sel).as_double())
        g_a.extend(ea.u_aniso_gradients.select(sel).as_double())
      if (g_a.size() == 0 or g_a.norm() == 0 or g_x.norm() == 0):
        wxu = 1.0
      else:
        wxu = g_a.norm() / g_x.norm()
      result.adp_weights_result = weights_module.weights(
        wx=wxu, wx_scale=p.wxu_scale, w=1)
    return result

  def _minimize(self, refine_sites, refine_adp, w, occupancy_groups=None):
    p = self.params.refinement
    self._set_flags(refine_sites, refine_adp)
    tls_groups = None
    if (refine_adp and self.tls_selections is not None
        and self._tls_joint(first = self._first_macro_cycle)):
      tls_groups = group_args(
        selections = self.tls_selections,
        tlsos      = self.model.tls_groups.tlsos)
    self.fmodel.update_xray_structure(
      xray_structure=self.model.get_xray_structure(), update_f_calc=True)
    m = minimization.lbfgs(
      fmodels                  = mmtbx.fmodels(fmodel_xray=self.fmodel),
      restraints_manager       = self.model.restraints_manager,
      model                    = self.model,
      target_weights           = w,
      refine_xyz               = refine_sites,
      refine_adp               = refine_adp,
      iso_restraints           = self.iso_restraints,
      h_params                 = group_args(refine="riding"),
      lbfgs_termination_params = scitbx.lbfgs.termination_parameters(
        max_iterations = p.max_iterations),
      collect_monitor          = False,
      log                      = null_out(),
      use_curvatures           = p.use_curvatures,
      curvature_floor_fraction = p.curvature_floor_fraction,
      occupancy_groups         = occupancy_groups,
      occupancy_min            = p.occupancy_min,
      occupancy_max            = p.occupancy_max,
      smooth_target            = p.smooth_target,
      riding_h                 = (self.model.riding_h_manager is not None),
      tls_groups               = tls_groups,
      tls_refine_trace_t       = self.params.tls.joint_refine_trace_t)
    if (tls_groups is not None):
      self.model.tls_groups.tlsos = m.tls_result
      self._tls_after_joint_minimization()
    return m.minimizer.iter()

  def _refine_occupancies(self):
    p = self.params.refinement
    model = self.model
    xrs = model.get_xray_structure()
    model.set_refinement_flags(mmtbx.refinement.refinement_flags.manager(
      occupancies   = True,
      s_occupancies = self.occupancy_groups))
    self.fmodel.update_xray_structure(xray_structure=xrs, update_f_calc=True)
    mmtbx.refinement.occupancies.manager(
      fmodels                  = mmtbx.fmodels(fmodel_xray=self.fmodel),
      model                    = model,
      max_number_of_iterations = p.occupancy_max_iterations,
      number_of_macro_cycles   = 1,
      occupancy_max            = p.occupancy_max,
      occupancy_min            = p.occupancy_min,
      log                      = None,
      use_curvatures           = p.use_curvatures,
      curvature_floor_fraction = p.curvature_floor_fraction)
    assert self.fmodel.xray_structure is model.get_xray_structure()

  def _geometry_summary(self):
    es = self.model.restraints_manager_energies_sites()
    return "rmsd(bonds)=%.4f rmsd(angles)=%.2f" % (
      es.geometry.bond_deviations()[2], es.geometry.angle_deviations()[2])

  # ---------------------------------------------------------------------------

  def _write_output(self):
    log = self.logger
    prefix = self.params.output.prefix
    if (prefix is None):
      prefix = os.path.splitext(os.path.basename(
        self.data_manager.get_default_model_name()))[0] + "_quickref"
    self.params.output.prefix = prefix
    model_out = self.model
    if (self.model.tls_groups is not None):
      # TLS groups and parameters (REMARK 3)
      model_out = self.model.tls_groups_as_pdb() + self.model.model_as_pdb()
    self.model_file_name = self.data_manager.write_model_file(
      model_out, filename = prefix + ".pdb", format = "pdb")
    print("\nWrote %s" % self.model_file_name, file=log)
    self.mtz_file_name = None
    if (self.params.output.write_map_coefficients):
      import iotbx.mtz
      # conventional labels (e.g. for Coot): 2FOFCWT,PH2FOFCWT FOFCWT,PHFOFCWT
      decorator = iotbx.mtz.label_decorator(phases_prefix="PH")
      emap = self.fmodel.electron_density_map()
      mtz = None
      for map_type, label in [("2mFo-DFc", "2FOFCWT"), ("mFo-DFc", "FOFCWT")]:
        mc = emap.map_coefficients(map_type=map_type, isotropize=False,
          fill_missing=False)
        if (mtz is None):
          mtz = mc.as_mtz_dataset(column_root_label=label,
            label_decorator=decorator)
        else:
          mtz.add_miller_array(mc, column_root_label=label,
            label_decorator=decorator)
      self.mtz_file_name = self.data_manager.write_miller_array_file(
        mtz.mtz_object(), filename = prefix + ".mtz")
      print("Wrote %s" % self.mtz_file_name, file=log)

  # ---------------------------------------------------------------------------

  def get_results(self):
    return group_args(
      r_work_start    = self.r_work_start,
      r_free_start    = self.r_free_start,
      r_work          = self.r_work,
      r_free          = self.r_free,
      model           = self.model,
      fmodel          = self.fmodel,
      model_file_name = self.model_file_name,
      mtz_file_name   = self.mtz_file_name)
