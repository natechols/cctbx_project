"""Quick reciprocal-space refinement of coordinates, ADPs and occupancies"""
from __future__ import absolute_import, division, print_function
from libtbx.program_template import ProgramTemplate
from libtbx.utils import Sorry, null_out
from libtbx import group_args, Auto
from cctbx import xray, adptbx
from cctbx.array_family import flex
import iotbx.pdb
import mmtbx
import mmtbx.f_model
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
  occupancy_states = Auto
    .type = bool
    .help = Multi-state ensembles (e.g. PanDDA: ground state A, B and \
            bound state C, D in many residues, the ligand only in C, D): \
            refine one occupancy per altloc, shared by all atoms with that \
            altloc and constrained to sum to 1, instead of independent \
            groups per residue. Auto: if such an ensemble is detected (each \
            altloc with the same occupancy everywhere, see \
            mmtbx.refinement.occupancies.ensemble_states).
  occupancy_states_fix_d_min = 2.5
    .type = float(value_min=0)
    .help = Multi-state ensembles (occupancy_states): if the resolution \
            (d_min) is lower than this, the occupancies of the states are \
            not refined (at low resolution they are strongly correlated \
            with the ADPs, and the bound state was refined to zero, e.g. \
            PDB 5QDM at 2.65 A). 0: always refined.
  identical_conformers_sigma = 0.02
    .type = float(value_min=0)
    .help = Multi-state ensembles (occupancy_states): conformers of a \
            residue that are identical in the input model (e.g. copies of \
            the ground state in the bound state) are restrained to stay \
            identical, with this sigma (A) for the coordinates and \
            identical_conformers_sigma_b for the B-factors (otherwise the \
            copies drift apart and absorb the differences between the \
            states). 0: not restrained.
  identical_conformers_sigma_b = 1.0
    .type = float(value_min=0)
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
  shake_aniso_b = 10
    .type = float(value_min=0)
    .help = For the ADP weight, a random isotropic component (B up to \
            +-shake_aniso_b) is added to the anisotropic ADPs, in addition \
            to the shaking of each component by +-10%, comparable to the \
            shaking of isotropic ADPs (+-10 A**2). Otherwise the weight is \
            too small for models refined anisotropically (smooth ADPs).
  regularize_ncycles = 0
    .type = int(value_min=0)
    .help = If > 0, the gradients for wxc are computed on a copy of the \
            model regularized against the geometry restraints with this \
            number of LBFGS iterations (after shaking the sites, see \
            shake_sites), like the "memory eraser" of phenix.refine: \
            the gradients of the input model reflect the previous \
            refinement (small X-ray gradients) and its geometry errors \
            (large geometry gradients), which both inflate wxc.
  shake_sites = 0
    .type = float(value_min=0)
    .help = Mean distance (A) by which the sites of the copy are shaken \
            before the regularization (regularize_ncycles > 0)
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
  find_groups = False
    .type = bool
    .help = If no tls.group is given: TLS groups from \
            mmtbx.command_line.find_tls_groups (segments of the chains \
            fitted to the ADPs of the starting model) instead of the TLS \
            groups of the input model or one group per chain
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
force_field
  .short_caption = Molecular mechanics force field
  .help = Experimental: OpenMM force field instead of the geometry \
          restraints of the protein and nucleic-acid residues (see \
          mmtbx.geometry_restraints.openmm_restraints). Coordinates and ADPs \
          are then refined in alternation, and the coordinates without \
          curvatures.
{
  include scope mmtbx.geometry_restraints.openmm_restraints.master_params
}
bulk_solvent_map
  .short_caption = User-supplied bulk solvent
  .help = Bulk-solvent structure factors from a file (amplitudes and \
          phases, e.g. from an explicit-solvent MD simulation), used instead \
          of the flat-mask F_mask in F_model = k_total*(F_calc + \
          k_mask*F_mask). k_mask is still determined per resolution bin. \
          The mask is not recomputed from the model. Equivalent to \
          FPART/PHIPART in Refmac.
{
  file_name = None
    .type = path
    .help = Reflection file with the bulk-solvent amplitudes and phases \
            (can be the data file)
  amplitudes_label = FPART
    .type = str
  phases_label = PHIPART
    .type = str
    .help = Phases in degrees
}
twinning
  .short_caption = Twinning check
{
  check = True
    .type = bool
    .help = Check for twinning (Xtriage intensity statistics and twin-law \
            tests): if twinning is suspected, the R-factors of the starting \
            model with and without each suspicious twin law are compared, \
            and a warning is printed if a twin law lowers R-free. The \
            refinement itself does not use twin laws.
  min_twin_fraction = 0.05
    .type = float(value_min=0, value_max=0.5)
    .help = Twin laws with a larger estimated twin fraction (H or Britton \
            test) are suspicious, and a refined twin fraction above this \
            is required for the warning
  min_r_free_decrease = 0.005
    .type = float(value_min=0)
    .help = Minimum decrease of R-free with the twin law for the warning
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
anomalous
  .short_caption = Anomalous data
{
  use_anomalous_data = True
    .type = bool
    .help = If the data are anomalous (F+ and F-, or I+ and I-), use the \
            Friedel mates as separate observations (otherwise they are \
            merged). An anomalous difference map is then written \
            (output.write_map_coefficients).
  set_inelastic_form_factors = False
    .type = bool
    .help = With anomalous data: set f' and f'' of all atoms from \
            form_factor_table at the wavelength (anomalous.wavelength, or \
            from the reflection file), so that the anomalous differences \
            are modeled.
  wavelength = None
    .type = float(value_min=0)
    .help = X-ray wavelength (A) for set_inelastic_form_factors (default: \
            from the reflection file)
  form_factor_table = *sasaki henke
    .type = choice
}
output
{
  write_map_coefficients = True
    .type = bool
    .help = Write 2mFo-DFc and mFo-DFc map coefficients (MTZ)
  clashscore = False
    .type = bool
    .help = Compute the MolProbity clashscore of the starting and final \
            models, with the riding hydrogens of the refinement (with probe \
            if it is installed, otherwise with mmtbx.probe2, which is slower)
}
'''

# =============================================================================

def shake_aniso_isotropically(xrs, spread_b):
  """Adds a random isotropic component (B in [-spread_b, spread_b], limited
  to half of the smallest eigenvalue for negative shifts) to the anisotropic
  ADPs, comparable to the shaking of isotropic ADPs."""
  uc = xrs.unit_cell()
  du_max = adptbx.b_as_u(spread_b)
  for sc in xrs.scatterers():
    if (not sc.flags.use_u_aniso()): continue
    u_cart = adptbx.u_star_as_u_cart(uc, sc.u_star)
    lo = -0.5 * min(adptbx.eigenvalues(u_cart))
    du = random.uniform(max(-du_max, lo), du_max)
    u_cart = [u + (du if i < 3 else 0) for i, u in enumerate(u_cart)]
    sc.u_star = adptbx.u_cart_as_u_star(uc, u_cart)

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
  mmtbx.quickref model.pdb data.mtz bulk_solvent_map.file_name=solvent.mtz
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
    unknown = flex.bool([e.strip().upper() == "X"
      for e in model.get_hierarchy().atoms().extract_element()])
    if (unknown.count(True) > 0):
      # e.g. UNX (unknown atom or ion): no scattering factor
      model = model.select(~unknown)
      self.data_manager.add_model(model_name, model)
      print("WARNING: %d atoms of unknown element (X, e.g. UNX) removed." %
        unknown.count(True), file=log)
    negative = model.get_hierarchy().atoms().extract_occ() < 0
    if (negative.count(True) > 0):
      # e.g. -99 for unobserved atoms in old entries (PDB 1M6C)
      model.set_occupancies(values = 0, selection = negative)
      self.data_manager.add_model(model_name, model)
      print("WARNING: %d atoms with negative occupancies: set to 0." %
        negative.count(True), file=log)
    if (model.get_hd_selection().count(True) > 0
        and self.params.remove_hydrogens):
      model = model.select(~model.get_hd_selection())
      self.data_manager.add_model(model_name, model)
      print("Hydrogen atoms removed.", file=log)
    if (not self.params.anomalous.use_anomalous_data):
      fmodel_params = self.data_manager.get_fmodel_params()
      fmodel_params.xray_data.force_anomalous_flag_to_be_equal_to = False
      self.data_manager.set_fmodel_params(fmodel_params)
    # get_fmodel() may reset the model (crystal symmetry), so the model
    # is processed afterwards
    fmodel = self.data_manager.get_fmodel(
      scattering_table = self.params.scattering_table,
      model_filename   = model_name)
    # get_fmodel() uses the twin law of the model file (e.g. the mmCIF
    # _pdbx_reflns_twin.operator), but the refinement does not use twin laws
    twin_law = getattr(fmodel, "twin_law_str", None)
    if (twin_law is not None):
      fmodel = mmtbx.f_model.manager(
        f_obs          = fmodel.f_obs(),
        r_free_flags   = fmodel.r_free_flags(),
        xray_structure = fmodel.xray_structure,
        mask_params    = fmodel.mask_params,
        sf_and_grads_accuracy_params = fmodel.sfg_params,
        origin         = fmodel.origin())
      print("Twin law of the model file (%s) not used." % twin_law, file=log)
    self._setup_user_bulk_solvent(fmodel)
    model.set_log(null_out()) # restraints processing is verbose
    # residues without restraint definitions do not stop the refinement:
    # their coordinates are fixed (see _setup_fixed_sites)
    model.set_stop_for_unknowns(False)
    if (self.params.add_hydrogens):
      # the restraints are made after the hydrogens are placed; only the
      # flips of symmetric side chains done by processing are needed before
      model.get_hierarchy().flip_symmetric_amino_acids()
      model = self._add_hydrogens(model)
    else:
      model.process(make_restraints=True)
    if (model.get_hd_selection().count(True) > 0):
      model.setup_riding_h_manager(idealize = True)
      n_h = model.get_hd_selection().count(True)
      n_riding = len(model.riding_h_manager.parameterization_cpp)
      print("Riding hydrogens: %d%s" % (n_riding, "" if n_riding == n_h else
        " (%d hydrogens without riding parameterization are fixed)" % (
        n_h - n_riding)), file=log)
      self._riding_h_adps_from_parents(model)
    self._remove_linear_dihedrals(model)
    self._setup_fixed_sites(model)
    self._setup_anomalous(model, fmodel)
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
    self._setup_force_field()
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
    self.geometry_start = self._geometry_rmsds()
    print("\nStart: r_work=%.4f r_free=%.4f %s" % (
      fmodel.r_work(), fmodel.r_free(),
      self._geometry_summary(self.geometry_start)), file=log)
    self.r_work_start = fmodel.r_work()
    self.r_free_start = fmodel.r_free()
    self.clashscore_start = self._clashscore()
    self.twinning = None
    if (self.params.twinning.check):
      self._check_twinning()
    self.occupancy_groups = None
    if (p.occupancies):
      self.occupancy_groups = mmtbx.refinement.occupancies.\
        occupancy_selections(model = model)
      self._setup_occupancy_states(model)
      if (self.occupancy_groups is not None
          and len(self.occupancy_groups) == 0):
        self.occupancy_groups = None
      if (self.occupancy_groups is None):
        if (self.occupancy_states is None):
          print("No alternative conformations or partial occupancies: "
            "occupancies are not refined.", file=log)
        else:
          print("No other occupancies to refine.", file=log)
      else:
        n_constrained = len([g for g in self.occupancy_groups if len(g) > 1])
        print("Occupancy groups: %d (%d constrained to sum to 1)" % (
          len(self.occupancy_groups), n_constrained), file=log)
    w = None
    if (self.params.weights.optimize and
        (self.individual_sites or self.individual_adp)):
      self._optimize_weights()
    r_work_previous = fmodel.r_work()
    # implicit solvent of the force field only in the final macro-cycle
    solvent_pending = (getattr(self, "force_field", None) is not None
      and self.force_field.solvent_final_cycle)
    solvent_final_cycle = False
    for cycle in range(p.number_of_macro_cycles):
      if (solvent_pending and cycle + 1 == p.number_of_macro_cycles):
        self._final_cycle_solvent()
        solvent_pending = False
      if (cycle > 0):
        # The bulk-solvent mask of the starting model is kept until the end:
        # it keeps surface atoms out of the solvent region. Recomputing it
        # in each macro-cycle lets exposed atoms drift (25-50% larger
        # shifts) and raises R-free at low resolution (PDB 1BDJ, 2.7 A:
        # 0.2375 -> 0.2517), at the cost of optimistic R-factors for the
        # macro-cycles (see the final R-factors).
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
      if (solvent_final_cycle):
        break # (the macro-cycle with the implicit solvent is the last one)
      if (cycle + 1 >= p.min_macro_cycles
          and cycle + 1 < p.number_of_macro_cycles
          and r_work_change < p.convergence_r_work):
        if (solvent_pending):
          # one more macro-cycle, with the implicit solvent
          print("Converged (r_work changed by %.4f < %g) after %d "
            "macro-cycles: final macro-cycle with the implicit solvent." % (
            r_work_change, p.convergence_r_work, cycle+1), file=log)
          self._final_cycle_solvent()
          solvent_pending = False
          solvent_final_cycle = True
          continue
        print("Converged (r_work changed by %.4f < %g): stopping after %d "
          "macro-cycles." % (r_work_change, p.convergence_r_work, cycle+1),
          file=log)
        break
    # R-factors and map coefficients of the final model. The update is
    # forced: by default the mask is recomputed only if the mean shift since
    # the last mask exceeds 0.1 A.
    fmodel.update_xray_structure(update_f_calc = True, update_f_mask = True,
      force_update_f_mask = True)
    fmodel.update_all_scales(remove_outliers = False)
    self.geometry_final = self._geometry_rmsds()
    print("Final: r_work=%.4f r_free=%.4f %s" % (fmodel.r_work(),
      fmodel.r_free(), self._geometry_summary(self.geometry_final)), file=log)
    self.clashscore_final = self._clashscore()
    if (getattr(self, "occupancy_states", None) is not None):
      occ = fmodel.xray_structure.scatterers().extract_occupancies()
      print("Multi-state ensemble, final occupancies: %s" % " ".join([
        "%s=%.2f" % (a, flex.mean(occ.select(sel))) for a, sel in zip(
        self.occupancy_states.altlocs, self.occupancy_states.selections)]),
        file=log)
    if (self.clashscore_final is not None):
      print("Clashscore: start=%.2f final=%.2f" % (self.clashscore_start,
        self.clashscore_final), file=log)
    if (self.fixed_sites_selection is not None):
      print("WARNING: the coordinates of %d atoms without geometry restraints "
        "were fixed\n(see the warning at the start)." %
        self.fixed_sites_selection.count(True), file=log)
    if (self.twinning is not None and self.twinning.confirmed):
      print("WARNING: the data are probably twinned; the twin law(s) were not "
        "used\n(see the warning at the start).", file=log)
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
        selection = None
        if (self.fixed_sites_selection is not None):
          selection = ~self.fixed_sites_selection
        xrs_trial.shake_sites_in_place(rms_difference = pw.trial_shake_sites,
          selection = selection)
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
        update_f_calc = True, update_f_mask = True, force_update_f_mask = True)
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
        update_f_calc = True, update_f_mask = True, force_update_f_mask = True)
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

  def _setup_user_bulk_solvent(self, fmodel):
    """User-supplied bulk-solvent structure factors (bulk_solvent_map) in
    place of the flat-mask F_mask: aligned to the F_obs indices (reflections
    missing from the file contribute nothing)."""
    p = self.params.bulk_solvent_map
    if (p.file_name is None): return
    from iotbx import reflection_file_reader
    log = self.logger
    arrays = reflection_file_reader.any_reflection_file(
      file_name = p.file_name).as_miller_arrays()
    def find(label):
      for ma in arrays:
        if (label in ma.info().labels): return ma
      raise Sorry("Column %s not found in %s (columns: %s)" % (label,
        p.file_name, ", ".join([ma.info().label_string() for ma in arrays])))
    amplitudes = find(p.amplitudes_label)
    phases = find(p.phases_label)
    if (amplitudes is phases):
      # amplitudes and phases combined (as for adjacent F and P MTZ columns)
      if (not amplitudes.is_complex_array()):
        raise Sorry("%s and %s in %s are not amplitudes and phases." % (
          p.amplitudes_label, p.phases_label, p.file_name))
      f_user = amplitudes
    else:
      f_user = amplitudes.phase_transfer(phase_source = phases, deg = True)
    f_obs = fmodel.f_obs()
    if (f_user.space_group_info() is not None and
        f_user.space_group().type().number() !=
        f_obs.space_group().type().number()):
      raise Sorry("The space group of %s (%s) differs from that of the "
        "data (%s)." % (p.file_name, str(f_user.space_group_info()),
        str(f_obs.space_group_info())))
    f_user = f_user.customized_copy(
      crystal_symmetry = f_obs.crystal_symmetry()).map_to_asu()
    if (f_obs.anomalous_flag() and not f_user.anomalous_flag()):
      f_user = f_user.generate_bijvoet_mates()
    f_mask = f_user.matching_set(other = f_obs, data_substitute = 0j,
      assert_is_similar_symmetry = False)
    n_missing = f_obs.size() - f_obs.common_set(f_user).size()
    fmodel.set_user_f_masks([f_mask])
    print("Bulk solvent: user-supplied structure factors %s,%s from %s "
      "(d_min=%.2f; the mask is not computed from the model)" % (
      p.amplitudes_label, p.phases_label, p.file_name, f_user.d_min()),
      file=log)
    if (n_missing > 0):
      print("  WARNING: %d of %d reflections are not in %s (no bulk-solvent "
        "contribution)" % (n_missing, f_obs.size(), p.file_name), file=log)

  def _check_twinning(self):
    """
    Twinning check (does not change the refinement): Xtriage intensity
    statistics and twin-law dependent tests. If twinning is suspected
    (abnormal intensity statistics with possible twin laws, or a twin law
    with an estimated twin fraction above twinning.min_twin_fraction), the
    R-factors of the current model without and with each suspicious twin
    law (twin fraction refined, least-squares twin target) are computed
    for the same (merged) data, and a warning with the comparison is printed
    if a twin law lowers R-free by at least twinning.min_r_free_decrease.
    The result is kept in self.twinning (None if twinning is not suspected
    or the check could not be done).
    """
    import mmtbx.utils
    p = self.params.twinning
    log = self.logger
    fmodel = self.fmodel
    try:
      f_obs = fmodel.f_obs()
      flags = fmodel.r_free_flags()
      if (f_obs.anomalous_flag()):
        f_obs = f_obs.average_bijvoet_mates()
        flags = flags.customized_copy(
          data = flags.data().as_double()).average_bijvoet_mates()
        flags = flags.customized_copy(data = flags.data() > 0)
      summary = f_obs.f_as_f_sq().analyze_intensity_statistics()
      if (summary is None or summary.n_twin_laws == 0): return
      abnormal = summary.has_abnormal_intensity_statistics()
      suspicious = []
      for i, law in enumerate(summary.twin_laws):
        alphas = [a for a in [summary.h_alpha[i], summary.britton_alpha[i]]
          if a is not None]
        if (abnormal or (len(alphas) > 0 and
                         max(alphas) > p.min_twin_fraction)):
          suspicious.append((law, summary.h_alpha[i],
            summary.britton_alpha[i]))
      if (len(suspicious) == 0): return
      def r_factors(twin_law):
        fm = mmtbx.utils.fmodel_manager2(
          f_obs               = f_obs,
          r_free_flags        = flags,
          abcd                = None,
          xray_structure      = fmodel.xray_structure.deep_copy_scatterers(),
          twin_law            = twin_law,
          ignore_r_free_flags = False,
          mask_params         = None,
          sf_accuracy_params  = None)
        fm.update_all_scales(log = null_out())
        return (fm.r_work(), fm.r_free(),
          getattr(fm, "twin_fraction", None) if twin_law else None)
      r_none = r_factors(None)
      rows = []
      for law, h_alpha, britton_alpha in suspicious:
        rows.append((law, h_alpha, britton_alpha) + r_factors(law))
    except Exception as e:
      print("Twinning check not done: %s" % str(e), file=log)
      return
    confirmed = [r for r in rows if (r[5] is not None and
      r[5] > p.min_twin_fraction and r[4] <= r_none[1] - p.min_r_free_decrease)]
    self.twinning = group_args(
      abnormal_intensity_statistics = abnormal,
      l_mean    = summary.l_mean,
      r_no_twin = r_none,
      twin_laws = rows,
      confirmed = (len(confirmed) > 0))
    fmt = lambda x: "  -  " if x is None else "%5.3f" % x
    width = max([10] + [len(r[0]) for r in rows])
    def show():
      print("  %-*s %7s %7s %8s %7s %7s" % (width, "twin law", "H-test",
        "Britton", "refined", "r_work", "r_free"), file=log)
      print("  %-*s %7s %7s %8s %7.4f %7.4f" % (width, "none", "", "", "",
        r_none[0], r_none[1]), file=log)
      for law, h_alpha, britton_alpha, r_work, r_free, fraction in rows:
        print("  %-*s %7s %7s %8s %7.4f %7.4f" % (width, law, fmt(h_alpha),
          fmt(britton_alpha), fmt(fraction), r_work, r_free), file=log)
    stats = "intensity statistics %s (<|L|>=%.3f)" % (
      ["normal", "abnormal"][int(abnormal)], summary.l_mean)
    if (not self.twinning.confirmed):
      print("Twinning check: %s;\nR-factors of the starting model with "
        "the possible twin laws (twin fractions:\nestimated by Xtriage and "
        "refined):" % stats, file=log)
      show()
      print("  No twin law lowers R-free by %g or more." %
        p.min_r_free_decrease, file=log)
      return
    print("\n" + "*" * 79, file=log)
    print("WARNING: the data are probably twinned\n(%s).\nR-factors of the "
      "starting model without and with the twin laws (least-squares\ntwin "
      "target; twin fractions: estimated by Xtriage and refined):" % stats,
      file=log)
    show()
    print("quickref does not refine against twinned data: the refinement "
      "below ignores\nthe twin law(s).", file=log)
    print("*" * 79, file=log)

  def _setup_anomalous(self, model, fmodel):
    """Anomalous data: report, and optionally set f' and f'' of the atoms
    (anomalous.set_inelastic_form_factors)."""
    p = self.params.anomalous
    log = self.logger
    f_obs = fmodel.f_obs()
    if (not f_obs.anomalous_flag()):
      return
    print("Anomalous data: %d Bijvoet pairs (F+ and F- are separate "
      "observations)" % f_obs.n_bijvoet_pairs(), file=log)
    if (not p.set_inelastic_form_factors):
      return
    wavelength = p.wavelength
    if (wavelength is None and f_obs.info() is not None):
      wavelength = getattr(f_obs.info(), "wavelength", None)
    if (wavelength is None or wavelength <= 0):
      print("  No wavelength (anomalous.wavelength): f' and f'' are not "
        "set.", file=log)
      return
    xrs = model.get_xray_structure()
    xrs.set_inelastic_form_factors(photon = wavelength,
      table = p.form_factor_table)
    model.set_xray_structure(xrs)
    values = {}
    for sc in xrs.scatterers():
      values.setdefault(sc.scattering_type, (sc.fp, sc.fdp))
    print("  f' and f'' at %.4f A (%s):" % (wavelength, p.form_factor_table),
      file=log)
    for t in sorted(values, key = lambda t: -values[t][1]):
      print("    %-4s f'=%6.3f f''=%6.3f" % (t, values[t][0], values[t][1]),
        file=log)

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

  def _remove_linear_dihedrals(self, model):
    """
    Dihedral restraints through a linear angle (e.g. an alkyne in a ligand
    dictionary) are undefined: their gradients diverge, which blew up
    refinements (PDB 6W0U, S0Y).
    """
    rm = model.get_restraints_manager()
    if (rm is None or rm.geometry is None): return
    removed = rm.geometry.remove_dihedrals_through_linear_angles()
    if (removed.size() == 0): return
    atoms = model.get_hierarchy().atoms()
    print("Dihedral restraints through a linear angle removed: %d" %
      removed.size(), file=self.logger)
    for proxy in removed:
      print("  %s" % " - ".join([atoms[i].id_str() for i in proxy.i_seqs]),
        file=self.logger)

  def _setup_fixed_sites(self, model):
    """Atoms without geometry restraints (residues or atoms without restraint
    definitions: unknown nonbonded energy types) are not moved by the
    refinement, and a warning lists them. Their ADPs and occupancies are
    refined. self.fixed_sites_selection is None if all atoms have
    restraints."""
    self.fixed_sites_selection = None
    unrestrained = (model._type_energies == "")
    if (unrestrained.count(True) == 0): return
    self.fixed_sites_selection = unrestrained
    residues = []
    for rg in model.get_hierarchy().residue_groups():
      for ag in rg.atom_groups():
        n = unrestrained.select(ag.atoms().extract_i_seq()).count(True)
        if (n > 0):
          altloc = "" if ag.altloc == "" else " altloc %s" % ag.altloc
          residues.append("%s %s %s%s (%d of %d atoms)" % (ag.resname,
            rg.parent().id.strip(), rg.resid().strip(), altloc, n,
            ag.atoms_size()))
    log = self.logger
    print("\n" + "*" * 79, file=log)
    print("WARNING: %d atoms have no geometry restraints (no restraint "
      "definitions\nfor these residues or atoms). Their coordinates are "
      "FIXED during refinement\n(their ADPs and occupancies are refined):" %
      unrestrained.count(True), file=log)
    for r in residues:
      print("  " + r, file=log)
    if (self.params.rigid_body.enabled):
      print("(In rigid-body refinement they move with their rigid-body "
        "groups.)", file=log)
    print("To refine these atoms, supply restraint definitions (CIF files, "
      "e.g. from\nphenix.elbow or the Grade server) as additional input "
      "files.", file=log)
    print("*" * 79, file=log)

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
    result.set_stop_for_unknowns(False)
    # hydrogens on special positions (e.g. PDB 1DJY: GLN side chain next to
    # its symmetry copy on a 2-fold axis) are removed: restraints processing
    # does not allow polymer atoms on special positions
    # (with the tolerance of restraints processing)
    from cctbx import crystal
    sps = crystal.special_position_settings(
      crystal_symmetry = result.crystal_symmetry(),
      min_distance_sym_equiv = result.get_current_pdb_interpretation_params().\
        pdb_interpretation.min_distance_sym_equiv)
    hd_isel = result.get_hd_selection().iselection()
    sst = sps.site_symmetry_table(
      sites_cart = result.get_sites_cart().select(hd_isel))
    h_special = flex.bool(result.get_number_of_atoms(), hd_isel.select(
      sst.special_position_indices()))
    if (h_special.count(True) > 0):
      atoms = result.get_hierarchy().atoms()
      print("WARNING: %d hydrogens on special positions removed:\n%s" % (
        h_special.count(True), "\n".join(["  " + atoms[i].id_str()
        for i in h_special.iselection()])), file=self.logger)
      result = result.select(~h_special)
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

  def _find_tls_groups(self):
    """TLS groups from mmtbx.command_line.find_tls_groups (selection
    strings; empty if none are found)."""
    from mmtbx.command_line import find_tls_groups
    log = self.logger
    model = self.model
    not_h = ~model.get_hd_selection()
    # (a copy: reset_i_seq() would change the atoms of the model)
    hierarchy = model.get_hierarchy().deep_copy().select(not_h)
    hierarchy.atoms().reset_i_seq()
    xrs = model.get_xray_structure().select(not_h).deep_copy_scatterers()
    params = find_tls_groups.master_phil.extract()
    params.nproc = 1
    params.random_seed = self.params.random_seed
    try:
      result = find_tls_groups.find_tls(
        params                   = params,
        pdb_hierarchy            = hierarchy,
        xray_structure           = xrs,
        return_as_list           = True,
        ignore_pdb_header_groups = True,
        out                      = null_out())
    except Sorry as e:
      print("TLS groups not found (%s): default groups are used." % str(e),
        file=log)
      return []
    print("TLS groups found (find_tls_groups): %d" % len(result), file=log)
    for s in result:
      print("  %s" % s, file=log)
    return list(result)

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
      group_strings = [s for s in self.params.tls.group if s is not None]
      if (len(group_strings) == 0 and self.params.tls.find_groups):
        group_strings = self._find_tls_groups()
      use_input = (self.input_tls is not None and len(group_strings) == 0)
      if (use_input):
        self.tls_selections = self.input_tls.selections
        strings = self.input_tls.strings
      else:
        self.tls_selections, strings = self._chain_selections(
          group_strings, "TLS groups",
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
    # tls_refinement makes non-positive definite ADPs positive definite
    # (tidy_us) and then refits the TLS parameters to them, but does not
    # apply the refitted parameters: U_tls of the atoms was then not the U_tls
    # of the TLS parameters of the model (and of the REMARK 3 records of the
    # output file; PDB 1D3H: |dU| up to 0.2 A**2). U_tls is set from the TLS
    # parameters (with the riding hydrogens and fmodel).
    self._tls_after_joint_minimization()

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
    """U_tls from the TLS parameters (after the joint minimization with
    tls.joint=True: with the refined coordinates); non-positive definite ADPs
    are corrected as in mmtbx.tls.tools.tls_refinement, and the TLS
    parameters made compatible with them are applied."""
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

  def _setup_force_field(self):
    """Force field (OpenMM) instead of the geometry restraints of the
    macromolecule: coordinates and ADPs are refined in alternation."""
    self.force_field = None
    if (not self.params.force_field.enabled):
      return
    from mmtbx.geometry_restraints import openmm_restraints
    log = self.logger
    model = self.model
    if (model.get_hd_selection().count(True) == 0):
      raise Sorry("The force field requires hydrogens (add_hydrogens=True).")
    self.force_field = openmm_restraints.setup(model = model,
      params = self.params.force_field, log = log)
    # (hydrogens moved by optimize_hydrogens)
    self.fmodel.update_xray_structure(
      xray_structure = model.get_xray_structure(), update_f_calc = True)
    p = self.params.refinement
    if (p.sites and p.adp and p.simultaneous):
      p.simultaneous = False
      print("  coordinates and ADPs are refined in alternation "
        "(simultaneous=False)", file=log)

  def _final_cycle_solvent(self):
    """Implicit solvent of the force field for the final macro-cycle
    (force_field.implicit_solvent_final_cycle)."""
    solvent = self.params.force_field.implicit_solvent
    self.force_field.set_solvent(solvent)
    print("Implicit solvent (%s) switched on: E_mm=%.0f" % (solvent,
      self._geometry_rmsds().mm_energy), file=self.logger)

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
    scatterers = xrs.scatterers()
    if (self.tls_selection_bool is not None):
      for i_seq in self.tls_selection_bool.iselection():
        scatterers[i_seq].flags.set_grad_u_aniso(False)
    sites_individual = flex.bool(n, refine_sites)
    if (refine_sites and self.fixed_sites_selection is not None):
      # atoms without geometry restraints
      for i_seq in self.fixed_sites_selection.iselection():
        scatterers[i_seq].flags.set_grad_site(False)
      sites_individual = ~self.fixed_sites_selection
    model.set_refinement_flags(mmtbx.refinement.refinement_flags.manager(
      individual_sites     = refine_sites,
      individual_adp       = refine_adp,
      sites_individual     = sites_individual,
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
      model, fmodel = self.model, self.fmodel
      if (p.regularize_ncycles > 0):
        model, fmodel = self._regularized_copy()
      g_x = fmodel.target_functor()(compute_gradients=True).\
        d_target_d_site_cart()
      # geometry restraints only: the (stiff) Cartesian NCS restraints
      # would inflate the weight
      rm = model.get_restraints_manager()
      ncs_manager = rm.cartesian_ncs_manager
      rm.cartesian_ncs_manager = None
      # with the force field: the geometry restraints of all atoms (the
      # force-field gradients of the input model are dominated by a few
      # strained atoms, which inflated wxc up to 200-fold)
      use_force_field = getattr(rm, "use_force_field", None)
      if (use_force_field is not None):
        rm.use_force_field = False
      try:
        g_c = model.restraints_manager_energies_sites(
          compute_gradients=True).gradients
      finally:
        rm.cartesian_ncs_manager = ncs_manager
        if (use_force_field is not None):
          rm.use_force_field = use_force_field
      # riding hydrogens are not parameters
      not_h = ~self.model.get_hd_selection()
      g_x, g_c = g_x.select(not_h), g_c.select(not_h)
      if (g_c.norm() == 0 or g_x.norm() == 0):
        wxc = 1.0
      else:
        wxc = g_c.norm() / g_x.norm()
      print("  weights: wxc=%.4g (|g_geometry|=%.4g |g_xray|=%.4g) "
        "wxc_scale=%g" % (wxc, g_c.norm(), g_x.norm(), p.wxc_scale),
        file=self.logger)
      result.xyz_weights_result = weights_module.weights(
        wx=wxc, wx_scale=p.wxc_scale, w=1)
    if (self.individual_adp):
      self._set_flags(refine_sites=False, refine_adp=True)
      # As in mmtbx.refinement.weights: gradients are computed after
      # shaking the ADPs (otherwise the ADP restraints gradients are small
      # for a smooth ADP distribution), using only C, N, O, S, P atoms
      # (heavy atoms may overwhelm the gradients). Anisotropic ADPs are
      # shaken too (each U* component by +-10%, and isotropically, see
      # shake_aniso_b): models refined anisotropically have smooth ADPs,
      # and the weight was too small (e.g. PDB 4YUH: wxu 0.001, ADPs
      # over-restrained, R-work increased).
      model = self.model.deep_copy()
      fmodel = self.fmodel.deep_copy()
      xrs = fmodel.xray_structure
      xrs.shake_adp_if_all_equal(b_iso_tolerance = 1.e-3)
      xrs.shake_adp(spread=10, keep_anisotropic=False)
      if (p.shake_aniso_b > 0):
        shake_aniso_isotropically(xrs, p.shake_aniso_b)
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
      print("  weights: wxu=%.4g (|g_adp|=%.4g |g_xray|=%.4g) wxu_scale=%g" % (
        wxu, g_a.norm(), g_x.norm(), p.wxu_scale), file=self.logger)
      result.adp_weights_result = weights_module.weights(
        wx=wxu, wx_scale=p.wxu_scale, w=1)
    return result

  def _regularized_copy(self):
    """
    Copies of self.model and self.fmodel with the sites shaken (shake_sites)
    and regularized against the geometry restraints (regularize_ncycles
    LBFGS iterations), for the gradients of the weights.
    """
    from cctbx import geometry_restraints
    from mmtbx.refinement import geometry_minimization
    p = self.params.weights
    model = self.model.deep_copy()
    fmodel = self.fmodel.deep_copy()
    xrs = fmodel.xray_structure
    # hydrogen atoms are idealized after the regularization (the geometry
    # minimizer does not support riding hydrogens with a selection)
    selection = ~model.get_hd_selection()
    if (self.fixed_sites_selection is not None):
      selection &= ~self.fixed_sites_selection
    if (p.shake_sites > 0):
      xrs.shake_sites_in_place(mean_distance = p.shake_sites,
        selection = selection)
    sites_cart = xrs.sites_cart()
    rmsd_start = self._bond_angle_rmsds(model, sites_cart)
    geometry_minimization.lbfgs(
      sites_cart                  = sites_cart,
      geometry_restraints_manager = model.get_restraints_manager().geometry,
      geometry_restraints_flags   = geometry_restraints.flags.flags(
        default = True),
      lbfgs_termination_params    = scitbx.lbfgs.termination_parameters(
        max_iterations = p.regularize_ncycles),
      correct_special_position_tolerance = 1.0,
      sites_cart_selection        = selection,
      lbfgs_exception_handling_params = scitbx.lbfgs.\
        exception_handling_parameters(
          ignore_line_search_failed_step_at_lower_bound = True,
          ignore_line_search_failed_step_at_upper_bound = True))
    if (model.riding_h_manager is not None):
      model.riding_h_manager.idealize_riding_h_positions(
        sites_cart = sites_cart)
    xrs.set_sites_cart(sites_cart)
    model.set_sites_cart(sites_cart)
    fmodel.update_xray_structure(xray_structure = xrs, update_f_calc = True)
    rmsd_final = self._bond_angle_rmsds(model, sites_cart)
    print("  weights: copy shaken by %.2f A and regularized (%d iterations): "
      "rmsd(bonds) %.4f -> %.4f, rmsd(angles) %.2f -> %.2f" % (
      p.shake_sites, p.regularize_ncycles, rmsd_start[0], rmsd_final[0],
      rmsd_start[1], rmsd_final[1]), file=self.logger)
    return model, fmodel

  def _bond_angle_rmsds(self, model, sites_cart):
    es = model.get_restraints_manager().geometry.energies_sites(
      sites_cart = sites_cart, compute_gradients = False)
    return es.bond_deviations()[2], es.angle_deviations()[2]

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
    # (the force field has no curvatures)
    use_curvatures = p.use_curvatures and not (refine_sites and
      getattr(self, "force_field", None) is not None)
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
      use_curvatures           = use_curvatures,
      curvature_floor_fraction = p.curvature_floor_fraction,
      occupancy_groups         = occupancy_groups,
      occupancy_min            = p.occupancy_min,
      occupancy_max            = p.occupancy_max,
      smooth_target            = p.smooth_target,
      riding_h                 = (self.model.riding_h_manager is not None),
      tls_groups               = tls_groups,
      tls_refine_trace_t       = self.params.tls.joint_refine_trace_t,
      equal_pairs              = getattr(self, "equal_pairs", None))
    if (tls_groups is not None):
      self.model.tls_groups.tlsos = m.tls_result
      self._tls_after_joint_minimization()
    return m.minimizer.iter()

  def _setup_occupancy_states(self, model):
    """Occupancy groups of multi-state ensembles (occupancy_states)."""
    p = self.params.refinement
    self.occupancy_states = None
    self.equal_pairs = None
    if (p.occupancy_states is False): return
    kwds = {}
    if (p.occupancy_states is True):
      kwds = dict(min_altlocs = 2, min_complete_residues = 1)
    states = mmtbx.refinement.occupancies.ensemble_states(
      hierarchy = model.get_hierarchy(), **kwds)
    if (states is None):
      if (p.occupancy_states is True):
        raise Sorry("occupancy_states=True: the model is not a multi-state "
          "ensemble (each altloc must have the same occupancy everywhere, "
          "summing to 1 over the altlocs).")
      return
    self.occupancy_groups = mmtbx.refinement.occupancies.\
      ensemble_occupancy_groups(
        groups  = self.occupancy_groups,
        states  = states,
        n_atoms = model.get_number_of_atoms())
    self.occupancy_states = states
    print("Multi-state ensemble: one occupancy per altloc (%s), constrained "
      "to sum to 1\n  start: %s; %d residues in all states, %d in some "
      "(excluding waters)" % (", ".join(states.altlocs),
      " ".join(["%s=%.2f" % (a, o) for a, o in zip(states.altlocs,
        states.occupancies)]), states.n_complete, states.n_partial),
      file=self.logger)
    d_min = self.fmodel.f_obs().d_min()
    if (p.occupancy_states_fix_d_min > 0
        and d_min > p.occupancy_states_fix_d_min):
      # (the group of the states is the last one)
      self.occupancy_groups = self.occupancy_groups[:-1]
      print("  WARNING: the occupancies of the states are not refined "
        "(d_min=%.2f > occupancy_states_fix_d_min=%.2f)" % (d_min,
        p.occupancy_states_fix_d_min), file=self.logger)
    if (p.identical_conformers_sigma > 0):
      pairs = mmtbx.refinement.occupancies.identical_conformer_pairs(
        hierarchy = model.get_hierarchy(),
        altlocs   = states.altlocs)
      if (pairs.i_seqs.size() > 0):
        self.equal_pairs = group_args(
          i_seqs     = pairs.i_seqs,
          j_seqs     = pairs.j_seqs,
          sigma_site = p.identical_conformers_sigma,
          sigma_b    = p.identical_conformers_sigma_b)
        print("  %d conformers in %d residues are identical to another "
          "conformer: restrained to stay identical" % (pairs.n_conformers,
          pairs.n_residues), file=self.logger)

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

  def _geometry_rmsds(self):
    """Rmsd of the bond lengths and angles from the restraints (all atoms,
    including riding hydrogens)."""
    es = self.model.restraints_manager_energies_sites()
    return group_args(
      bonds  = es.geometry.bond_deviations()[2],
      angles = es.geometry.angle_deviations()[2],
      mm_energy = getattr(es, "mm_energy", None))

  def _geometry_summary(self, rmsds=None):
    if (rmsds is None):
      rmsds = self._geometry_rmsds()
    result = "rmsd(bonds)=%.4f rmsd(angles)=%.2f" % (rmsds.bonds, rmsds.angles)
    if (rmsds.mm_energy is not None):
      result += " E_mm=%.0f" % rmsds.mm_energy
    return result

  # ---------------------------------------------------------------------------

  def _clashscore(self):
    """MolProbity clashscore of the current model (output.clashscore), with
    its riding hydrogens, so that Reduce is not run. None if not requested or
    if it cannot be computed."""
    if (not self.params.output.clashscore):
      return None
    import libtbx.load_env
    log = self.logger
    hierarchy = self.model.get_hierarchy().deep_copy()
    hierarchy.atoms().set_xyz(self.model.get_sites_cart())
    try:
      if (libtbx.env.has_module("probe")):
        from mmtbx.validation.clashscore import clashscore
        result = clashscore(pdb_hierarchy = hierarchy, fast = True,
          condensed_probe = True, keep_hydrogens = True, out = null_out())
      else:
        import iotbx.phil
        from iotbx.data_manager import DataManager
        from mmtbx.programs import clashscore2 as clashscore2_program
        from mmtbx.validation.clashscore2 import clashscore2
        import mmtbx.model
        model = mmtbx.model.manager(
          model_input       = None,
          pdb_hierarchy     = hierarchy,
          crystal_symmetry  = self.model.crystal_symmetry(),
          restraint_objects = self.model.get_restraint_objects(),
          log               = null_out())
        dm = DataManager(["model"])
        dm.add_model("model", model)
        params = iotbx.phil.parse(
          clashscore2_program.Program.master_phil_str,
          process_includes = True).extract()
        result = clashscore2(params.probe, data_manager = dm, fast = True,
          condensed_probe = True, keep_hydrogens = True, out = null_out())
      return result.get_clashscore()
    except Exception as e:
      print("Clashscore not computed: %s" % str(e), file=log)
      return None

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
      maps = [("2mFo-DFc", "2FOFCWT"), ("mFo-DFc", "FOFCWT")]
      if (self.fmodel.f_obs().anomalous_flag()):
        # anomalous difference map (model phases)
        maps.append(("anom", "ANOM"))
      for map_type, label in maps:
        mc = emap.map_coefficients(map_type=map_type, isotropize=False,
          fill_missing=False)
        if (mc.anomalous_flag()):
          # (one map: the Friedel mates are averaged)
          mc = mc.average_bijvoet_mates()
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
      geometry_start  = self.geometry_start,
      geometry_final  = self.geometry_final,
      clashscore_start = self.clashscore_start,
      clashscore_final = self.clashscore_final,
      model           = self.model,
      fmodel          = self.fmodel,
      model_file_name = self.model_file_name,
      mtz_file_name   = self.mtz_file_name,
      twinning        = self.twinning)
