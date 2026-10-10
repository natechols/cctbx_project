from __future__ import absolute_import, division, print_function
from cctbx import xray
from cctbx import crystal
from cctbx.array_family import flex
import scitbx.lbfgs
from scitbx.lbfgs.curvature_preconditioner import curvature_preconditioner
from libtbx import adopt_init_args
from libtbx import group_args
import math
import sys
from libtbx.utils import user_plus_sys_time
from cctbx import adptbx
from libtbx.str_utils import format_value
from libtbx.utils import Sorry
import mmtbx.refinement.minimization_ncs_constraints

def curvatures_reflection_stride(n_reflections, n_target=5000):
  """
  Stride for the direct sums over reflections in
  cctbx.xray.structure.curvatures_random_phase() (anisotropic scatterers):
  about n_target reflections are sufficient (~1% error).
  """
  return max(1, n_reflections // n_target)

class lbfgs(object):

  def __init__(self, fmodels,
                     restraints_manager       = None,
                     model                    = None,
                     is_neutron_scat_table    = None,
                     target_weights           = None,
                     refine_xyz               = False,
                     refine_adp               = False,
                     lbfgs_termination_params = None,
                     use_fortran              = False,
                     verbose                  = 0,
                     correct_special_position_tolerance = 1.0,
                     iso_restraints           = None,
                     h_params                 = None,
                     qblib_params             = None,
                     macro_cycle              = None,
                     u_min                    = adptbx.b_as_u(-5.0),
                     u_max                    = adptbx.b_as_u(999.99),
                     collect_monitor          = True,
                     log                      = None,
                     use_curvatures           = False,
                     curvature_floor_fraction = 0.1,
                     occupancy_groups         = None,
                     occupancy_min            = 0.0,
                     occupancy_max            = 1.0,
                     smooth_target            = False,
                     riding_h                 = False):
    """
    occupancy_groups (e.g. from mmtbx.refinement.occupancies.
    occupancy_selections) are refined simultaneously with the coordinates
    and/or ADPs: one occupancy per selection, with the selections of a
    group constrained to sum to 1 (same parameterization as in
    mmtbx.refinement.occupancies). The occupancies are limited to
    [occupancy_min, occupancy_max] at the end.

    If use_curvatures is True, the minimizer is preconditioned with
    approximate curvatures (diagonal of the 2nd derivative matrix): the
    random-phase approximation for the X-ray target (computed once, see
    cctbx.xray.structure.curvatures_random_phase()) and the geometry
    restraints curvatures (see cctbx.geometry_restraints.energies) or the
    ADP restraints curvatures (see mmtbx.restraints.manager.energies_adp_iso
    and cctbx.adp_restraints.adp_aniso_restraints). Parameters without
    curvatures are handled as described in
    scitbx.lbfgs.curvature_preconditioner.

    If smooth_target is True, the target is a smooth function of the
    parameters during the minimization (otherwise the line search fails
    near convergence, where the remaining decrease of the target is
    smaller than its discontinuities):
      - the pair table of the ADP restraints (local spheres) is fixed
        during the minimization (updated at the start); otherwise it is
        updated together with the nonbonded pair proxies, and pairs
        entering or leaving the spheres change the target discontinuously;
      - the normalization of the geometry and ADP restraints (by the
        number of restraints, which includes the nonbonded pairs) is fixed
        at its value at the start of the minimization.
    The truncation of u_iso shifts to [u_min, u_max] is not smooth (the
    target is flat along truncated parameters) but is kept: without it the
    ADPs drift much further in each minimization, which made the results
    worse at low resolution (the truncation acts as a regularization).

    If riding_h is True (requires model.riding_h_manager), the hydrogen
    atoms ride on their parent atoms and are not parameters: after each
    shift their positions are idealized from the parent atoms, and their
    ADPs (u_iso, and u_aniso if the parent is anisotropic) are those of the
    parent atoms. The gradients with respect to the hydrogen positions are
    added to those of the parent atoms (and their neighbors) with the chain
    rule of the riding model; the gradients and curvatures with respect to
    the hydrogen ADPs, and the curvatures with respect to the hydrogen
    positions, are added to those of the parent atoms. This also works for
    simultaneous refinement of coordinates, ADPs and occupancies (unlike
    the riding model without riding_h, which keeps the hydrogen positions
    as parameters with zero shifts).
    """
    timer = user_plus_sys_time()
    adopt_init_args(self, locals())
    self.f=None
    self.xray_structure = self.fmodels.fmodel_xray().xray_structure
    self.fmodels.create_target_functors()
    self.fmodels.prepare_target_functors_for_minimization()
    if(self.refine_adp and fmodels.fmodel_neutron() is None):
      self.xray_structure.tidy_us()
      self.fmodels.update_xray_structure(
        xray_structure = self.xray_structure,
        update_f_calc  = True)
    self.weights = None
# QBLIB INSERT
    self.qblib_params = qblib_params
    if(self.qblib_params is not None and self.qblib_params.qblib):
        self.macro = macro_cycle
        self.qblib_cycle_count = 0
        self.tmp_XYZ = None
        self.XYZ_diff_curr=None
# QBLIB END
    self.correct_special_position_tolerance = correct_special_position_tolerance
    if(refine_xyz and target_weights is not None):
      self.weights = target_weights.xyz_weights_result
    elif(refine_adp and target_weights is not None):
      self.weights = target_weights.adp_weights_result
    else:
      from mmtbx.refinement import weights
      self.weights = weights.weights(wx = 1, wx_scale = 1, w = 0)
    # Weight of the ADP restraints. For simultaneous refinement of
    # coordinates and ADPs the targets of the separate refinements,
    #   wxc*Exray + wc*Echem  and  wxu*Exray + wu*Eadp,
    # are combined on the scale of the first:
    #   wxc*Exray + wc*Echem + wxc/wxu*wu*Eadp
    self.adp_restraints_weight = self.weights.w
    if(refine_xyz and refine_adp):
      if(self.riding_h_manager_is_present() and not riding_h):
        # the riding model without riding_h assumes that the parameters are
        # the coordinates only (see apply_shifts)
        raise Sorry("Simultaneous refinement of coordinates and ADPs with"
          " riding hydrogens requires riding_h=True.")
      if(target_weights is not None):
        wa = target_weights.adp_weights_result
        self.adp_restraints_weight = wa.w * (
          self.weights.wx * self.weights.wx_scale) / (wa.wx * wa.wx_scale)
    if(self.collect_monitor):
      self.monitor = monitor(
        weights        = self.weights,
        fmodels        = fmodels,
        model          = model,
        iso_restraints = iso_restraints,
        refine_xyz     = refine_xyz,
        refine_adp     = refine_adp,
        refine_occ     = False)
    if(self.collect_monitor): self.monitor.collect()
    self.neutron_refinement = (self.fmodels.fmodel_n is not None)
    self.riding_h_manager = self.model.riding_h_manager
    self._riding = None
    if(riding_h):
      if(self.riding_h_manager is None):
        raise Sorry("riding_h=True requires a riding hydrogen manager.")
      self._setup_riding_h()
    self._setup_occupancy_groups()
    self._scatterers_start = self.xray_structure.scatterers()
    self._fixed_normalization = {}
    geometry = None
    if(smooth_target and self.restraints_manager is not None):
      geometry = self.restraints_manager.geometry
    plain_pairs_radius = None
    if(geometry is not None and geometry.plain_pairs_radius is not None):
      plain_pairs_radius = geometry.plain_pairs_radius
      if(self.refine_adp):
        geometry.update_plain_pair_sym_table(
          sites_frac = self.xray_structure.sites_frac())
      geometry.plain_pairs_radius = None # no updates during minimization
    self.curvatures = None
    if(use_curvatures):
      if(self.neutron_refinement):
        raise Sorry("Curvatures are not available for neutron refinement.")
      self._xray_curvatures = None
      self.curvature_preconditioner = curvature_preconditioner(
        floor_fraction = curvature_floor_fraction)
      self.diag_mode = "always"
    #lbfgs_core_params = scitbx.lbfgs.core_parameters(
    #  stpmin = 1.e-9,
    #  stpmax = adptbx.b_as_u(10))
    try:
      self.minimizer = scitbx.lbfgs.run(
        target_evaluator          = self,
        termination_params        = lbfgs_termination_params,
        use_fortran               = use_fortran,
      #  core_params = lbfgs_core_params,
      #  gradient_only=True,
        exception_handling_params = scitbx.lbfgs.exception_handling_parameters(
                           ignore_line_search_failed_step_at_lower_bound = True))
    finally:
      if(plain_pairs_radius is not None):
        geometry.plain_pairs_radius = plain_pairs_radius
    self.apply_shifts()
    if(self._occ is not None and len(self._occ.groups) > 0):
      self._finalize_occupancies()
    del self._scatterers_start
    self.compute_target(compute_gradients = False,u_iso_refinable_params = None)
    if(self.refine_adp and self.fmodels.fmodel_neutron() is None):
      self.xray_structure.tidy_us()
      self.fmodels.update_xray_structure(
        xray_structure = self.xray_structure,
        update_f_calc  = True)
    if(self.collect_monitor):
      self.monitor.collect(iter = self.minimizer.iter(),
                           nfun = self.minimizer.nfun())
    self.fmodels.create_target_functors()
# QBLIB INSERT
    if(self.qblib_params is not None and self.qblib_params.qblib):
       print('{:-^80}'.format(""), file=self.qblib_params.qblib_log)
       print(file=self.qblib_params.qblib_log)
# QBLIB END

  def _setup_occupancy_groups(self):
    """
    Parameters: [coordinates and/or ADPs as packed by the scatterer flags]
    followed by the packed group occupancies. The per-atom occupancy
    gradients (for the atoms in the groups) are computed together with the
    other gradients and summed for each group.
    """
    from mmtbx.refinement import occupancies
    self._occ = None
    groups = self.occupancy_groups
    if(groups is None): groups = []
    if(len(groups) == 0 and self._riding is None):
      self.x = flex.double(self.xray_structure.n_parameters(), 0)
      return
    if(len(groups) > 0 and self.riding_h_manager is not None
       and self._riding is None):
      raise Sorry("Simultaneous refinement of occupancies with riding"
        " hydrogens requires riding_h=True.")
    if(len(groups) > 0 and self.neutron_refinement):
      raise Sorry("Simultaneous refinement of occupancies is not available"
        " for neutron refinement.")
    scatterers = self.xray_structure.scatterers()
    n_sc = scatterers.size()
    i_sel = flex.size_t()
    for group in groups:
      for sel in group:
        i_sel.extend(sel)
    if(i_sel.size() > 0):
      scatterers.flags_set_grad_occupancy(iselection = i_sel)
    n_full = self.xray_structure.n_parameters()
    mask = flex.double(n_full, 0)
    xray.minimization.add_gradients(
      scatterers          = scatterers,
      xray_gradients      = mask,
      occupancy_gradients = flex.double(n_sc, 1))
    occ_slots = mask > 0
    # parameters: all slots except the occupancies of the groups and the
    # slots of riding hydrogens
    other_slots = ~occ_slots
    if(self._riding is not None):
      self._riding_slot_maps(n_full)
      other_slots = other_slots & ~self._riding.dependent_slots
    table = []
    par = flex.double()
    occ = scatterers.extract_occupancies()
    sel_bools = []
    for group in groups:
      indices = []
      for sel in group:
        indices.append(par.size())
        par.append(flex.mean(occ.select(sel)))
        sel_bools.append(flex.bool(n_sc, sel))
      table.append(indices)
    n_other = other_slots.count(True)
    self._occ = group_args(
      groups      = groups,
      table       = table,
      sel_bools   = sel_bools,
      i_sel       = i_sel,
      occ_slots   = occ_slots,
      other_slots = other_slots,
      # occupancy slots are packed in the order of the atoms
      occ_atoms   = flex.bool(n_sc, i_sel).iselection(),
      n_full      = n_full,
      n_other     = n_other,
      i_other     = flex.size_t(range(n_other)))
    self.x = flex.double(n_other, 0)
    self.x.extend(occupancies.pack_unpack(x = par, table = table))

  def _setup_riding_h(self):
    """Hydrogen atoms riding on their parent atoms (riding_h=True)."""
    rhm = self.riding_h_manager
    scatterers = self.xray_structure.scatterers()
    n_sc = scatterers.size()
    h_isel, p_isel = flex.size_t(), flex.size_t()
    for rc in rhm.parameterization_cpp:
      h_isel.append(rc.ih)
      p_isel.append(rc.a0)
    hd = rhm.hd_selection
    assert hd.size() == n_sc
    is_dependent = flex.bool(n_sc, False)
    is_dependent.set_selected(h_isel, True)
    # hydrogens without riding parameterization are kept fixed
    fixed_h = (hd & ~is_dependent).iselection()
    for i_seq in fixed_h:
      scatterers[i_seq].flags.set_grads(False)
    # hydrogen ADPs are isotropic: u_iso of the hydrogen = u_iso or U_eq
    # (u_iso + U_eq of u_star, for TLS groups) of the parent
    uc = self.xray_structure.unit_cell()
    p_iso, p_aniso = flex.bool(), flex.bool()
    for i_h, i_p in zip(h_isel, p_isel):
      fh, fp = scatterers[i_h].flags, scatterers[i_p].flags
      if(fh.use_u_aniso()):
        scatterers[i_h].convert_to_isotropic(unit_cell = uc)
      fh.set_use_u_iso(True)
      fh.set_use_u_aniso(False)
      fh.set_grad_u_iso(fp.grad_u_iso() or fp.grad_u_aniso())
      fh.set_grad_u_aniso(False)
      fh.set_grad_site(fp.grad_site())
      p_iso.append(fp.use_u_iso())
      p_aniso.append(fp.use_u_aniso())
    self._riding = group_args(
      h_isel       = h_isel,
      p_isel       = p_isel,
      p_iso        = p_iso,
      p_aniso      = p_aniso,
      is_dependent = is_dependent,
      h_bool       = is_dependent,
      parameterization = rhm.parameterization_cpp)
    self._apply_riding()

  def _riding_slot_maps(self, n_full):
    """
    Positions of the site, u_iso and u_aniso parameters of each atom in the
    full (packed by the scatterer flags) parameter array, and the slots of
    the riding hydrogens (which are not parameters).
    """
    r = self._riding
    scatterers = self.xray_structure.scatterers()
    n_sc = scatterers.size()
    def slots(**kwds):
      codes = flex.double(n_full, 0)
      xray.minimization.add_gradients(scatterers = scatterers,
        xray_gradients = codes, **kwds)
      sel = (codes > 0).iselection()
      return sel, (codes.select(sel) - 1).iround().as_size_t()
    r.site_slots, r.site_codes = slots(site_gradients = flex.vec3_double(
      flex.double(range(1, 3*n_sc+1))))
    r.u_iso_slots, r.u_iso_codes = slots(u_iso_gradients =
      flex.double(range(1, n_sc+1)))
    r.u_aniso_slots, r.u_aniso_codes = slots(u_aniso_gradients =
      flex.sym_mat3_double(flex.double(range(1, 6*n_sc+1))))
    dependent = flex.bool(n_full, False)
    for sl, codes, k in [(r.site_slots, r.site_codes, 3),
                         (r.u_iso_slots, r.u_iso_codes, 1),
                         (r.u_aniso_slots, r.u_aniso_codes, 6)]:
      atoms = codes / k
      dependent.set_selected(sl.select(r.is_dependent.select(atoms)), True)
    r.dependent_slots = dependent

  def _apply_riding(self):
    """Hydrogen positions from the parent atoms, hydrogen u_iso = U_eq of
    the parent atoms."""
    r = self._riding
    xrs = self.xray_structure
    sites_cart = xrs.sites_cart()
    self.riding_h_manager.idealize_riding_h_positions(sites_cart = sites_cart)
    xrs.set_sites_cart(sites_cart)
    scatterers = xrs.scatterers()
    u_iso = scatterers.extract_u_iso()
    u_iso.set_selected(r.h_isel,
      xrs.extract_u_iso_or_u_equiv().select(r.p_isel))
    scatterers.set_u_iso(u_iso, r.h_bool, xrs.unit_cell())

  def _fold_riding(self, full, curvatures = False):
    """
    Gradients (or curvatures) with respect to the full parameter array:
    contributions of the riding hydrogens are added to the parent atoms.
    """
    from mmtbx_hydrogens_ext import modify_gradients_cpp
    r = self._riding
    n_sc = self.xray_structure.scatterers().size()
    result = full.deep_copy()
    # sites
    s = flex.double(3*n_sc, 0)
    s.set_selected(r.site_codes, full.select(r.site_slots))
    s = flex.vec3_double(s)
    if(curvatures):
      s.add_selected(r.p_isel, s.select(r.h_isel))
    else:
      s = modify_gradients_cpp(
        gradients        = s,
        sites_cart       = self.xray_structure.sites_cart(),
        parameterization = r.parameterization)
    result.set_selected(r.site_slots, s.as_double().select(r.site_codes))
    # ADPs: u_iso of the hydrogen = u_iso + U_eq of the parent (the u_aniso
    # parameters are u_cart: U_eq = trace/3)
    if(r.u_iso_slots.size() == 0): return result
    u_iso = flex.double(n_sc, 0)
    u_iso.set_selected(r.u_iso_codes, full.select(r.u_iso_slots))
    g_h = u_iso.select(r.h_isel)
    u_iso.add_selected(r.p_isel.select(r.p_iso), g_h.select(r.p_iso))
    result.set_selected(r.u_iso_slots, u_iso.select(r.u_iso_codes))
    if(r.u_aniso_slots.size() > 0):
      u_cart = flex.double(6*n_sc, 0)
      u_cart.set_selected(r.u_aniso_codes, full.select(r.u_aniso_slots))
      p_aniso = r.p_isel.select(r.p_aniso)
      g_h = g_h.select(r.p_aniso)
      d = [1/3, 1/9][int(curvatures)]
      for k in range(3):
        u_cart.add_selected(p_aniso * 6 + k, g_h * d)
      result.set_selected(r.u_aniso_slots, u_cart.select(r.u_aniso_codes))
    return result

  def _group_occupancies(self):
    from mmtbx.refinement import occupancies
    o = self._occ
    return occupancies.pack_unpack(
      x = self.x[o.n_other:], table = o.table)

  def _to_parameters(self, full, curvatures=False):
    """Per-atom packed gradients or curvatures -> parameters."""
    from mmtbx.refinement import occupancies
    o = self._occ
    if(self._riding is not None):
      full = self._fold_riding(full, curvatures = curvatures)
    per_atom = flex.double(self.xray_structure.scatterers().size(), 0)
    per_atom.set_selected(o.occ_atoms, full.select(o.occ_slots))
    group_values = flex.double([flex.sum(per_atom.select(sel_bool))
      for sel_bool in o.sel_bools])
    if(curvatures):
      packed = occupancies.pack_curvatures(x = group_values, table = o.table)
    else:
      packed = occupancies.pack_gradients(x = group_values, table = o.table)
    result = full.select(o.other_slots)
    result.extend(packed)
    return result

  def _finalize_occupancies(self):
    from mmtbx.refinement import occupancies
    o = self._occ
    occupancies.set_refinable_parameters(
      xray_structure     = self.xray_structure,
      parameters         = self._group_occupancies(),
      selections         = o.groups,
      enforce_positivity = (self.occupancy_min >= 0))
    self.xray_structure.adjust_occupancy(
      occ_max   = self.occupancy_max,
      occ_min   = self.occupancy_min,
      selection = o.i_sel)

  def riding_h_manager_is_present(self):
    return (self.model is not None and
            getattr(self.model, "riding_h_manager", None) is not None)

  def u_iso_parameter_selection(self):
    """Positions of the u_iso parameters in self.x."""
    scatterers = self.xray_structure.scatterers()
    mask = flex.double(self.xray_structure.n_parameters(), 0)
    xray.minimization.add_gradients(
      scatterers      = scatterers,
      xray_gradients  = mask,
      u_iso_gradients = flex.double(scatterers.size(), 1))
    return mask > 0

  def apply_shifts(self):
    o = self._occ
    if(o is None):
      shifts = self.x
    else:
      # per-atom shifts; the occupancies are set from the group parameters
      shifts = flex.double(o.n_full, 0)
      shifts.set_selected(o.other_slots, self.x[:o.n_other])
    if(self.refine_adp and not self.refine_xyz):
      xray.ext.truncate_shifts(
        shifts    = shifts,
        min_value = self.u_min,
        max_value = self.u_max)
    elif(self.refine_adp):
      # coordinate shifts must not be truncated
      if(getattr(self, "_u_iso_sel", None) is None):
        self._u_iso_sel = self.u_iso_parameter_selection()
      u_shifts = shifts.select(self._u_iso_sel)
      xray.ext.truncate_shifts(
        shifts    = u_shifts,
        min_value = self.u_min,
        max_value = self.u_max)
      shifts.set_selected(self._u_iso_sel, u_shifts)
    if(o is not None):
      self.x.set_selected(o.i_other, shifts.select(o.other_slots))
    if(self.riding_h_manager is not None and self.refine_xyz
       and self._riding is None):
      # temporary array with zero shifts for all atoms
      tmp = flex.vec3_double(self._scatterers_start.size(), [0,0,0])
      # Set shifts according to self.x
      tmp = tmp.set_selected(self.model.refinement_flags.sites_individual,
        flex.vec3_double(self.x))
      # Set shifts of H atoms to zero (they will be idealized later anyway)
      shifts_h = flex.vec3_double(self.riding_h_manager.hd_selection.count(True),
        [0,0,0])
      tmp = tmp.set_selected(self.riding_h_manager.hd_selection, shifts_h)
      # Select entries for shifted atoms only (e.g. if there is xyz selection)
      tmp_x = tmp.select(self.model.refinement_flags.sites_individual)
      # orig
      #tmp = tmp.set_selected(self.riding_h_manager.not_hd_selection,
      #  flex.vec3_double(self.x))
      # Set entries in self.x corresponding to H atoms to 0
      self.x.set_selected(flex.bool(self.x.size()), tmp_x.as_double())
      # orig
      #self.x.set_selected(flex.bool(self.x.size()), tmp.as_double())

    apply_shifts_result = xray.ext.minimization_apply_shifts(
      unit_cell      = self.xray_structure.unit_cell(),
      scatterers     = self._scatterers_start,
      shifts         = shifts)
    scatterers_shifted = apply_shifts_result.shifted_scatterers
    if(self.refine_xyz):
      site_symmetry_table = self.xray_structure.site_symmetry_table()
      for i_seq in site_symmetry_table.special_position_indices():
        try:
          scatterers_shifted[i_seq].site = crystal.correct_special_position(
            crystal_symmetry = self.xray_structure,
            special_op       = site_symmetry_table.get(i_seq).special_op(),
            site_frac        = scatterers_shifted[i_seq].site,
            site_label       = scatterers_shifted[i_seq].label,
            tolerance        = self.correct_special_position_tolerance)
        except Exception as e:
          print(str(e), file=self.log)
    self.xray_structure.replace_scatterers(scatterers = scatterers_shifted)
    if(o is not None):
      for value, sel_bool in zip(self._group_occupancies(), o.sel_bools):
        self.xray_structure.set_occupancies(value, sel_bool)
    if(self._riding is not None):
      self._apply_riding()
    #
    elif(self.riding_h_manager is not None and self.refine_xyz):
      sites_cart = self.xray_structure.sites_cart()
      self.riding_h_manager.idealize_riding_h_positions(
        sites_cart = sites_cart,
        selection_bool = self.model.refinement_flags.sites_individual)
      self.xray_structure.set_sites_cart(sites_cart=sites_cart)
    #
    if(self.refine_adp):
      return apply_shifts_result.u_iso_refinable_params
    else:
      return None

  def xray_curvatures(self):
    """
    Approximate curvatures of the weighted X-ray target, packed like the
    parameters. Computed once: they do not depend on the coordinates.
    """
    if(self._xray_curvatures is None):
      xrs = self.xray_structure
      tr = self.fmodels.target_functor_xray(compute_gradients = False)
      indices = self.fmodels.fmodel_xray().f_obs_work().indices()
      rp = xrs.curvatures_random_phase(
        miller_indices    = indices,
        target_curvatures = tr.curvatures_work(),
        reflection_stride = curvatures_reflection_stride(indices.size()))
      u_iso = rp.u_iso
      both = xrs.use_u_iso() & xrs.use_u_aniso()
      if(both.count(True) > 0):
        # atoms with isotropic and anisotropic contributions (e.g. TLS): the
        # u_iso curvatures from the diagonal U_cart curvatures, for
        # isotropically distributed reflections c(U_kk) = c(u_iso)/5
        u_iso = u_iso.deep_copy()
        u_iso.set_selected(both, flex.double([(5/3) * (u[0] + u[1] + u[2])
          for u in rp.u_cart.select(both)]))
      result = flex.double(xrs.n_parameters(), 0)
      xray.minimization.add_gradients(
        scatterers          = xrs.scatterers(),
        xray_gradients      = result,
        site_gradients      = rp.sites_cart,
        u_iso_gradients     = u_iso,
        u_aniso_gradients   = rp.u_cart,
        occupancy_gradients = rp.occupancy)
      self._xray_curvatures = result * (self.weights.wx*self.weights.wx_scale)
    return self._xray_curvatures

  def compute_target(self, compute_gradients, u_iso_refinable_params,
                     compute_curvatures = False):
    self.stereochemistry_residuals = None
    self.fmodels.update_xray_structure(
      xray_structure = self.xray_structure,
      update_f_calc  = True)
    fmodels_target_and_gradients = self.fmodels.target_and_gradients(
      weights                = self.weights,
      compute_gradients      = compute_gradients,
      u_iso_refinable_params = u_iso_refinable_params)
    self.f = fmodels_target_and_gradients.target()
    self.g = fmodels_target_and_gradients.gradients()
    if(compute_curvatures):
      self.curvatures = self.xray_curvatures().deep_copy()
    if(self.refine_xyz and self.restraints_manager is not None and
       self.weights.w > 0.0):
      self.stereochemistry_residuals = \
        self.model.restraints_manager_energies_sites(
          compute_gradients  = compute_gradients,
          compute_curvatures = compute_curvatures)
# QBLIB INSERT
      if(self.qblib_params is not None and self.qblib_params.qblib):
        from qbpy import qb_refinement
        self.qblib_cycle_count +=1
        if(self.tmp_XYZ is not None):
          diff,max_diff,max_elem = qb_refinement.array_difference(
            self.tmp_XYZ,
            self.model.get_sites_cart(),
            )
          if(self.XYZ_diff_curr is not None):
            self.XYZ_diff_curr=max_elem
          else:
            self.XYZ_diff_curr=max_elem
          if(max_elem>=self.qblib_params.skip_divcon_threshold):
               self.tmp_XYZ = self.model.get_sites_cart()
        else:
          self.tmp_XYZ = self.model.get_sites_cart()
        if (self.macro_cycle != self.qblib_params.macro_cycle_to_skip):
          qblib_call = qb_refinement.QBblib_call_manager(
            hierarchy = self.model.get_hierarchy(),
            xray_structure=self.model.get_xray_structure(),
            geometry_residuals = self.stereochemistry_residuals,
            qblib_params=self.qblib_params,
            diff_curr=self.XYZ_diff_curr,
            macro=self.macro_cycle,
            micro=self.qblib_cycle_count,
            )
          try:
            qblib_call.run()
          except Exception as e:
            if e.__class__.__name__=="QB_none_fatal_error":
              raise Sorry(str(e))
            else:
              raise e

          if(qblib_call.QBStatus):
            qblib_g=qblib_call.result_QBlib.gradients
            qblib_f=qblib_call.result_QBlib.target
            self.stereochemistry_residuals.gradients=qblib_g
            self.stereochemistry_residuals.target=qblib_f
# QBLIB END
      er = self.stereochemistry_residuals.target
      geometry_scale = self._normalization_scale("geometry",
        self.stereochemistry_residuals)
      er = er * geometry_scale
      self.f += er * self.weights.w
      if(compute_gradients):
        sgc = self.stereochemistry_residuals.gradients * geometry_scale
        # ias do not participate in geometry restraints
        if self.model is not None and self.model.ias_manager is not None:
          ias_selection = self.model.ias_manager.get_ias_selection()
          if ias_selection is not None and ias_selection.count(True) > 0:
            sgc.extend(flex.vec3_double(ias_selection.count(True),[0,0,0]))
        xray.minimization.add_gradients(
          scatterers     = self.xray_structure.scatterers(),
          xray_gradients = self.g,
          site_gradients = sgc*self.weights.w)
      if(compute_curvatures and
         self.stereochemistry_residuals.curvatures is not None):
        scc = self.stereochemistry_residuals.curvatures * geometry_scale
        if self.model is not None and self.model.ias_manager is not None:
          ias_selection = self.model.ias_manager.get_ias_selection()
          if ias_selection is not None and ias_selection.count(True) > 0:
            scc.extend(flex.vec3_double(ias_selection.count(True),[0,0,0]))
        xray.minimization.add_gradients(
          scatterers     = self.xray_structure.scatterers(),
          xray_gradients = self.curvatures,
          site_gradients = scc*self.weights.w)
    if(self.refine_adp and self.restraints_manager is not None and
       self.restraints_manager.geometry is not None
       and self.adp_restraints_weight > 0.0
       and self.iso_restraints is not None):
      use_hd = False
      if(self.fmodels.fmodel_n is not None or
         (self.is_neutron_scat_table is not None and
          self.is_neutron_scat_table == "neutron") or
         self.h_params.refine == "individual"):
        use_hd = True
      energies_adp = self.model.energies_adp(
        iso_restraints     = self.iso_restraints,
        use_hd             = use_hd,
        compute_gradients  = compute_gradients,
        compute_curvatures = compute_curvatures)
      adp_scale = self._normalization_scale("adp", energies_adp)
      self.f += energies_adp.target * self.adp_restraints_weight * adp_scale
      if(compute_curvatures):
        w_a = self.adp_restraints_weight * adp_scale
        u_iso_curvatures = energies_adp.u_iso_curvatures
        if(u_iso_curvatures is not None):
          u_iso_curvatures = u_iso_curvatures * w_a
        u_aniso_curvatures = energies_adp.u_aniso_curvatures
        if(u_aniso_curvatures is not None):
          u_aniso_curvatures = u_aniso_curvatures.deep_copy()
          u_aniso_curvatures *= w_a
        xray.minimization.add_gradients(
          scatterers        = self.xray_structure.scatterers(),
          xray_gradients    = self.curvatures,
          u_iso_gradients   = u_iso_curvatures,
          u_aniso_gradients = u_aniso_curvatures)
      if(compute_gradients):
        w_a = self.adp_restraints_weight * adp_scale
        if(energies_adp.u_aniso_gradients is None):
          xray.minimization.add_gradients(
            scatterers      = self.xray_structure.scatterers(),
            xray_gradients  = self.g,
            u_iso_gradients = energies_adp.u_iso_gradients * w_a)
        else:
          energies_adp.u_aniso_gradients *= w_a
          if(energies_adp.u_iso_gradients is not None):
            energies_adp.u_iso_gradients *= w_a
          xray.minimization.add_gradients(
            scatterers        = self.xray_structure.scatterers(),
            xray_gradients    = self.g,
            u_aniso_gradients = energies_adp.u_aniso_gradients,
            u_iso_gradients   = energies_adp.u_iso_gradients)
          energies_adp.u_aniso_gradients = None # just for safety
          energies_adp.u_iso_gradients = None
    if(self._occ is not None):
      if(compute_gradients):
        self.g = self._to_parameters(self.g)
      if(compute_curvatures):
        self.curvatures = self._to_parameters(self.curvatures,
          curvatures = True)

  def _normalization_scale(self, key, energies):
    """
    Factor that replaces the normalization of energies (target divided by
    the current number of restraints) by the normalization at the first
    evaluation (smooth_target=True), otherwise 1.
    """
    if(not self.smooth_target): return 1.0
    factor = getattr(energies, "normalization_factor", None)
    if(factor is None or factor == 0): return 1.0
    fixed = self._fixed_normalization.setdefault(key, factor)
    return fixed / factor

  def callback_after_step(self, minimizer):
    if (self.verbose > 0):
      print("refinement.minimization step: f,iter,nfun:", end=' ')
      print(self.f,minimizer.iter(),minimizer.nfun())

  def compute_functional_and_gradients(self, compute_curvatures = False):
    u_iso_refinable_params = self.apply_shifts()
    self.compute_target(compute_gradients     = True,
                        u_iso_refinable_params = u_iso_refinable_params,
                        compute_curvatures     = compute_curvatures)
    if (self.verbose > 1):
      print("xray.minimization line search: f,rms(g):", end=' ')
      print(self.f, math.sqrt(flex.mean_sq(self.g)))
    return self.f, self.g

  def compute_functional_gradients_diag(self):
    """
    Called by scitbx.lbfgs.run() if self.diag_mode is not None.
    Returns the diagonal of the inverse Hessian approximation Hk0.
    """
    def compute():
      f, g = self.compute_functional_and_gradients(compute_curvatures = True)
      return f, g, self.curvatures
    return self.curvature_preconditioner(self.x, compute)

class monitor(object):
  def __init__(self, weights,
                     fmodels,
                     model,
                     iso_restraints,
                     refine_xyz = False,
                     refine_adp = False,
                     refine_occ = False):
    adopt_init_args(self, locals())
    self.ex = []
    self.en = []
    self.er = []
    self.et = []
    self.rxw = []
    self.rxf = []
    self.rnw = []
    self.rnf = []
    self.iter = None
    self.nfun = None

  def collect(self, iter = None, nfun = None):
    if(iter is not None): self.iter = format_value("%-4d", iter)
    if(nfun is not None): self.nfun = format_value("%-4d", nfun)
    self.fmodels.fmodel_xray().xray_structure == self.model.get_xray_structure()
    fmodels_tg = self.fmodels.target_and_gradients(
      weights           = self.weights,
      compute_gradients = False)
    self.rxw.append(format_value("%6.4f", self.fmodels.fmodel_xray().r_work()))
    self.rxf.append(format_value("%6.4f", self.fmodels.fmodel_xray().r_free()))
    self.ex.append(format_value("%10.4f", fmodels_tg.target_work_xray))
    if(self.fmodels.fmodel_neutron() is not None):
      self.rnw.append(format_value("%6.4f", self.fmodels.fmodel_neutron().r_work()))
      self.rnf.append(format_value("%6.4f", self.fmodels.fmodel_neutron().r_free()))
      self.en.append(format_value("%10.4f", fmodels_tg.target_work_neutron))
    if(self.refine_xyz and self.weights.w > 0):
      er = self.model.restraints_manager_energies_sites(
        compute_gradients = False).target
    elif(self.refine_adp and self.weights.w > 0):
      use_hd = False
      if(self.fmodels.fmodel_n is not None): use_hd = True
      er = self.model.energies_adp(
        iso_restraints = self.iso_restraints,
        use_hd = use_hd,
        compute_gradients = False).target
    elif(self.refine_occ):
      er = 0
    else: er = 0
    self.er.append(format_value("%10.4f", er))
    self.et.append(format_value(
      "%10.4f", fmodels_tg.target()+er*self.weights.w))

  def show(self, message = "", log = None):
    if(log is None): log = sys.stdout
    print("|-"+message+"-"*(79-len("| "+message+"|"))+"|", file=log)
    if(self.fmodels.fmodel_neutron() is not None):
      print("|"+" "*33+"x-ray data:"+" "*33+"|", file=log)
    print("| start r-factor (work) = %s      final r-factor "\
     "(work) = %s          |"%(self.rxw[0], self.rxw[1]), file=log)
    print("| start r-factor (free) = %s      final r-factor "\
     "(free) = %s          |"%(self.rxf[0], self.rxf[1]), file=log)
    if(self.fmodels.fmodel_neutron() is not None):
      print("|"+" "*32+"neutron data:"+" "*32+"|", file=log)
      print("| start r-factor (work) = %s      final r-factor "\
      "(work) = %s          |"%(self.rnw[0], self.rnw[1]), file=log)
      print("| start r-factor (free) = %s      final r-factor "\
      "(free) = %s          |"%(self.rnf[0], self.rnf[1]), file=log)
    print("|"+"-"*77+"|", file=log)
    #
    if(self.fmodels.fmodel_neutron() is None):
      for i_seq, stage in enumerate(["start", "final"]):
        if(self.refine_xyz):
          print("| T_%s = wxc * wxc_scale * Exray + wc * Echem"%stage+" "*30+"|", file=log)
          print(self.xray_line(i_seq), file=log)
        elif(self.refine_adp):
          print("| T_%s = wxu * wxu_scale * Exray + wu * Eadp"%stage+" "*31+"|", file=log)
          print(self.xray_line(i_seq), file=log)
        elif(self.refine_occ):
          print("| T_%s = 1.0 * 1.0 * Exray"%stage+" "*43+"|", file=log)
          print(self.xray_line(i_seq), file=log)
        if(i_seq == 0): print("|"+" "*77+"|", file=log)
    #
    if(self.fmodels.fmodel_neutron() is not None):
      for i_seq, stage in enumerate(["start", "final"]):
        if(self.refine_xyz):
          print("| T_%s = wnxc * (wxc_scale * Exray + wnc_scale * Eneutron) + wc * Echem"%stage+" "*4+"|", file=log)
          print(self.neutron_line(i_seq), file=log)
        elif(self.refine_adp):
          print("| T_%s = wnxu * (wxu_scale * Exray + wnu_scale * Eneutron) + wu * Eadp"%stage+" "*5+"|", file=log)
          print(self.neutron_line(i_seq), file=log)
        elif(self.refine_occ):
          print("| T_%s = 1.0 * (1.0 * Exray + 1.0 * Eneutron)"%stage+" "*30+"|", file=log)
          print(self.neutron_line(i_seq), file=log)
        if(i_seq == 0): print("|"+" "*77+"|", file=log)
    #
    print("|"+"-"*77+"|", file=log)
    print("| number of iterations = %s    |    number of function "\
                  "evaluations = %s   |"%(self.iter, self.nfun), file=log)
    print("|"+"-"*77+"|", file=log)
    log.flush()

  def neutron_line(self, i_seq):
    line = "| %s = %s * (%s * %s + %s * %s) + %s * %s"%(
      self.et[i_seq].strip(),
      format_value("%6.2f",self.weights.wxn).strip(),
      format_value("%6.2f",self.weights.wx_scale).strip(),
      self.ex[i_seq].strip(),
      format_value("%6.2f",self.weights.wn_scale).strip(),
      self.en[i_seq].strip(),
      format_value("%6.2f",self.weights.w).strip(),
      self.er[i_seq].strip())
    line = line + " "*(78 - len(line))+"|"
    return line

  def xray_line(self, i_seq):
    line = "| %s = %s * %s * %s + %s * %s"%(
      self.et[i_seq].strip(),
      format_value("%6.2f",self.weights.wx).strip(),
      format_value("%6.2f",self.weights.wx_scale).strip(),
      self.ex[i_seq].strip(),
      format_value("%6.2f",self.weights.w).strip(),
      self.er[i_seq].strip())
    line = line + " "*(78 - len(line))+"|"
    return line

class run_constrained(object):
  def __init__(self,
               model,
               fmodel,
               target_weight,
               log,
               params,
               prefix,
               refine_sites           = False,
               refine_u_iso           = False,
               refine_transformations = False):
    #
    # XXX Same needs to be done for refine_u_iso
    #
    refine_selection = None
    if(refine_sites):
      refine_selection = model.refinement_flags.sites_individual.iselection()
    #
    tg_object = mmtbx.refinement.minimization_ncs_constraints.\
      target_function_and_grads_reciprocal_space(
        fmodel                    = fmodel,
        ncs_restraints_group_list = model.get_ncs_groups(),
        refine_selection          = refine_selection, # XXX
        restraints_manager        = model.restraints_manager,
        iso_restraints            = params.adp_restraints.iso,
        data_weight               = target_weight,
        refine_sites              = refine_sites,
        refine_u_iso              = refine_u_iso,
        refine_transformations    = refine_transformations)
    self.minimized = mmtbx.refinement.minimization_ncs_constraints.lbfgs(
       target_and_grads_object      = tg_object,
       xray_structure               = fmodel.xray_structure,
       ncs_restraints_group_list    = model.get_ncs_groups(),
       refine_selection             = refine_selection, # XXX
       finite_grad_differences_test = False,
       max_iterations               = params.main.max_number_of_iterations,
       refine_sites                 = refine_sites,
        refine_u_iso                = refine_u_iso,
        refine_transformations      = refine_transformations)
    fmodel.update_xray_structure(update_f_calc=True)
    model.set_xray_structure(fmodel.xray_structure)
