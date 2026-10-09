from __future__ import absolute_import, division, print_function
from cctbx import crystal
from cctbx.array_family import flex
import scitbx.lbfgs
from scitbx.lbfgs.curvature_preconditioner import curvature_preconditioner

class empty: pass

class lbfgs(object):

  def __init__(self,
      sites_cart,
      geometry_restraints_manager,
      riding_h_manager=None,
      correct_special_position_tolerance=1.0,
      geometry_restraints_flags=None,
      lbfgs_termination_params=None,
      lbfgs_core_params=None,
      lbfgs_exception_handling_params=None,
      disable_asu_cache=False,
      sites_cart_selection=None,
      site_labels=None,
      states_collector=None,
      use_curvatures=False,
      curvature_floor_fraction=0.1):
    """
    If use_curvatures is True, the LBFGS minimizer is preconditioned with
    the inverse of the approximate diagonal of the 2nd derivative matrix
    (see energies(compute_curvatures=True)). Coordinates without curvature
    information (e.g. atoms restrained only by terms that do not compute
    curvatures yet) are assigned curvature_floor_fraction times the mean
    of the positive curvatures. See scitbx.lbfgs.curvature_preconditioner.
    """
    if (lbfgs_termination_params is None):
      lbfgs_termination_params = scitbx.lbfgs.termination_parameters(
        max_iterations=1000)
    self.riding_h_manager = riding_h_manager
    self.correct_special_position_tolerance=correct_special_position_tolerance
    self.site_labels = site_labels
    self.states_collector = states_collector
    self.tmp = empty()
    self.rmsd_bonds, self.rmsd_angles = None, None
    self.use_curvatures = use_curvatures
    if (use_curvatures):
      self.curvature_preconditioner = curvature_preconditioner(
        floor_fraction=curvature_floor_fraction)
      assert riding_h_manager is None # not implemented
      self.diag_mode = "always"
    if sites_cart_selection:
      self.sites_cart_selection = flex.bool(sites_cart_selection)
      self.tmp.reduced_sites_cart=sites_cart.select(self.sites_cart_selection)
      self.x = flex.double(self.tmp.reduced_sites_cart.size()*3, 0)
    else:
      self.sites_cart_selection = None
      if(self.riding_h_manager is not None):
        self.x = self.riding_h_manager.refinable_parameters_init()
      else:
        self.x = flex.double(sites_cart.size()*3, 0)
    self.tmp.geometry_restraints_manager = geometry_restraints_manager
    self.tmp.geometry_restraints_flags = geometry_restraints_flags
    self.tmp.disable_asu_cache = disable_asu_cache
    self.tmp.sites_cart = sites_cart
    self.tmp.sites_shifted = sites_cart
    self.first_target_result = None
    self.minimizer = scitbx.lbfgs.run(
      target_evaluator=self,
      termination_params=lbfgs_termination_params,
      core_params=lbfgs_core_params,
      exception_handling_params=lbfgs_exception_handling_params)
    self.apply_shifts()
    self.compute_target(compute_gradients=True)
    self.final_target_result = self.tmp.target_result
    sites_cart.clear()
    sites_cart.extend(self.tmp.sites_shifted)
    del self.tmp
    del self.x
    self.first_target_value = self.first_target_result.target
    self.final_target_value = self.final_target_result.target

  def apply_shifts(self):
    if self.sites_cart_selection:
      shifted = self.tmp.reduced_sites_cart + flex.vec3_double(self.x)
      self.tmp.sites_shifted = self.tmp.sites_cart.deep_copy()
      self.tmp.sites_shifted.set_selected(self.sites_cart_selection, shifted)
    else:
      if(self.riding_h_manager is None):
        self.tmp.sites_shifted = self.tmp.sites_cart + flex.vec3_double(self.x)
      else:
        new_sites = self.tmp.sites_cart.select(
          self.riding_h_manager.not_hd_selection) + flex.vec3_double(self.x)
        self.tmp.sites_shifted.set_selected(
          self.riding_h_manager.not_hd_selection, new_sites)
        self.riding_h_manager.idealize_riding_h_positions(sites_cart = self.tmp.sites_shifted)
    if(self.states_collector is not None):
      self.states_collector.add(sites_cart = self.tmp.sites_shifted)
    if (self.tmp.geometry_restraints_manager.crystal_symmetry is not None):
      crystal_symmetry = self.tmp.geometry_restraints_manager.crystal_symmetry
      site_symmetry_table \
        = self.tmp.geometry_restraints_manager.site_symmetry_table
      assert site_symmetry_table is not None
      for i_seq in site_symmetry_table.special_position_indices():
        self.tmp.sites_shifted[i_seq] = crystal.correct_special_position(
          crystal_symmetry=crystal_symmetry,
          tolerance=self.correct_special_position_tolerance,
          special_op=site_symmetry_table.get(i_seq).special_op(),
          site_cart=self.tmp.sites_shifted[i_seq])

  def compute_target(self, compute_gradients, compute_curvatures=False):
    kwds = {}
    if (compute_curvatures):
      kwds["compute_curvatures"] = True
    self.tmp.target_result = \
      self.tmp.geometry_restraints_manager.energies_sites(
        sites_cart=self.tmp.sites_shifted,
        flags=self.tmp.geometry_restraints_flags,
        compute_gradients=compute_gradients,
        disable_asu_cache=self.tmp.disable_asu_cache,
        site_labels=self.site_labels,
        **kwds)

  def compute_functional_and_gradients(self, compute_curvatures=False):
    if (self.first_target_result is None):
      assert self.x.all_eq(0)
    else:
      self.apply_shifts()
    self.compute_target(
      compute_gradients=True,
      compute_curvatures=compute_curvatures)
    self.f = self.tmp.target_result.target
    if (self.first_target_result is None):
      self.first_target_result = self.tmp.target_result
    if self.sites_cart_selection:
      ptr = self.tmp.target_result.gradients
      self.g = ptr.select(self.sites_cart_selection).as_double()
    else:
      if(self.riding_h_manager is None):
        self.g = self.tmp.target_result.gradients.as_double()
      else:
        self.g = self.riding_h_manager.gradients_reduced_cpp(
          gradients    = self.tmp.target_result.gradients,
          sites_cart   = self.tmp.sites_shifted,
          hd_selection = self.riding_h_manager.hd_selection)
    return self.f, self.g

  def compute_functional_gradients_diag(self):
    """
    Called by scitbx.lbfgs.run() if self.diag_mode is not None.
    Returns the diagonal of the inverse Hessian approximation Hk0.
    """
    def compute():
      f, g = self.compute_functional_and_gradients(compute_curvatures=True)
      c = self.tmp.target_result.curvatures
      if self.sites_cart_selection:
        c = c.select(self.sites_cart_selection)
      return f, g, c.as_double()
    return self.curvature_preconditioner(self.x, compute)
