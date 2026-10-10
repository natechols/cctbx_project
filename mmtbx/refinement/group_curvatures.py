"""
Approximate curvatures (diagonal of the 2nd derivatives) of the X-ray target
with respect to group parameters (group ADP, group occupancy, rigid-body
translation and rotation), for preconditioning of the LBFGS minimizer.

The per-atom random-phase curvatures (cctbx.xray.structure.
curvatures_random_phase) are summed over the atoms of a group with the chain
rule. This neglects the correlation between the contributions of the atoms
of the group: a group parameter changes the structure factor contribution
F_g of the whole group coherently. For the group contribution in random
phase relation to the rest of the structure (F = F_g + F_rest), the
expected Gauss-Newton curvature of a parameter with dF_g/dp = z*F_g is

  acentric: 1/2 |z|^2 <|F_g|^2> (1 + f_g)   z real      (B, occupancy)
            1/2 |z|^2 <|F_g|^2> (1 - f_g)   z imaginary (translation)
  centric:      |z|^2 <|F_g|^2>             (no coherent part)

where f_g = <|F_g|^2>/<|F|^2> is the fraction of the scattering of the
group. The per-atom sums correspond to 1/2 |z|^2 <|F_g|^2> (acentric), so
the coherence factors are

  B, occupancy: 1 + f_g * a   (a: acentric share of the curvature weight)
  translation:  1 - f_g * P_kk

where P = (1/n) sum_s R_s (Cartesian) is the projector onto the polar
directions of the point group (only translations along polar directions
change the phases of all symmetry copies of the group equally; P = 0 for
groups without polar directions, e.g. P212121 or P-1). Rotations about the
center of the group are not coherent with F_g (the per-atom sum is used).

For parameters that change the ADPs of the atoms of a group by different
tensors D_i = dU_i/dp (e.g. TLS), the per-atom curvatures for isotropically
distributed reflections are kappa_i ((tr D_i)^2 + 2 D_i:D_i)/15 (kappa_i:
curvature for an isotropic shift of U_i), and the coherent part is reduced
by rho = q(<D_mean>)/mean(q(D_i)), q(D) = (tr D)^2 + 2 D:D, with means
weighted by the scattering power, and <D> = (1/n) sum_s R_s D R_s^T the
average over the point group (the ADP changes of the symmetry copies of
the group are R_s D R_s^T; rho = 1 for D_i = I, i.e. group B).
"""
from __future__ import absolute_import, division, print_function
from cctbx.array_family import flex
from scitbx import matrix

class group_curvatures(object):
  """
  Per-atom random-phase curvatures and coherence corrections for group
  parameters; computed once (the curvatures do not depend strongly on the
  parameters).
  """

  def __init__(self, fmodel, reflection_stride=None):
    from mmtbx.refinement.minimization import curvatures_reflection_stride
    xrs = fmodel.xray_structure
    self.xray_structure = xrs
    tr = fmodel.target_functor()(compute_gradients=False)
    target_curvatures = tr.curvatures_work()
    f_obs_work = fmodel.f_obs_work()
    indices = f_obs_work.indices()
    if (reflection_stride is None):
      reflection_stride = curvatures_reflection_stride(indices.size())
    self.atoms = xrs.curvatures_random_phase(
      miller_indices    = indices,
      target_curvatures = target_curvatures,
      reflection_stride = reflection_stride)
    # acentric share of the curvature weights for occupancy and B
    centric = f_obs_work.centric_flags().data()
    kappa = flex.double(indices.size(), 0.5)
    kappa.set_selected(centric, 1.0)
    w = target_curvatures * f_obs_work.epsilons().data().as_double() * kappa
    def acentric_share(w):
      s = flex.sum(w)
      if (s <= 0): return 1.0
      return flex.sum(w.select(~centric)) / s
    self.acentric_share_occupancy = acentric_share(w)
    self.acentric_share_b = acentric_share(
      w * flex.pow2(f_obs_work.d_star_sq().data()))
    # projector onto the polar directions of the point group
    o = matrix.sqr(xrs.unit_cell().orthogonalization_matrix())
    f = o.inverse()
    ops = xrs.space_group().all_ops()
    p = matrix.sqr([0]*9)
    self.rotations_cart = []
    for op in ops:
      r = o * matrix.sqr(op.r().as_double()) * f
      self.rotations_cart.append(r)
      p += r
    self.polar_projector = p / len(ops)
    self.rotations_cart_flex = flex.mat3_double(
      [r.elems for r in self.rotations_cart])
    # scattering power of the atoms (f(0) occupancy)^2
    reg = xrs.scattering_type_registry()
    f0 = flex.double([g.at_d_star_sq(0)
      for g in reg.unique_gaussians_as_list()])
    self.scattering_power = flex.pow2(
      f0.select(reg.unique_indices(xrs.scatterers()))
      * xrs.scatterers().extract_occupancies())
    self.total_scattering_power = flex.sum(self.scattering_power)
    self._u_iso = None

  def group_fraction(self, iselection):
    if (self.total_scattering_power <= 0): return 0
    return min(1., flex.sum(self.scattering_power.select(iselection))
      / self.total_scattering_power)

  def u_iso(self):
    """
    Per-atom curvatures with respect to an isotropic shift of the ADP.
    For anisotropic atoms they are estimated from the diagonal U_cart
    curvatures: for isotropically distributed reflections
    c(U_kk) = c(u_iso)/5.
    """
    if (self._u_iso is None):
      result = self.atoms.u_iso.deep_copy()
      aniso = self.xray_structure.use_u_aniso()
      n_aniso = aniso.count(True)
      if (n_aniso > 0):
        u = self.atoms.u_cart.select(aniso).as_double()
        i = flex.size_t_range(0, 6*n_aniso, 6)
        result.set_selected(aniso,
          (5/3) * (u.select(i) + u.select(i+1) + u.select(i+2)))
      self._u_iso = result
    return self._u_iso.deep_copy()

  def group_u_iso(self, selections):
    c = self.u_iso()
    return flex.double([flex.sum(c.select(sel)) *
      (1 + self.group_fraction(sel) * self.acentric_share_b)
      for sel in selections])

  def group_u_cart(self, iselection, d_u_cart):
    """
    Curvature for a parameter that changes the U_cart of the atoms of the
    group (iselection) by d_u_cart (flex.sym_mat3_double, one tensor per
    selected atom).
    """
    assert d_u_cart.size() == iselection.size()
    def q(d):
      tr = d[0] + d[1] + d[2]
      dd = d[0]**2 + d[1]**2 + d[2]**2 + 2*(d[3]**2 + d[4]**2 + d[5]**2)
      return tr*tr + 2*dd
    kappa = self.u_iso().select(iselection)
    w = self.scattering_power.select(iselection)
    q_i = flex.double([q(d) for d in d_u_cart])
    incoherent = flex.sum(kappa * q_i) / 15
    w_sum = flex.sum(w)
    if (w_sum <= 0 or incoherent <= 0): return incoherent
    d_mean = [0]*6
    for d, wi in zip(d_u_cart, w):
      for k in range(6): d_mean[k] += wi * d[k] / w_sum
    # average over the point group
    dm = matrix.sym(sym_mat3=d_mean)
    d_sym = matrix.sqr([0]*9)
    for r in self.rotations_cart:
      d_sym += r * dm * r.transpose()
    d_sym = d_sym / len(self.rotations_cart)
    d_mean = (d_sym[0], d_sym[4], d_sym[8], d_sym[1], d_sym[2], d_sym[5])
    q_mean = flex.sum(w * q_i) / w_sum
    rho = q(d_mean) / q_mean if q_mean > 0 else 0
    return incoherent * (1 + self.group_fraction(iselection)
      * self.acentric_share_b * rho)

  def tls_group(self, iselection, origin):
    """
    Curvatures for the 21 TLS parameters (T, L, S) of a group, as
    group_u_cart() for the U_cart changes dU/dp of each parameter (with
    trace(S) = 0, as mmtbx.tls.tools.uaniso_from_tls_one_group with
    zeroize_trace=True); computed in C++ (mmtbx_tls_ext).
    """
    from mmtbx_tls_ext import tls_group_curvatures
    return tls_group_curvatures(
      sites_cart      = self.xray_structure.sites_cart().select(iselection),
      origin          = origin,
      kappa           = self.u_iso().select(iselection),
      weights         = self.scattering_power.select(iselection),
      rotations_cart  = self.rotations_cart_flex,
      coherent_factor = self.group_fraction(iselection)
                        * self.acentric_share_b)

  def group_occupancy(self, selections):
    c = self.atoms.occupancy
    return flex.double([flex.sum(c.select(sel)) *
      (1 + self.group_fraction(sel) * self.acentric_share_occupancy)
      for sel in selections])

  def group_translation(self, iselection):
    """Curvatures for Cartesian translations of the group."""
    s = self.atoms.sites_cart.select(iselection)
    s = [flex.sum(x) for x in s.parts()]
    f_g = self.group_fraction(iselection)
    p = self.polar_projector
    return [s[k] * (1 - f_g * p(k, k)) for k in range(3)]

  def group_rotation(self, iselection, center, axes_derivatives=None):
    """
    Curvatures for rotations of the group about center. With
    axes_derivatives=None, rotations about the Cartesian axes (in
    radians); otherwise axes_derivatives(r) returns, for a site at r
    (relative to center), the derivatives of the site with respect to
    each rotation parameter (list of 3-vectors).
    """
    c_sites = self.atoms.sites_cart.select(iselection)
    sites = self.xray_structure.sites_cart().select(iselection)
    center = matrix.col(center)
    if (axes_derivatives is None):
      axes = [matrix.col(v) for v in [(1,0,0), (0,1,0), (0,0,1)]]
      axes_derivatives = lambda r: [a.cross(r) for a in axes]
    result = None
    for site, c in zip(sites, c_sites):
      ds = axes_derivatives(matrix.col(site) - center)
      values = [sum(c[k] * d[k]**2 for k in range(3)) for d in ds]
      if (result is None): result = values
      else: result = [a + b for a, b in zip(result, values)]
    return result
