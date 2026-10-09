"""
Tests for the Gauss-Newton curvatures (approximate diagonal of the 2nd
derivative matrix) of bond, angle and nonbonded restraints, which can be
used to precondition the LBFGS minimizer.

The Gauss-Newton curvature of a restraint with residual R(delta) is
  max(0, d2R/d(delta)^2) * (d(delta)/d(x))^2
For a residual sum the curvature of each coordinate is the sum over all
restraints. All derivatives are checked here with finite differences.
"""
from __future__ import absolute_import, division, print_function
from cctbx import geometry_restraints
from cctbx import crystal
from cctbx import sgtbx
from cctbx import xray
from cctbx.crystal import direct_space_asu
from cctbx.array_family import flex
from libtbx.test_utils import approx_equal
import random
from six.moves import range

def finite_difference_d1(f, x, eps=1.e-6):
  """d(f)/d(x) for each coordinate of a list of sites."""
  result = []
  for i_site in range(len(x)):
    gs = []
    for i_coor in range(3):
      xs = [list(site) for site in x]
      xs[i_site][i_coor] += eps
      fp = f(xs)
      xs[i_site][i_coor] -= 2*eps
      fm = f(xs)
      gs.append((fp-fm)/(2*eps))
    result.append(gs)
  return result

def finite_difference_d2_along(f, eps=1.e-4):
  """Second derivative of f(t) at t=0."""
  return (f(eps) - 2*f(0) + f(-eps)) / eps**2

def gauss_newton(d2r, grad_delta):
  d2r = max(0, d2r)
  return [[d2r*g**2 for g in gs] for gs in grad_delta]

def random_sites(n):
  return [[random.uniform(-2,2) for i in range(3)] for j in range(n)]

def exercise_bond():
  for slack, limit, top_out in [(0,-1,False), (0.1,-1,False), (0,0.5,True)]:
    for i_trial in range(10):
      sites = random_sites(2)
      distance_ideal = random.uniform(1,5)
      def make(sites):
        return geometry_restraints.bond(
          sites=sites, distance_ideal=distance_ideal, weight=12,
          slack=slack, limit=limit, top_out=top_out)
      b = make(sites)
      grad_delta = finite_difference_d1(lambda xs: make(xs).distance_model,
        sites)
      # move sites[1] along the bond to change delta linearly
      u = [(sites[1][i]-sites[0][i])/b.distance_model for i in range(3)]
      def residual_along(t):
        return make([sites[0], [sites[1][i]+t*u[i] for i in range(3)]]) \
          .residual()
      d2r = finite_difference_d2_along(residual_along)
      if (abs(abs(b.delta) - slack) < 1.e-3): continue # kink at slack
      expected = gauss_newton(d2r, grad_delta)
      assert approx_equal(b.curvatures(), expected, eps=1.e-4)
  # inside the slack region the curvatures are zero
  b = geometry_restraints.bond(
    sites=[(0,0,0),(1.05,0,0)], distance_ideal=1, weight=1, slack=0.1)
  assert approx_equal(b.curvatures(), [(0,0,0),(0,0,0)])
  # harmonic potential: curvatures are 2*weight*u**2
  b = geometry_restraints.bond(
    sites=[(0,0,0),(3,4,0)], distance_ideal=2, weight=3)
  assert approx_equal(b.curvatures(),
    [(2*3*0.36,2*3*0.64,0), (2*3*0.36,2*3*0.64,0)])

def exercise_angle():
  for slack in [0, 2]:
    for i_trial in range(10):
      sites = random_sites(3)
      angle_ideal = random.uniform(60,170)
      def make(sites):
        return geometry_restraints.angle(
          sites=sites, angle_ideal=angle_ideal, weight=3, slack=slack)
      a = make(sites)
      if (abs(abs(a.delta) - slack) < 1.e-3): continue
      grad_delta = finite_difference_d1(lambda xs: make(xs).angle_model,
        sites)
      d2r = 0
      if (abs(a.delta) > slack): d2r = 2 * a.weight
      expected = gauss_newton(d2r, grad_delta)
      assert approx_equal(a.curvatures(), expected, eps=1.e-4)
      # the exact curvatures differ from the Gauss-Newton approximation
      # by -2 * weight * delta_slack * d2(angle)/d(x)**2
      exact = a.grads_and_curvs()[3:]
      for i_site in range(3):
        for i_coor in range(3):
          def angle_along(t):
            xs = [list(site) for site in sites]
            xs[i_site][i_coor] += t
            return make(xs).angle_model
          d2_angle = finite_difference_d2_along(angle_along)
          value = exact[i_site][i_coor]
          assert approx_equal(
            value,
            a.curvatures()[i_site][i_coor]
              - 2 * a.weight * a.delta_slack * d2_angle,
            eps=1.e-4*max(1,abs(value)))

def nonbonded_functions():
  return [
    (geometry_restraints.nonbonded_prolsq,
     geometry_restraints.prolsq_repulsion_function()),
    (geometry_restraints.nonbonded_prolsq,
     geometry_restraints.prolsq_repulsion_function(
       c_rep=2, k_rep=0.9, irexp=2, rexp=3)),
    (geometry_restraints.nonbonded_inverse_power,
     geometry_restraints.inverse_power_repulsion_function(
       nonbonded_distance_cutoff=10, k_rep=2, irexp=2)),
    (geometry_restraints.nonbonded_cos,
     geometry_restraints.cos_repulsion_function(max_residual=13)),
    (geometry_restraints.nonbonded_cos,
     geometry_restraints.cos_repulsion_function(
       max_residual=13, exponent=2)),
    (geometry_restraints.nonbonded_cos,
     geometry_restraints.cos_repulsion_function(
       max_residual=13, exponent=2.5)),
    (geometry_restraints.nonbonded_gaussian,
     geometry_restraints.gaussian_repulsion_function(
       max_residual=13, norm_height_at_vdw_distance=0.2))]

def exercise_nonbonded():
  for nonbonded_type, function in nonbonded_functions():
    n_nonzero = 0
    for i_trial in range(20):
      sites = random_sites(2)
      vdw_distance = random.uniform(1,4)
      def make(sites):
        return nonbonded_type(
          sites=sites, vdw_distance=vdw_distance, function=function)
      r = make(sites)
      grad_delta = finite_difference_d1(lambda xs: make(xs).delta, sites)
      u = [(sites[1][i]-sites[0][i])/r.delta for i in range(3)]
      def residual_along(t):
        return make([sites[0], [sites[1][i]+t*u[i] for i in range(3)]]) \
          .residual()
      if (abs(r.delta - vdw_distance) < 1.e-3): continue
      d2r = finite_difference_d2_along(residual_along)
      expected = gauss_newton(d2r, grad_delta)
      assert approx_equal(r.curvatures(), expected,
        eps=1.e-5*max(1,abs(d2r)))
      if (flex.vec3_double(r.curvatures()).as_double().all_ne(0)):
        n_nonzero += 1
    assert n_nonzero > 0

def gauss_newton_curvatures_by_finite_differences(
      sites_cart, deltas, residuals_and_weights, eps=1.e-6):
  """
  Brute-force sum over all restraints of d2R/d(delta)^2*(d(delta)/d(x))^2,
  where residuals_and_weights(i_proxy, delta_shift) returns the residual of
  a restraint (multiplied by its weight in the residual sum) given a shift
  of its delta.
  """
  delta_0 = deltas(sites_cart)
  n_proxies = delta_0.size()
  d2r = flex.double()
  for i_proxy in range(n_proxies):
    d2r.append(max(0, finite_difference_d2_along(
      lambda t: residuals_and_weights(i_proxy, t))))
  result = flex.double(sites_cart.size()*3, 0)
  for i_seq in range(sites_cart.size()):
    for i_coor in range(3):
      sp = sites_cart.deep_copy()
      site = list(sp[i_seq]); site[i_coor] += eps; sp[i_seq] = site
      sm = sites_cart.deep_copy()
      site = list(sm[i_seq]); site[i_coor] -= eps; sm[i_seq] = site
      grad_delta = (deltas(sp) - deltas(sm)) / (2*eps)
      result[i_seq*3+i_coor] = flex.sum(d2r * grad_delta**2)
  return result

def symmetric_structure():
  random.seed(0)
  xs = xray.structure(crystal_symmetry=crystal.symmetry(
    unit_cell=(6,7,8,90,100,90), space_group_symbol="P 1 21 1"))
  for i in range(8):
    xs.add_scatterer(xray.scatterer(label="C%d" % i,
      site=[random.random() for j in range(3)]))
  return xs

def asu_pair_proxies(xray_structure, distance_cutoff, make_proxy,
      sorted_asu_proxies_type):
  asu_mappings = xray_structure.asu_mappings(buffer_thickness=distance_cutoff)
  pair_generator = crystal.neighbors_fast_pair_generator(
    asu_mappings=asu_mappings, distance_cutoff=distance_cutoff,
    minimal=False)
  result = sorted_asu_proxies_type(asu_mappings=asu_mappings)
  for pair in pair_generator:
    result.process(make_proxy(pair))
  return result

def exercise_bond_residual_sum():
  xs = symmetric_structure()
  sites_cart = xs.sites_cart()
  proxies = asu_pair_proxies(xs, 4.0,
    lambda pair: geometry_restraints.bond_asu_proxy(
      pair=pair, distance_ideal=2.5, weight=3, slack=0.05),
    geometry_restraints.bond_sorted_asu_proxies)
  assert proxies.simple.size() > 0
  assert proxies.asu.size() > 0
  assert [p for p in proxies.asu if p.i_seq == p.j_seq] # self-interactions
  def residuals_and_weights(i_proxy, t):
    n_simple = proxies.simple.size()
    if (i_proxy < n_simple):
      p = proxies.simple[i_proxy]
      w = 1
    else:
      p = proxies.asu[i_proxy - n_simple]
      w = 1
      if (p.j_sym != 0): w = 0.5
    # the residual depends on the model distance only through delta
    d = geometry_restraints.bond_deltas(
      sites_cart=sites_cart, sorted_asu_proxies=proxies)[i_proxy]
    b = geometry_restraints.bond(
      sites=[(0,0,0),(p.distance_ideal-d+t,0,0)],
      distance_ideal=p.distance_ideal, weight=p.weight, slack=p.slack)
    return w * b.residual()
  expected = gauss_newton_curvatures_by_finite_differences(
    sites_cart=sites_cart,
    deltas=lambda s: geometry_restraints.bond_deltas(
      sites_cart=s, sorted_asu_proxies=proxies),
    residuals_and_weights=residuals_and_weights)
  assert expected.all_ge(0)
  assert flex.max(expected) > 0
  residual_sum_plain = geometry_restraints.bond_residual_sum(
    sites_cart=sites_cart,
    sorted_asu_proxies=proxies,
    gradient_array=None)
  for disable_cache in [False, True]:
    gradients = flex.vec3_double(sites_cart.size(), (0,0,0))
    gradients_plain = flex.vec3_double(sites_cart.size(), (0,0,0))
    curvatures = flex.vec3_double(sites_cart.size(), (0,0,0))
    residual_sum = geometry_restraints.bond_residual_sum(
      sites_cart=sites_cart,
      sorted_asu_proxies=proxies,
      gradient_array=gradients,
      curvature_array=curvatures,
      disable_cache=disable_cache)
    geometry_restraints.bond_residual_sum(
      sites_cart=sites_cart,
      sorted_asu_proxies=proxies,
      gradient_array=gradients_plain,
      disable_cache=disable_cache)
    assert approx_equal(residual_sum, residual_sum_plain)
    assert approx_equal(gradients, gradients_plain)
    assert approx_equal(curvatures.as_double(), expected, eps=1.e-4)
  # curvatures only
  curvatures = flex.vec3_double(sites_cart.size(), (0,0,0))
  geometry_restraints.bond_residual_sum(
    sites_cart=sites_cart,
    sorted_asu_proxies=proxies,
    gradient_array=None,
    curvature_array=curvatures)
  assert approx_equal(curvatures.as_double(), expected, eps=1.e-4)

def exercise_bond_residual_sum_rt_mx():
  # simple proxies with rt_mx_ji, unit_cell overload
  xs = symmetric_structure()
  unit_cell = xs.unit_cell()
  sites_cart = xs.sites_cart()
  proxies = geometry_restraints.shared_bond_simple_proxy()
  for i_seqs, rt_mx_ji in [((0,1), None), ((1,2), "-x,y+1/2,-z"),
                           ((3,3), "-x+1,y+1/2,-z+1"), ((2,4), "x+1,y,z")]:
    p = geometry_restraints.bond_simple_proxy(
      i_seqs=i_seqs, distance_ideal=3, weight=2)
    if (rt_mx_ji is not None):
      p = geometry_restraints.bond_simple_proxy(
        i_seqs=i_seqs, rt_mx_ji=sgtbx.rt_mx(rt_mx_ji),
        distance_ideal=3, weight=2)
    proxies.append(p)
  def deltas(s):
    return geometry_restraints.bond_deltas(
      unit_cell=unit_cell, sites_cart=s, proxies=proxies)
  expected = gauss_newton_curvatures_by_finite_differences(
    sites_cart=sites_cart, deltas=deltas,
    residuals_and_weights=lambda i_proxy, t: proxies[i_proxy].weight
      * (deltas(sites_cart)[i_proxy] + t)**2)
  curvatures = flex.vec3_double(sites_cart.size(), (0,0,0))
  geometry_restraints.bond_residual_sum(
    unit_cell=unit_cell,
    sites_cart=sites_cart,
    proxies=proxies,
    gradient_array=None,
    curvature_array=curvatures)
  assert approx_equal(curvatures.as_double(), expected,
    eps=1.e-4*max(1,flex.max(expected)))
  assert curvatures[3] != (0,0,0) # bond to symmetry copy of the same site

def exercise_angle_residual_sum():
  xs = symmetric_structure()
  unit_cell = xs.unit_cell()
  sites_cart = xs.sites_cart()
  proxies = geometry_restraints.shared_angle_proxy()
  for i_seqs, sym_ops in [
        ((0,1,2), None),
        ((1,2,3), None),
        ((2,4,5), ["x,y,z", "x,y,z", "-x,y+1/2,-z"]),
        ((5,6,7), ["x+1,y,z", "x,y,z", "-x+1,y+1/2,-z"]),
        ((3,6,3), ["x,y,z", "x,y,z", "-x+1,y+1/2,-z+1"])]:
    if (sym_ops is None):
      p = geometry_restraints.angle_proxy(
        i_seqs=i_seqs, angle_ideal=110, weight=2)
    else:
      p = geometry_restraints.angle_proxy(
        i_seqs=i_seqs, sym_ops=[sgtbx.rt_mx(s) for s in sym_ops],
        angle_ideal=110, weight=2)
    proxies.append(p)
  for use_unit_cell in [False, True]:
    def deltas(s):
      if (use_unit_cell):
        return geometry_restraints.angle_deltas(
          unit_cell=unit_cell, sites_cart=s, proxies=proxies)
      return geometry_restraints.angle_deltas(sites_cart=s, proxies=proxies)
    expected = gauss_newton_curvatures_by_finite_differences(
      sites_cart=sites_cart, deltas=deltas,
      residuals_and_weights=lambda i_proxy, t: proxies[i_proxy].weight
        * (deltas(sites_cart)[i_proxy] + t)**2)
    assert flex.max(expected) > 0
    curvatures = flex.vec3_double(sites_cart.size(), (0,0,0))
    gradients = flex.vec3_double(sites_cart.size(), (0,0,0))
    gradients_plain = flex.vec3_double(sites_cart.size(), (0,0,0))
    if (use_unit_cell):
      geometry_restraints.angle_residual_sum(
        unit_cell=unit_cell, sites_cart=sites_cart, proxies=proxies,
        gradient_array=gradients, curvature_array=curvatures)
      geometry_restraints.angle_residual_sum(
        unit_cell=unit_cell, sites_cart=sites_cart, proxies=proxies,
        gradient_array=gradients_plain)
    else:
      geometry_restraints.angle_residual_sum(
        sites_cart=sites_cart, proxies=proxies,
        gradient_array=gradients, curvature_array=curvatures)
      geometry_restraints.angle_residual_sum(
        sites_cart=sites_cart, proxies=proxies,
        gradient_array=gradients_plain)
    assert approx_equal(gradients, gradients_plain)
    assert approx_equal(curvatures.as_double(), expected,
      eps=1.e-4*max(1,flex.max(expected)))

def exercise_nonbonded_residual_sum():
  xs = symmetric_structure()
  sites_cart = xs.sites_cart()
  vdw_distance = 4.0
  proxies = asu_pair_proxies(xs, 5.0,
    lambda pair: geometry_restraints.nonbonded_asu_proxy(
      pair=pair, vdw_distance=vdw_distance),
    geometry_restraints.nonbonded_sorted_asu_proxies)
  assert proxies.simple.size() > 0
  assert proxies.asu.size() > 0
  for nonbonded_type, function in nonbonded_functions():
    def residuals_and_weights(i_proxy, t):
      w = 1
      n_simple = proxies.simple.size()
      if (i_proxy >= n_simple and proxies.asu[i_proxy-n_simple].j_sym != 0):
        w = 0.5
      d = geometry_restraints.nonbonded_deltas(
        sites_cart=sites_cart, sorted_asu_proxies=proxies)[i_proxy]
      return w * nonbonded_type(
        sites=[(0,0,0),(d+t,0,0)],
        vdw_distance=vdw_distance,
        function=function).residual()
    expected = gauss_newton_curvatures_by_finite_differences(
      sites_cart=sites_cart,
      deltas=lambda s: geometry_restraints.nonbonded_deltas(
        sites_cart=s, sorted_asu_proxies=proxies),
      residuals_and_weights=residuals_and_weights)
    assert flex.max(expected) > 0
    for disable_cache in [False, True]:
      gradients = flex.vec3_double(sites_cart.size(), (0,0,0))
      gradients_plain = flex.vec3_double(sites_cart.size(), (0,0,0))
      curvatures = flex.vec3_double(sites_cart.size(), (0,0,0))
      residual_sum = geometry_restraints.nonbonded_residual_sum(
        sites_cart=sites_cart,
        sorted_asu_proxies=proxies,
        gradient_array=gradients,
        curvature_array=curvatures,
        function=function,
        disable_cache=disable_cache)
      residual_sum_plain = geometry_restraints.nonbonded_residual_sum(
        sites_cart=sites_cart,
        sorted_asu_proxies=proxies,
        gradient_array=gradients_plain,
        function=function,
        disable_cache=disable_cache)
      assert approx_equal(residual_sum, residual_sum_plain)
      assert approx_equal(gradients, gradients_plain)
      assert approx_equal(curvatures.as_double(), expected,
        eps=1.e-5*max(1,flex.max(expected)))

def exercise_dihedral():
  for periodicity, limit, top_out, slack in [
        (0, -1, False, 0), (0, -1, False, 5), (2, -1, False, 0),
        (3, -1, False, 0), (0, 20, True, 0)]:
    n_nonzero = 0
    for i_trial in range(20):
      sites = random_sites(4)
      angle_ideal = random.uniform(-180,180)
      def make(sites, t=0):
        return geometry_restraints.dihedral(
          sites=sites, angle_ideal=angle_ideal+t, weight=0.7,
          periodicity=periodicity, limit=limit, top_out=top_out,
          slack=slack)
      d = make(sites)
      if (abs(abs(d.delta) - slack) < 1.e-2): continue
      if (abs(abs(d.delta) - 180) < 1): continue # periodicity of delta
      grad_delta = finite_difference_d1(lambda xs: make(xs).delta, sites)
      # shifting angle_ideal changes delta linearly
      d2r = finite_difference_d2_along(lambda t: make(sites, t).residual(),
        eps=1.e-2)
      expected = gauss_newton(d2r, grad_delta)
      assert approx_equal(d.curvatures(), expected,
        eps=1.e-4*max(1,flex.max(flex.vec3_double(expected).as_double())))
      if (flex.vec3_double(d.curvatures()).as_double().all_ne(0)):
        n_nonzero += 1
    assert n_nonzero > 0

def exercise_chirality():
  for both_signs in [False, True]:
    for i_trial in range(10):
      sites = random_sites(4)
      def make(sites):
        return geometry_restraints.chirality(
          sites=sites, volume_ideal=2.5, both_signs=both_signs, weight=3)
      c = make(sites)
      # the curvatures are the exact diagonal of the 2nd derivatives
      for i_site in range(4):
        for i_coor in range(3):
          def residual_along(t):
            xs = [list(site) for site in sites]
            xs[i_site][i_coor] += t
            return make(xs).residual()
          d2 = finite_difference_d2_along(residual_along)
          assert approx_equal(c.curvatures()[i_site][i_coor], d2,
            eps=1.e-4*max(1,abs(d2)))

def exercise_planarity():
  for i_trial in range(10):
    n_sites = random.choice([4,5,6])
    sites = flex.vec3_double(random_sites(n_sites))
    weights = flex.double([random.uniform(0.5,2) for i in range(n_sites)])
    p = geometry_restraints.planarity(sites=sites, weights=weights)
    curvatures = flex.vec3_double(p.curvatures())
    # exact for the residual with the plane normal held fixed
    n0 = p.normal()
    def residual_fixed_normal(xs):
      xs = flex.vec3_double(xs)
      com = [flex.sum(xs.parts()[k]*weights)/flex.sum(weights)
        for k in range(3)]
      result = 0
      for x, w in zip(xs, weights):
        d = sum(n0[k]*(x[k]-com[k]) for k in range(3))
        result += w * d**2
      return result
    for i_site in range(n_sites):
      for i_coor in range(3):
        def along(f):
          def g(t):
            xs = [list(site) for site in sites]
            xs[i_site][i_coor] += t
            return f(xs)
          return g
        c = curvatures[i_site][i_coor]
        d2_fixed = finite_difference_d2_along(along(residual_fixed_normal))
        assert approx_equal(c, d2_fixed, eps=1.e-4*max(1,abs(c)))
        # upper bound of the exact diagonal
        d2_exact = finite_difference_d2_along(along(
          lambda xs: geometry_restraints.planarity(
            sites=flex.vec3_double(xs), weights=weights).residual()))
        assert c >= d2_exact - 1.e-4*max(1,abs(c))

def p41_structure():
  random.seed(1)
  from cctbx import uctbx
  unit_cell = uctbx.unit_cell((10,10,12,90,90,90))
  sites_cart = flex.vec3_double(
    [[random.uniform(0,4) for i in range(3)] for j in range(8)])
  return unit_cell, sites_cart

def finite_difference_gradients(f, sites_cart, eps=1.e-6):
  result = flex.double()
  for i_seq in range(sites_cart.size()):
    for i_coor in range(3):
      fs = []
      for e in [eps, -eps]:
        s = sites_cart.deep_copy()
        site = list(s[i_seq]); site[i_coor] += e; s[i_seq] = site
        fs.append(f(s))
      result.append((fs[0]-fs[1])/(2*eps))
  return result

def symmetry_proxies():
  """Proxies with non-involutory (4-fold screw) symmetry operations and
  repeated i_seqs."""
  ops = lambda l: [sgtbx.rt_mx(s) for s in l]
  dihedral_proxies = geometry_restraints.shared_dihedral_proxy()
  for i_seqs, sym_ops in [
        ((0,1,2,3), None),
        ((0,1,2,3), ["x,y,z","x,y,z","x,y,z","-y,x,z+1/4"]),
        ((4,5,6,4), ["x,y,z","x,y,z","x,y,z","y,-x,z-1/4"])]:
    if (sym_ops is None):
      dihedral_proxies.append(geometry_restraints.dihedral_proxy(
        i_seqs=i_seqs, angle_ideal=30, weight=0.5))
    else:
      dihedral_proxies.append(geometry_restraints.dihedral_proxy(
        i_seqs=i_seqs, sym_ops=ops(sym_ops), angle_ideal=30, weight=0.5))
  chirality_proxies = geometry_restraints.shared_chirality_proxy()
  for i_seqs, sym_ops in [
        ((0,1,2,3), None),
        ((4,5,6,7), ["x,y,z","x,y,z","-y,x,z+1/4","x,y,z"]),
        ((1,5,1,7), ["x,y,z","x,y,z","y,-x,z-1/4","x,y,z"])]:
    if (sym_ops is None):
      chirality_proxies.append(geometry_restraints.chirality_proxy(
        i_seqs=i_seqs, volume_ideal=2, both_signs=False, weight=3))
    else:
      chirality_proxies.append(geometry_restraints.chirality_proxy(
        i_seqs=i_seqs, sym_ops=ops(sym_ops), volume_ideal=2,
        both_signs=False, weight=3))
  planarity_proxies = geometry_restraints.shared_planarity_proxy()
  for i_seqs, sym_ops in [
        ([0,1,2,3,4], None),
        ([2,3,4,5,6], ["x,y,z"]*4+["-y,x,z+1/4"]),
        ([3,4,5,6,3], ["x,y,z"]*4+["y,-x,z-1/4"])]:
    weights = flex.double([1,2,1.5,1,0.5])
    if (sym_ops is None):
      planarity_proxies.append(geometry_restraints.planarity_proxy(
        i_seqs=flex.size_t(i_seqs), weights=weights))
    else:
      planarity_proxies.append(geometry_restraints.planarity_proxy(
        i_seqs=flex.size_t(i_seqs), sym_ops=ops(sym_ops), weights=weights))
  return dihedral_proxies, chirality_proxies, planarity_proxies

def exercise_symmetry_gradients():
  # gradients with respect to sites related by symmetry operations that
  # are not 2-folds (regression test)
  unit_cell, sites_cart = p41_structure()
  for proxies, residual_sum in zip(symmetry_proxies(), [
        geometry_restraints.dihedral_residual_sum,
        geometry_restraints.chirality_residual_sum,
        geometry_restraints.planarity_residual_sum]):
    gradients = flex.vec3_double(sites_cart.size(), (0,0,0))
    residual_sum(unit_cell=unit_cell, sites_cart=sites_cart,
      proxies=proxies, gradient_array=gradients)
    expected = finite_difference_gradients(
      lambda s: residual_sum(unit_cell=unit_cell, sites_cart=s,
        proxies=proxies, gradient_array=None),
      sites_cart)
    assert approx_equal(gradients.as_double(), expected,
      eps=1.e-5*max(1,flex.max(flex.abs(expected))))

def site_with_sym_op(unit_cell, site, rt_mx):
  return unit_cell.orthogonalize(rt_mx * unit_cell.fractionalize(site))

def exercise_dihedral_chirality_residual_sums():
  unit_cell, sites_cart = p41_structure()
  dihedral_proxies, chirality_proxies, _ = symmetry_proxies()
  for all_proxies, residual_sum, deltas in [
        (dihedral_proxies, geometry_restraints.dihedral_residual_sum,
         geometry_restraints.dihedral_deltas),
        (chirality_proxies, geometry_restraints.chirality_residual_sum,
         geometry_restraints.chirality_deltas)]:
    for use_unit_cell in [False, True]:
      uc_kwds = {}
      proxies = all_proxies
      if (use_unit_cell):
        uc_kwds = {"unit_cell": unit_cell}
      elif (residual_sum is geometry_restraints.dihedral_residual_sum):
        # without symmetry, a dihedral with repeated i_seqs is always zero
        # and its finite-difference derivatives are pure rounding noise
        proxies = proxies.__class__(
          [p for p in proxies if len(set(p.i_seqs)) == 4])
      # harmonic residuals: d2r = 2 * weight
      expected = gauss_newton_curvatures_by_finite_differences(
        sites_cart=sites_cart,
        deltas=lambda s: deltas(sites_cart=s, proxies=proxies, **uc_kwds),
        residuals_and_weights=lambda i_proxy, t: proxies[i_proxy].weight
          * (deltas(sites_cart=sites_cart, proxies=proxies,
                    **uc_kwds)[i_proxy] + t)**2)
      assert flex.max(expected) > 0
      gradients = flex.vec3_double(sites_cart.size(), (0,0,0))
      gradients_plain = flex.vec3_double(sites_cart.size(), (0,0,0))
      curvatures = flex.vec3_double(sites_cart.size(), (0,0,0))
      r = residual_sum(sites_cart=sites_cart, proxies=proxies,
        gradient_array=gradients, curvature_array=curvatures, **uc_kwds)
      r_plain = residual_sum(sites_cart=sites_cart, proxies=proxies,
        gradient_array=gradients_plain, **uc_kwds)
      assert approx_equal(r, r_plain)
      assert approx_equal(gradients, gradients_plain)
      assert approx_equal(curvatures.as_double(), expected,
        eps=1.e-4*max(1,flex.max(expected)))

def exercise_planarity_residual_sum():
  unit_cell, sites_cart = p41_structure()
  _, _, proxies = symmetry_proxies()
  for use_unit_cell in [False, True]:
    # brute force: 2 * sum_k w_k * (d(delta_k)/d(x))**2 with fixed normals
    normals = []
    for proxy in proxies:
      if (use_unit_cell):
        p = geometry_restraints.planarity(
          unit_cell=unit_cell, sites_cart=sites_cart, proxy=proxy)
      else:
        p = geometry_restraints.planarity(sites_cart=sites_cart, proxy=proxy)
      normals.append(p.normal())
    def plane_deltas(s):
      result = flex.double()
      for proxy, n in zip(proxies, normals):
        xs = []
        for k, i_seq in enumerate(proxy.i_seqs):
          x = s[i_seq]
          if (use_unit_cell and proxy.sym_ops is not None
              and not proxy.sym_ops[k].is_unit_mx()):
            x = site_with_sym_op(unit_cell, x, proxy.sym_ops[k])
          xs.append(x)
        xs = flex.vec3_double(xs)
        com = [flex.sum(xs.parts()[k]*proxy.weights)/flex.sum(proxy.weights)
          for k in range(3)]
        for x in xs:
          result.append(sum(n[k]*(x[k]-com[k]) for k in range(3)))
      return result
    w = flex.double()
    for proxy in proxies: w.extend(proxy.weights)
    expected = flex.double(sites_cart.size()*3, 0)
    eps = 1.e-6
    for i_seq in range(sites_cart.size()):
      for i_coor in range(3):
        sp = sites_cart.deep_copy()
        site = list(sp[i_seq]); site[i_coor] += eps; sp[i_seq] = site
        sm = sites_cart.deep_copy()
        site = list(sm[i_seq]); site[i_coor] -= eps; sm[i_seq] = site
        g = (plane_deltas(sp) - plane_deltas(sm)) / (2*eps)
        expected[i_seq*3+i_coor] = 2 * flex.sum(w * g**2)
    assert flex.max(expected) > 0
    uc_kwds = {}
    if (use_unit_cell): uc_kwds = {"unit_cell": unit_cell}
    curvatures = flex.vec3_double(sites_cart.size(), (0,0,0))
    gradients = flex.vec3_double(sites_cart.size(), (0,0,0))
    gradients_plain = flex.vec3_double(sites_cart.size(), (0,0,0))
    geometry_restraints.planarity_residual_sum(
      sites_cart=sites_cart, proxies=proxies, gradient_array=gradients,
      curvature_array=curvatures, **uc_kwds)
    geometry_restraints.planarity_residual_sum(
      sites_cart=sites_cart, proxies=proxies,
      gradient_array=gradients_plain, **uc_kwds)
    assert approx_equal(gradients, gradients_plain)
    assert approx_equal(curvatures.as_double(), expected,
      eps=1.e-5*max(1,flex.max(expected)))

def exercise_energies():
  from cctbx.geometry_restraints.energies import energies
  xs = symmetric_structure()
  unit_cell = xs.unit_cell()
  sites_cart = xs.sites_cart()
  bond_proxies = asu_pair_proxies(xs, 3.0,
    lambda pair: geometry_restraints.bond_asu_proxy(
      pair=pair, distance_ideal=2.5, weight=3),
    geometry_restraints.bond_sorted_asu_proxies)
  nonbonded_proxies = asu_pair_proxies(xs, 5.0,
    lambda pair: geometry_restraints.nonbonded_asu_proxy(
      pair=pair, vdw_distance=4.0),
    geometry_restraints.nonbonded_sorted_asu_proxies)
  nonbonded_function = geometry_restraints.prolsq_repulsion_function()
  angle_proxies = geometry_restraints.shared_angle_proxy()
  angle_proxies.append(geometry_restraints.angle_proxy(
    i_seqs=(0,1,2), angle_ideal=110, weight=2))
  angle_proxies.append(geometry_restraints.angle_proxy(
    i_seqs=(3,6,3),
    sym_ops=[sgtbx.rt_mx(s) for s in ["x,y,z","x,y,z","-x+1,y+1/2,-z+1"]],
    angle_ideal=110, weight=2))
  expected = flex.vec3_double(sites_cart.size(), (0,0,0))
  geometry_restraints.bond_residual_sum(
    sites_cart=sites_cart, sorted_asu_proxies=bond_proxies,
    gradient_array=None, curvature_array=expected)
  geometry_restraints.nonbonded_residual_sum(
    sites_cart=sites_cart, sorted_asu_proxies=nonbonded_proxies,
    gradient_array=None, curvature_array=expected,
    function=nonbonded_function)
  geometry_restraints.angle_residual_sum(
    unit_cell=unit_cell, sites_cart=sites_cart, proxies=angle_proxies,
    gradient_array=None, curvature_array=expected)
  dihedral_proxies, chirality_proxies, planarity_proxies = \
    symmetry_proxies()
  geometry_restraints.dihedral_residual_sum(
    unit_cell=unit_cell, sites_cart=sites_cart, proxies=dihedral_proxies,
    gradient_array=None, curvature_array=expected)
  geometry_restraints.chirality_residual_sum(
    sites_cart=sites_cart, proxies=chirality_proxies,
    gradient_array=None, curvature_array=expected)
  geometry_restraints.planarity_residual_sum(
    unit_cell=unit_cell, sites_cart=sites_cart, proxies=planarity_proxies,
    gradient_array=None, curvature_array=expected)
  assert flex.max(expected.as_double()) > 0
  for normalization in [False, True]:
    kwds = dict(
      sites_cart=sites_cart,
      unit_cell=unit_cell,
      bond_proxies=bond_proxies,
      nonbonded_proxies=nonbonded_proxies,
      nonbonded_function=nonbonded_function,
      angle_proxies=angle_proxies,
      dihedral_proxies=dihedral_proxies,
      chirality_proxies=chirality_proxies,
      planarity_proxies=planarity_proxies,
      normalization=normalization)
    e = energies(compute_curvatures=True, **kwds)
    e_plain = energies(**kwds)
    assert e_plain.curvatures is None
    assert approx_equal(e.target, e_plain.target)
    assert approx_equal(e.gradients, e_plain.gradients)
    factor = 1
    if (normalization): factor = e.normalization_factor
    assert approx_equal(e.curvatures, expected*factor)

def exercise_lbfgs():
  import cctbx.geometry_restraints.manager
  import cctbx.geometry_restraints.lbfgs
  import scitbx.lbfgs
  sites_cart_start = flex.vec3_double([
    (1,3,0), (2,3,0), (3,2,0), (3,1,0), (4,1,0), (3,4,0), (4,3,0), (5,3,0)])
  bond_proxies = geometry_restraints.shared_bond_simple_proxy()
  for i_seqs in [(0,1),(1,2),(2,3),(3,4),(1,5),(2,6),(5,6),(6,7)]:
    bond_proxies.append(geometry_restraints.bond_simple_proxy(
      i_seqs=i_seqs, distance_ideal=1.5, weight=100))
  angle_proxies = geometry_restraints.shared_angle_proxy()
  for i_seqs, angle_ideal in [[(0,1,2),135], [(0,1,5),135], [(1,2,3),135],
                              [(3,2,6),135], [(2,3,4),120], [(1,2,6),90],
                              [(2,6,5),90], [(6,5,1),90], [(5,1,2),90],
                              [(2,6,7),135], [(5,6,7),135]]:
    angle_proxies.append(geometry_restraints.angle_proxy(
      i_seqs=i_seqs, angle_ideal=angle_ideal, weight=1))
  manager = geometry_restraints.manager.manager(
    bond_params_table=geometry_restraints.extract_bond_params(
      n_seq=sites_cart_start.size(), bond_simple_proxies=bond_proxies),
    angle_proxies=angle_proxies)
  for sites_cart_selection in [None, flex.bool([True]*4+[False]+[True]*3)]:
    sites_cart = sites_cart_start.deep_copy()
    minimized = geometry_restraints.lbfgs.lbfgs(
      sites_cart=sites_cart,
      geometry_restraints_manager=manager,
      lbfgs_termination_params=scitbx.lbfgs.termination_parameters(
        max_iterations=1000),
      sites_cart_selection=sites_cart_selection,
      use_curvatures=True)
    assert minimized.final_target_value < 1.e-6
    assert flex.max(flex.abs(geometry_restraints.bond_deltas(
      sites_cart=sites_cart, proxies=bond_proxies))) < 1.e-4
    if (sites_cart_selection is not None):
      assert sites_cart[4] == sites_cart_start[4]

def run():
  random.seed(0)
  exercise_bond()
  exercise_angle()
  exercise_nonbonded()
  exercise_bond_residual_sum()
  exercise_bond_residual_sum_rt_mx()
  exercise_angle_residual_sum()
  exercise_nonbonded_residual_sum()
  exercise_dihedral()
  exercise_chirality()
  exercise_planarity()
  exercise_symmetry_gradients()
  exercise_dihedral_chirality_residual_sums()
  exercise_planarity_residual_sum()
  exercise_energies()
  exercise_lbfgs()
  print("OK")

if (__name__ == "__main__"):
  run()
