"""
Tests for manager.remove_dihedrals_through_linear_angles().
"""
from __future__ import absolute_import, division, print_function
from cctbx import geometry_restraints
import cctbx.geometry_restraints.manager
from cctbx.array_family import flex

def exercise():
  # chain 0-1-2-3-4-5 with a linear angle 2-3-4 (e.g. an alkyne)
  sites_cart = flex.vec3_double([
    (0,0,0), (1.5,0,0), (2,1.4,0), (3,2,0.3), (4,2.6,0.6), (5,3.2,0.9)])
  angles = geometry_restraints.shared_angle_proxy()
  for i_seqs, ideal in [((0,1,2), 109.5), ((1,2,3), 120), ((4,3,2), 180),
                        ((3,4,5), 179)]:
    angles.append(geometry_restraints.angle_proxy(
      i_seqs=i_seqs, angle_ideal=ideal, weight=1))
  dihedrals = geometry_restraints.shared_dihedral_proxy()
  for i_seqs in [(0,1,2,3), (1,2,3,4), (2,3,4,5), (5,4,3,2), (3,4,5,0)]:
    dihedrals.append(geometry_restraints.dihedral_proxy(
      i_seqs=i_seqs, angle_ideal=60, weight=1))
  manager = geometry_restraints.manager.manager(
    angle_proxies=angles, dihedral_proxies=dihedrals)
  removed = manager.remove_dihedrals_through_linear_angles()
  # 1-2-3-4 through 2-3-4 (given as 4-3-2); 2-3-4-5 and 5-4-3-2 through
  # 2-3-4 and 3-4-5; 3-4-5-0 through 3-4-5
  assert [tuple(p.i_seqs) for p in removed] == [
    (1,2,3,4), (2,3,4,5), (5,4,3,2), (3,4,5,0)]
  assert [tuple(p.i_seqs) for p in manager.dihedral_proxies] == [(0,1,2,3)]
  # with a larger limit, 3-4-5 (179 degrees) is not linear
  manager = geometry_restraints.manager.manager(
    angle_proxies=angles, dihedral_proxies=dihedrals.deep_copy())
  removed = manager.remove_dihedrals_through_linear_angles(
    min_angle_ideal=179.5)
  assert [tuple(p.i_seqs) for p in removed] == [
    (1,2,3,4), (2,3,4,5), (5,4,3,2)]
  assert [tuple(p.i_seqs) for p in manager.dihedral_proxies] == [
    (0,1,2,3), (3,4,5,0)]
  removed = manager.remove_dihedrals_through_linear_angles(
    min_angle_ideal=179.5)
  assert removed.size() == 0
  # nothing to remove
  manager = geometry_restraints.manager.manager(
    angle_proxies=angles[:2], dihedral_proxies=dihedrals.deep_copy())
  assert manager.remove_dihedrals_through_linear_angles().size() == 0
  assert manager.dihedral_proxies.size() == 5
  # the gradients of a dihedral through nearly collinear atoms diverge
  for eps, g_min in [(1.e-3, 1.e2), (1.e-6, 1.e8)]:
    s3, s4 = sites_cart[3], sites_cart[4]
    sites_cart[5] = (2*s4[0]-s3[0], 2*s4[1]-s3[1]+eps, 2*s4[2]-s3[2])
    g = geometry_restraints.dihedral(sites_cart=sites_cart,
      proxy=dihedrals[2]).gradients()
    assert max([abs(x) for v in g for x in v]) > g_min

def run():
  exercise()
  print("OK")

if (__name__ == "__main__"):
  run()
