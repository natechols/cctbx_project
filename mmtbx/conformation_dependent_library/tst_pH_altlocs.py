"""
pH-dependent restraints: atoms of different conformers are not paired
(mmtbx.conformation_dependent_library.pH_dependent_restraints).
"""
from __future__ import absolute_import, division, print_function
import iotbx.pdb
from mmtbx.conformation_dependent_library import pH_dependent_restraints as phr

pdb_str = """\
HETATM    1  N   GLY A   1       1.000   1.000   1.000  1.00 10.00           N
HETATM    2  CA AGLY A   1       2.000   1.000   1.000  0.50 10.00           C
HETATM    3  C  AGLY A   1       3.000   1.000   1.000  0.50 10.00           C
HETATM    4  CA BGLY A   1       2.000   1.300   1.000  0.50 10.00           C
HETATM    5  C  BGLY A   1       3.000   1.300   1.000  0.50 10.00           C
"""

def exercise():
  h = iotbx.pdb.input(source_info=None, lines=pdb_str).construct_hierarchy()
  rg = h.only_residue_group()
  def bonded(name1, name2):
    # (the generator depends on the order of the atom groups: both orders)
    result = set()
    for n1, n2 in [(name1, name2), (name2, name1)]:
      for x in phr._generate_bond_atoms(rg, n1, n2):
        result.add(tuple(sorted((x[0].parent().altloc, x[1].parent().altloc))))
    return sorted(result)
  # N (no altloc) with CA of each conformer; CA-C within each conformer only
  assert bonded("N", "CA") == [("", "A"), ("", "B")], bonded("N", "CA")
  assert bonded("CA", "C") == [("A", "A"), ("B", "B")], bonded("CA", "C")
  angles = [(x[0], x[2]) for x in phr._generate_angle_atoms(
    rg, "N", "CA", "C")]
  for a1, a3 in angles:
    assert phr._compatible_atom_groups(a1.parent(), a3.parent())
  dihedrals = [(x[0], x[3]) for x in phr._generate_dihedral_atoms(
    rg, "N", "CA", "C", "CA")]
  for a1, a4 in dihedrals:
    assert phr._compatible_atom_groups(a1.parent(), a4.parent())

if (__name__ == "__main__"):
  exercise()
  print("OK")
