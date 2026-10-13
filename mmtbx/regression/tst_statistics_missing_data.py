"""
mmtbx.model.statistics without the probe program, the CaBLAM contours and the
Rama-Z database: the statistics are None, and the model can still be written
as mmCIF (which includes the statistics).
"""
from __future__ import absolute_import, division, print_function
import iotbx.pdb
import libtbx.load_env
import mmtbx.hydrogens
import mmtbx.model
import mmtbx.validation.clashscore2
from mmtbx.regression.tst_minimization_curvatures import pdb_str

def exercise():
  n_clashscore2 = [0]
  def clashscore2(*args, **kwds):
    n_clashscore2[0] += 1
    raise RuntimeError("mmtbx.probe2 not available")
  saved = (libtbx.env.has_module, libtbx.env.find_in_repositories,
    mmtbx.validation.clashscore2.clashscore2, mmtbx.hydrogens.use_old_reduce)
  find_in_repositories = libtbx.env.find_in_repositories
  def find_in_repositories_no_data(relative_path, *args, **kwds):
    if (relative_path.startswith("chem_data/cablam_data") or
        relative_path.startswith("chem_data/rama_z")):
      return None
    return find_in_repositories(relative_path, *args, **kwds)
  try:
    libtbx.env.has_module = lambda name: (name != "probe")
    libtbx.env.find_in_repositories = find_in_repositories_no_data
    mmtbx.validation.clashscore2.clashscore2 = clashscore2
    # clashscore also for the model without hydrogens
    mmtbx.hydrogens.use_old_reduce = lambda: True
    model = mmtbx.model.manager(
      model_input = iotbx.pdb.input(source_info=None, lines=pdb_str))
    model.process(make_restraints=True)
    geometry = model.get_model_statistics_info().geometry
    r = geometry.result()
    assert r.clash.score is None
    assert r.molprobity_score is None
    assert r.cablam.outliers is None and r.cablam.ca_outliers is None
    assert r.rama_z.whole.value is None and r.rama_z.whole.n == 0
    assert n_clashscore2[0] == 1 # not retried
    assert geometry.show_short().count("None") > 0
    cif = model.model_as_mmcif()
    assert cif.count("ATOM") > 0
  finally:
    (libtbx.env.has_module, libtbx.env.find_in_repositories,
      mmtbx.validation.clashscore2.clashscore2,
      mmtbx.hydrogens.use_old_reduce) = saved

if (__name__ == "__main__"):
  exercise()
  print("OK")
