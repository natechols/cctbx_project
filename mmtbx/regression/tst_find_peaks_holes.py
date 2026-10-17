"""Tests for mmtbx.command_line.find_peaks_holes."""
from __future__ import absolute_import, division, print_function
import iotbx.pdb
import mmtbx.f_model
from mmtbx.command_line.find_peaks_holes import find_peaks_holes
from mmtbx.regression.tst_minimization_curvatures import pdb_str
from libtbx.utils import null_out

def exercise_anomalous():
  """
  Anomalous data (F+ and F-) from a model with f''=4 for the sulfur of Met 1;
  the model of the fmodel has no f'', so the anomalous residual map has a
  peak at the sulfur (the anomalous branch used to fail with
  "unexpected keyword argument 'fmodel'").
  """
  pdb_inp = iotbx.pdb.input(source_info=None, lines=pdb_str)
  hierarchy = pdb_inp.construct_hierarchy()
  xrs = pdb_inp.xray_structure_simple()
  atoms = hierarchy.atoms()
  i_sd = [a.i_seq for a in atoms if a.name.strip() == "SD"]
  assert len(i_sd) == 1
  xrs_anom = xrs.deep_copy_scatterers()
  xrs_anom.scatterers()[i_sd[0]].fdp = 4
  f_obs = abs(xrs_anom.structure_factors(d_min=2.0, anomalous_flag=True,
    algorithm="direct").f_calc())
  assert f_obs.anomalous_flag()
  flags = f_obs.generate_r_free_flags(fraction=0.1)
  fmodel = mmtbx.f_model.manager(
    f_obs          = f_obs,
    r_free_flags   = flags,
    xray_structure = xrs)
  fmodel.update_all_scales(log=null_out())
  # (peaks at model atoms are excluded by default, min_model_peak_dist)
  result = find_peaks_holes(
    fmodel                   = fmodel,
    pdb_hierarchy            = hierarchy,
    use_phaser_if_available  = False,
    include_peaks_near_model = True,
    out                      = null_out())
  assert result.anom_peaks is not None
  heights = result.anom_peaks.heights
  assert len(heights) > 0
  # highest anomalous peak at the sulfur (also with symmetry)
  uc = xrs.unit_cell()
  sg = xrs.space_group()
  site = uc.fractionalize(xrs.sites_cart()[i_sd[0]])
  i_max = list(heights).index(max(heights))
  peak = uc.fractionalize(result.anom_peaks.sites[i_max])
  d = min([uc.distance(site, op * peak) for op in sg.all_ops()])
  assert d < 0.5, d
  # without anomalous data: no anomalous peaks
  fmodel = mmtbx.f_model.manager(
    f_obs          = f_obs.average_bijvoet_mates(),
    r_free_flags   = flags.average_bijvoet_mates(),
    xray_structure = xrs)
  fmodel.update_all_scales(log=null_out())
  result = find_peaks_holes(
    fmodel                  = fmodel,
    pdb_hierarchy           = hierarchy,
    use_phaser_if_available = False,
    out                     = null_out())
  assert result.anom_peaks is None

def run():
  exercise_anomalous()
  print("OK")

if (__name__ == "__main__"):
  run()
