"""
Molecular mechanics force field (OpenMM) in place of the geometry restraints
of the macromolecule, for coordinate refinement.

The atoms of complete protein and nucleic-acid residues (with hydrogens) are
described by an OpenMM force field (default: Amber ff14SB/OL15/OL3,
amber14-all.xml); all other atoms (water, ligands, ions, residues without a
force-field template) keep the cctbx geometry restraints. The target is

  E = E_cctbx(all atoms) - E_cctbx(force-field atoms) + 2*E_MM/kT

where E_cctbx(force-field atoms) are the standard geometry restraints (bonds,
angles, dihedrals, chiralities, planarities, parallelities and nonbonded
repulsion) among the force-field atoms within the asymmetric unit. Kept from
E_cctbx(all atoms): restraints involving other atoms (including the bonds to
ligands and to residues outside the force field), the nonbonded repulsion
between symmetry copies, and NCS, reference and other additional restraints.
The MM energy is scaled like the cctbx residuals (sum of (delta/sigma)**2,
i.e. twice the energy in units of kT).

Alternative conformations: one OpenMM system per altloc (atoms without altloc
and the atoms with this altloc; residues without this altloc contribute their
first altloc); E_MM is the mean over the systems.

Residues that do not match a template of the force field exactly (truncated
side chains, chain breaks, C-termini without OXT, residues next to residues
outside the force field) get a template derived from the template of the same
residue type with the most similar atoms: the missing atoms are removed, their
charge is spread evenly over the remaining atoms (the net charge of the
residue is kept), and the external bonds are those of the model. Residues
that still do not match keep the cctbx geometry restraints.

No curvatures: refine coordinates with use_curvatures=False.

OpenMM (MIT license for the API and the CPU and Reference platforms) is an
optional dependency: conda install -c conda-forge openmm
"""
from __future__ import absolute_import, division, print_function
from cctbx.array_family import flex
from cctbx import geometry_restraints, uctbx
from libtbx.utils import Sorry
import iotbx.phil
import mmtbx.restraints
import scitbx.restraints
import os

master_phil_str = """
enabled = False
  .type = bool
  .help = Use a molecular mechanics force field (OpenMM) for the protein and \
          nucleic-acid residues instead of the geometry restraints \
          (coordinate refinement). Requires hydrogens.
forcefield = amber14-all.xml
  .type = strings
  .help = OpenMM force field files
implicit_solvent = *none gbn2 obc2
  .type = choice
  .help = Implicit solvent model (OpenMM implicit/*.xml). none: reaction \
          field with the solvent dielectric beyond the cutoff
implicit_solvent_final_cycle = False
  .type = bool
  .help = Use the implicit solvent model only in the final macro-cycle \
          (faster). If the refinement converges before \
          number_of_macro_cycles, one more macro-cycle with the implicit \
          solvent model is done.
nonbonded_cutoff = 10.0
  .type = float(value_min=1)
  .help = Cutoff (A) of the nonbonded interactions
optimize_hydrogens = True
  .type = bool
  .help = Minimize the force-field energy with respect to the hydrogens \
          (heavy atoms fixed) before refinement, e.g. to orient hydroxyl \
          groups (the riding hydrogens do not rotate during refinement)
start_minimization_sigma = 0.05
  .type = float(value_min=0)
  .help = Before refinement: minimize the force-field energy with respect to \
          all atoms of the force field, with harmonic restraints of the \
          heavy atoms to their positions (sigma, A), to remove clashes (the \
          repulsion of the force field is much steeper than that of the \
          geometry restraints). 0: not done.
temperature = 300
  .type = float(value_min=1)
  .help = Temperature (K) for the scale of the energy (kT)
selection = protein or nucleotide
  .type = atom_selection
  .help = Residues described by the force field (residues with atoms outside \
          the selection, or without a template, keep the geometry restraints)
platform = *CPU Reference
  .type = choice
threads = None
  .type = int(value_min=1)
  .help = Threads of the CPU platform (default: OMP_NUM_THREADS if set, \
          otherwise all cores)
"""
master_params = iotbx.phil.parse(master_phil_str)

kilojoule_per_mole_per_kelvin = 0.0083144626 # Boltzmann constant * N_A

def import_openmm():
  try:
    import openmm
    import openmm.app
    import openmm.unit
  except ImportError:
    raise Sorry("OpenMM is not installed (conda install -c conda-forge "
      "openmm).")
  return openmm

# -----------------------------------------------------------------------------

def residue_variants(resname):
  """Names of the Amber templates of a residue type (terminal variants,
  protonation states, disulfides, 5'/3' nucleotides)."""
  r = resname.strip()
  names = [r]
  names += {"HIS": ["HID", "HIE", "HIP"], "CYS": ["CYX", "CYM"],
    "ASP": ["ASH"], "GLU": ["GLH"], "LYS": ["LYN"]}.get(r, [])
  result = []
  for n in names:
    result += [n, "N" + n, "C" + n]
    result += [n + "5", n + "3", n + "N"] # nucleotides
  return result

class residue_templates(object):
  """
  The template of each residue: the template of the force field that matches
  it exactly, otherwise a template derived from the template of the same
  residue type that contains all atoms of the residue and the fewest other
  atoms. Derived templates are registered in the force field that creates the
  systems (forcefield), not in the one used for the matching
  (forcefield_match): truncated residues of different types can have
  identical graphs (e.g. LYS and GLU without side chain beyond CB), so all
  residues get explicit templates (createSystem(residueTemplates=...)).
  """
  def __init__(self, files, solvents=(None,)):
    app = import_openmm().app
    # one force field per implicit solvent model (None: without)
    self.forcefields = dict([(solvent, app.ForceField(*(files +
      ([] if solvent is None else ["implicit/%s.xml" % solvent]))))
      for solvent in solvents])
    self.forcefield_match = app.ForceField(*files)
    self.derived = {} # signature -> name of the derived template
    self.n_derived = {} # name of the original template -> residues

  def __call__(self, topology):
    """Template names of the residues of the topology (dict), and the
    residues without a template."""
    ff = self.forcefield_match
    bonded_to_atom = ff._buildBondedToAtomList(topology)
    result, unmatched = {}, []
    for residue in topology.residues():
      try:
        template, matches = ff._getResidueTemplateMatches(residue,
          bonded_to_atom)
      except Exception: # several non-identical matches
        template, matches = None, None
      if (matches is not None):
        result[residue] = template.name
        continue
      name = self.derive(residue)
      if (name is None):
        unmatched.append(residue)
      else:
        result[residue] = name
    return result, unmatched

  def find_template(self, residue):
    names = set([a.name for a in residue.atoms()])
    elements = dict([(a.name, a.element) for a in residue.atoms()])
    best = None
    for name in residue_variants(residue.name):
      t = self.forcefield_match._templates.get(name)
      if (t is None): continue
      t_names = dict([(a.name, a) for a in t.atoms])
      if (not names.issubset(t_names)): continue
      if (any([t_names[n].element is not None and elements[n] is not None and
               t_names[n].element.symbol != elements[n].symbol
               for n in names])):
        continue
      n_missing = len(t_names) - len(names)
      if (best is None or n_missing < best[0]):
        best = (n_missing, t)
    return None if best is None else best[1]

  def derive(self, residue):
    """Name of the derived template for the residue (None if there is no
    template of this residue type with all its atoms)."""
    ff = import_openmm().app.ForceField
    t = self.find_template(residue)
    if (t is None): return None
    bonds = tuple(sorted([tuple(sorted([b[0].name, b[1].name]))
      for b in residue.internal_bonds()]))
    external = tuple(sorted([(b[0] if b[0].residue is residue else b[1]).name
      for b in residue.external_bonds()]))
    names = set([a.name for a in residue.atoms()])
    signature = (t.name, tuple(sorted(names)), bonds, external)
    if (signature in self.derived):
      return self.derived[signature]
    kept = [a for a in t.atoms if a.name in names]
    q_total = sum([a.parameters.get("charge", 0) for a in t.atoms])
    q_kept = sum([a.parameters.get("charge", 0) for a in kept])
    dq = (q_total - q_kept) / len(kept)
    new = ff._TemplateData("%s_quickref%d" % (t.name, len(self.derived)+1))
    for a in kept:
      parameters = dict(a.parameters)
      if ("charge" in parameters): parameters["charge"] += dq
      new.addAtom(ff._TemplateAtomData(a.name, a.type, a.element, parameters))
    for n1, n2 in bonds:
      new.addBondByName(n1, n2)
    for n in external:
      new.addExternalBondByName(n)
    for forcefield in self.forcefields.values():
      forcefield.registerResidueTemplate(new)
    self.derived[signature] = new.name
    self.n_derived[t.name] = self.n_derived.get(t.name, 0) + 1
    return new.name

# -----------------------------------------------------------------------------

class conformer_system(object):
  """One OpenMM system (one conformer) and its atoms (i_seqs)."""
  def __init__(self, altloc, i_seqs, topology, system, context):
    self.altloc = altloc
    self.i_seqs = i_seqs
    self.topology = topology
    self.system = system
    self.context = context

class force_field(object):
  """
  OpenMM systems of the force-field atoms of a model (one per conformer) and
  the cctbx geometry restraints that they replace.
  """
  def __init__(self, model, params, log):
    import_openmm()
    self.params = params
    self.log = log
    self.kt = kilojoule_per_mole_per_kelvin * params.temperature
    hierarchy = model.get_hierarchy()
    hierarchy.atoms().reset_i_seq()
    geometry = model.get_restraints_manager().geometry
    sites_cart = model.get_sites_cart()
    n_atoms = sites_cart.size()
    self.files = list(params.forcefield)
    solvent = params.implicit_solvent
    if (solvent == "none"): solvent = None
    # implicit solvent only in the final macro-cycle: set_solvent()
    self.solvent_final_cycle = (solvent is not None and
      params.implicit_solvent_final_cycle)
    # covalent bonds (within the asymmetric unit)
    self.bonded = {}
    self.distance_ideal = {}
    for proxy in geometry.pair_proxies(
        sites_cart = sites_cart).bond_proxies.simple:
      i, j = proxy.i_seqs
      self.bonded.setdefault(i, []).append(j)
      self.bonded.setdefault(j, []).append(i)
      self.distance_ideal[(i, j)] = self.distance_ideal[(j, i)] = \
        proxy.distance_ideal
    # candidate residues: all atoms selected, with hydrogens and restraints
    selection = model.selection(params.selection)
    unrestrained = (model._type_energies == "")
    has_h = model.get_hd_selection()
    excluded = {} # residue_group id -> reason
    candidates = []
    for rg in hierarchy.residue_groups():
      i_seqs = rg.atoms().extract_i_seq()
      if (selection.select(i_seqs).count(False) > 0): continue
      if (unrestrained.select(i_seqs).count(True) > 0):
        excluded[self._rg_id(rg)] = "no restraints"
        continue
      if (has_h.select(i_seqs).count(True) == 0):
        excluded[self._rg_id(rg)] = "no hydrogens"
        continue
      candidates.append(rg)
    if (len(candidates) == 0):
      raise Sorry("Force field: no residues (selection: %s)" %
        params.selection)
    altlocs = sorted(set([ag.altloc for rg in candidates
      for ag in rg.atom_groups() if ag.altloc != ""]))
    self.altlocs = altlocs if len(altlocs) > 0 else [""]
    # residues without a template are removed, then the topologies rebuilt
    self.templates = residue_templates(self.files, solvents = [None, solvent])
    for i_pass in range(10):
      topologies = [self._topology(candidates, altloc)
        for altloc in self.altlocs]
      unmatched = set()
      templates = []
      for topology, i_seqs, residue_groups in topologies:
        names, residues = self.templates(topology)
        templates.append(names)
        for residue in residues:
          unmatched.add(id(residue_groups[residue.index]))
      if (len(unmatched) == 0): break
      for rg in candidates:
        if (id(rg) in unmatched):
          excluded[self._rg_id(rg)] = "no force-field template"
      candidates = [rg for rg in candidates if id(rg) not in unmatched]
    else:
      raise Sorry("Force field: residues without templates.")
    self.excluded = excluded
    self._topologies = topologies
    self._residue_templates = templates
    self.solvent = None
    self.set_solvent(None if self.solvent_final_cycle else solvent)
    # all force-field atoms (all conformers)
    self.selection = flex.bool(n_atoms, False)
    for s in self.systems:
      self.selection.set_selected(s.i_seqs, True)
    self.iselection = self.selection.iselection()
    self.n_residues = len(candidates)
    # cctbx restraints among the force-field atoms, without symmetry
    sites_ff = sites_cart.select(self.iselection)
    self.geometry_ff = geometry.select(iselection = self.iselection).\
      discard_symmetry(new_unit_cell = uctbx.non_crystallographic_unit_cell(
        sites_cart = sites_ff, buffer_layer = 50))
    self.flags_ff = geometry_restraints.flags.flags(bond = True,
      nonbonded = True, angle = True, dihedral = True, chirality = True,
      planarity = True, parallelity = True)

  def set_solvent(self, solvent):
    """(Re)creates the OpenMM systems, with the implicit solvent model
    (None: without)."""
    openmm = import_openmm()
    params = self.params
    method = openmm.app.CutoffNonPeriodic
    # (a Quantity: the scripts of implicit/*.xml use it)
    cutoff = params.nonbonded_cutoff * 0.1 * openmm.unit.nanometer
    platform = openmm.Platform.getPlatformByName(params.platform)
    properties = {}
    if (params.platform == "CPU"):
      threads = params.threads
      if (threads is None and os.environ.get("OMP_NUM_THREADS")):
        threads = int(os.environ["OMP_NUM_THREADS"])
      if (threads is not None):
        properties["Threads"] = str(threads)
    forcefield = self.templates.forcefields[solvent]
    self.systems = []
    for altloc, (topology, i_seqs, residue_groups), names in zip(
        self.altlocs, self._topologies, self._residue_templates):
      system = forcefield.createSystem(topology,
        residueTemplates = names,
        nonbondedMethod = method,
        nonbondedCutoff = cutoff,
        constraints     = None,
        rigidWater      = False,
        removeCMMotion  = False)
      integrator = openmm.VerletIntegrator(0.001)
      context = openmm.Context(system, integrator, platform, properties)
      s = conformer_system(altloc = altloc, i_seqs = i_seqs,
        topology = topology, system = system, context = context)
      self._setup_hydrogen_scale(s)
      self.systems.append(s)
    self.solvent = solvent

  def _setup_hydrogen_scale(self, s):
    """
    The riding hydrogens are at the X-H distances of the geometry restraints
    (X-ray: electron-cloud distances, e.g. N-H 0.86 A), the force field
    expects nuclear distances (N-H 1.01 A). In the force field, each hydrogen
    is moved along the X-H bond to the equilibrium distance of the force field:
    H_mm = X + scale*(H - X), scale = r0(force field)/distance_ideal(cctbx).
    """
    openmm = import_openmm()
    atoms = list(s.topology.atoms())
    h_local, x_local, scale = flex.size_t(), flex.size_t(), flex.double()
    for force in s.system.getForces():
      if (not isinstance(force, openmm.HarmonicBondForce)): continue
      for k in range(force.getNumBonds()):
        i, j, r0, kb = force.getBondParameters(k)
        if (atoms[j].element.symbol == "H"): i, j = j, i
        if (atoms[i].element.symbol != "H"): continue
        d = self.distance_ideal.get((s.i_seqs[i], s.i_seqs[j]))
        if (d is None or d <= 0): continue
        h_local.append(i)
        x_local.append(j)
        scale.append(r0.value_in_unit(openmm.unit.angstrom) / d)
    s.h_local, s.x_local, s.h_scale = h_local, x_local, scale

  def _rg_id(self, rg):
    return "%s %s %s" % (rg.atom_groups()[0].resname, rg.parent().id.strip(),
      rg.resid().strip())

  def _topology(self, residue_groups, altloc):
    """OpenMM topology of the conformer with the altloc (residue groups
    without it: their first altloc). Returns the topology, the i_seqs of
    its atoms and the residue group of each residue."""
    app = import_openmm().app
    topology = app.Topology()
    i_seqs = flex.size_t()
    rgs = []
    index = {}
    chain = None
    previous_chain = None
    for rg in residue_groups:
      if (rg.parent() is not previous_chain):
        chain = topology.addChain(rg.parent().id)
        previous_chain = rg.parent()
      ags = rg.atom_groups()
      alts = [ag.altloc for ag in ags if ag.altloc != ""]
      use = altloc if altloc in alts else (alts[0] if len(alts) else "")
      atoms = []
      resname = None
      for ag in ags:
        if (ag.altloc in ["", use]):
          atoms.extend(ag.atoms())
          if (resname is None or ag.altloc != ""): resname = ag.resname
      residue = topology.addResidue(resname.strip(), chain,
        id = rg.resseq.strip(), insertionCode = rg.icode.strip())
      rgs.append(rg)
      for atom in atoms:
        e = atom.element.strip().capitalize()
        if (e == "D"): e = "H"
        element = app.element.Element.getBySymbol(e)
        a = topology.addAtom(atom.name.strip(), element, residue)
        index[atom.i_seq] = a
        i_seqs.append(atom.i_seq)
    added = set()
    for i_seq, a in index.items():
      for j_seq in self.bonded.get(i_seq, []):
        if (j_seq in index and (j_seq, i_seq) not in added):
          topology.addBond(a, index[j_seq])
          added.add((i_seq, j_seq))
    return topology, i_seqs, rgs

  def show(self):
    log = self.log
    n_ff = self.iselection.size()
    files = list(self.files)
    if (self.solvent is not None):
      files.append("implicit/%s.xml" % self.solvent)
    print("Force field (OpenMM %s): %s; %d residues, %d atoms%s" % (
      import_openmm().__version__, " ".join(files), self.n_residues,
      n_ff, ("; %d conformers (altlocs %s)" % (len(self.systems),
      " ".join(self.altlocs))) if len(self.systems) > 1 else ""), file=log)
    if (len(self.templates.n_derived) > 0):
      print("  templates derived for incomplete residues or chain breaks: "
        "%s" % ", ".join(["%s %d" % (n, c) for n, c in
        sorted(self.templates.n_derived.items())]), file=log)
    if (self.solvent_final_cycle):
      print("  implicit solvent (%s) in the final macro-cycle" %
        self.params.implicit_solvent, file=log)
    reasons = {}
    for rg_id, reason in self.excluded.items():
      reasons.setdefault(reason, []).append(rg_id)
    for reason, rg_ids in sorted(reasons.items()):
      print("  geometry restraints for %d residues (%s)%s" % (len(rg_ids),
        reason, (": " + ", ".join(rg_ids[:10]) + (", ..." if len(rg_ids) > 10
        else "")) if reason != "no hydrogens" else ""), file=log)

  def _mm_positions(self, s, sites_cart):
    """Coordinates (A) of the atoms of the system, with the hydrogens at the
    force-field X-H distances."""
    xyz = sites_cart.select(s.i_seqs)
    x = xyz.select(s.x_local)
    xyz.set_selected(s.h_local, x + (xyz.select(s.h_local) - x) * s.h_scale)
    return xyz

  def minimize(self, sites_cart, hydrogens_only, sigma=None,
               max_iterations=500):
    """
    Minimizes the force-field energy with respect to the hydrogens (heavy
    atoms fixed), or to all atoms of the force field with harmonic restraints
    of the heavy atoms to their positions (sigma, A). Atoms in several
    conformers are moved in the first one and fixed in the others. Returns the
    new sites (hydrogens at the X-H distances of the geometry restraints) and
    the energies (kJ/mol) before and after.
    """
    openmm = import_openmm()
    unit = openmm.unit
    result = sites_cart.deep_copy()
    done = flex.bool(sites_cart.size(), False)
    e_start = self.energy_and_gradients(sites_cart, False)[0]
    for s in self.systems:
      system = openmm.XmlSerializer.deserialize(
        openmm.XmlSerializer.serialize(s.system))
      atoms = list(s.topology.atoms())
      xyz = self._mm_positions(s, result)
      if (not hydrogens_only):
        # E = kT/2 * (|r - r0|/sigma)**2
        restraints = openmm.CustomExternalForce(
          "k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
        restraints.addGlobalParameter("k", 0.5 * self.kt / (sigma*0.1)**2)
        for name in ["x0", "y0", "z0"]:
          restraints.addPerParticleParameter(name)
      for i, atom in enumerate(atoms):
        if (done[s.i_seqs[i]] or
            (hydrogens_only and atom.element.symbol != "H")):
          system.setParticleMass(i, 0) # fixed
        elif (not hydrogens_only and atom.element.symbol != "H"):
          restraints.addParticle(i, [c * 0.1 for c in xyz[i]])
      if (not hydrogens_only):
        system.addForce(restraints)
      context = openmm.Context(system, openmm.VerletIntegrator(0.001),
        s.context.getPlatform())
      context.setPositions(xyz.as_double().as_numpy_array().reshape(-1, 3)
        * 0.1)
      openmm.LocalEnergyMinimizer.minimize(context, 1.0, max_iterations)
      x = context.getState(getPositions = True).getPositions(asNumpy = True)
      xyz = flex.vec3_double(flex.double(
        x.value_in_unit(unit.angstrom).ravel()))
      # back to the X-H distances of the geometry restraints
      p = xyz.select(s.x_local)
      xyz.set_selected(s.h_local, p + (xyz.select(s.h_local) - p) /
        s.h_scale)
      for i, atom in enumerate(atoms):
        i_seq = s.i_seqs[i]
        if (not done[i_seq] and
            (not hydrogens_only or atom.element.symbol == "H")):
          result[i_seq] = xyz[i]
          done[i_seq] = True
    e_final = self.energy_and_gradients(result, False)[0]
    return result, e_start, e_final

  def energy_and_gradients(self, sites_cart, compute_gradients=True):
    """E_MM (kJ/mol, mean over the conformers) and its gradients (kJ/mol/A;
    zero for the other atoms)."""
    openmm = import_openmm()
    unit = openmm.unit
    energy = 0
    gradients = None
    if (compute_gradients):
      gradients = flex.vec3_double(sites_cart.size(), (0,0,0))
    w = 1. / len(self.systems)
    for s in self.systems:
      xyz = self._mm_positions(s, sites_cart)
      s.context.setPositions(
        xyz.as_double().as_numpy_array().reshape(-1, 3) * 0.1)
      state = s.context.getState(getEnergy = True,
        getForces = compute_gradients)
      energy += w * state.getPotentialEnergy().value_in_unit(
        unit.kilojoule_per_mole)
      if (compute_gradients):
        f = state.getForces(asNumpy = True).value_in_unit(
          unit.kilojoule_per_mole / unit.nanometer)
        # dE/dx (kJ/mol/A) = -F (kJ/mol/nm) / 10
        g = flex.vec3_double(flex.double(f.ravel() * (-0.1 * w)))
        g_h = g.select(s.h_local)
        g.set_selected(s.h_local, g_h * s.h_scale)
        g.add_selected(s.x_local, g_h * (1 - s.h_scale))
        gradients.add_selected(s.i_seqs, g)
    return energy, gradients

# -----------------------------------------------------------------------------

class manager(mmtbx.restraints.manager):
  """
  Restraints manager with the force field in place of the geometry restraints
  of the force-field atoms (energies_sites); otherwise as
  mmtbx.restraints.manager. result.geometry of energies_sites() are the
  geometry restraints of all atoms (for the rmsd(bonds) etc. reports), and
  result.mm_energy the force-field energy (kJ/mol). With
  use_force_field=False, the geometry restraints of all atoms are used
  (e.g. for the X-ray weight).
  """
  def __init__(self, restraints_manager, force_field):
    mmtbx.restraints.manager.__init__(self,
      geometry              = restraints_manager.geometry,
      cartesian_ncs_manager = restraints_manager.cartesian_ncs_manager,
      normalization         = restraints_manager.normalization)
    self.force_field = force_field
    self.use_force_field = True

  def select(self, selection):
    # (selected copies, e.g. for the ADP weights, use the geometry restraints)
    return mmtbx.restraints.manager(geometry = self.geometry,
      cartesian_ncs_manager = self.cartesian_ncs_manager,
      normalization = self.normalization).select(selection)

  def energies_sites(self,
        sites_cart,
        geometry_flags=None,
        external_energy_function=None,
        custom_nonbonded_function=None,
        compute_gradients=False,
        gradients=None,
        force_restraints_model=False,
        disable_asu_cache=False,
        hd_selection=None,
        compute_curvatures=False):
    if (not self.use_force_field):
      return mmtbx.restraints.manager.energies_sites(self,
        sites_cart=sites_cart,
        geometry_flags=geometry_flags,
        external_energy_function=external_energy_function,
        custom_nonbonded_function=custom_nonbonded_function,
        compute_gradients=compute_gradients,
        gradients=gradients,
        disable_asu_cache=disable_asu_cache,
        hd_selection=hd_selection,
        compute_curvatures=compute_curvatures)
    assert external_energy_function is None
    assert custom_nonbonded_function is None
    ff = self.force_field
    result = scitbx.restraints.energies(
      compute_gradients=compute_gradients,
      gradients=gradients,
      gradients_size=sites_cart.size(),
      gradients_factory=flex.vec3_double,
      normalization=self.normalization)
    # (no curvatures: result.curvatures is None)
    result.geometry = self.geometry.energies_sites(
      sites_cart=sites_cart,
      flags=geometry_flags,
      compute_gradients=compute_gradients,
      gradients=result.gradients,
      disable_asu_cache=disable_asu_cache,
      normalization=False)
    result += result.geometry
    # restraints replaced by the force field
    e_ff = ff.geometry_ff.energies_sites(
      sites_cart=sites_cart.select(ff.iselection),
      flags=ff.flags_ff,
      compute_gradients=compute_gradients,
      normalization=False)
    result.residual_sum -= e_ff.residual_sum
    mm_energy, mm_gradients = ff.energy_and_gradients(sites_cart,
      compute_gradients=compute_gradients)
    scale = 2 / ff.kt
    result.residual_sum += mm_energy * scale
    result.mm_energy = mm_energy
    if (compute_gradients):
      result.gradients.add_selected(ff.iselection, e_ff.gradients * -1)
      result.gradients += mm_gradients * scale
    if (self.cartesian_ncs_manager is None):
      result.cartesian_ncs_manager = None
    else:
      result.cartesian_ncs_manager = self.cartesian_ncs_manager.energies_sites(
        sites_cart=sites_cart,
        compute_gradients=compute_gradients,
        gradients=result.gradients,
        normalization=False)
      result += result.cartesian_ncs_manager
    result.finalize_target_and_gradients()
    return result

def setup(model, params, log):
  """Replaces the restraints manager of the model by the force-field
  manager, after optimizing the hydrogens (optimize_hydrogens) and the
  restrained minimization (start_minimization_sigma); returns the
  force_field."""
  ff = force_field(model = model, params = params, log = log)
  ff.show()
  def update_riding(sites_cart):
    model.set_sites_cart(sites_cart)
    if (model.riding_h_manager is not None):
      # new riding parameterization (dihedrals of the rotatable hydrogens)
      model.riding_h_manager = None
      model.setup_riding_h_manager(idealize = True)
  if (params.optimize_hydrogens):
    sites_cart, e_start, e_final = ff.minimize(model.get_sites_cart(),
      hydrogens_only = True)
    update_riding(sites_cart)
    # (after the idealization of the riding hydrogens)
    e_final = ff.energy_and_gradients(model.get_sites_cart(), False)[0]
    print("  hydrogens optimized against the force field: E_mm %.0f -> %.0f "
      "kJ/mol" % (e_start, e_final), file=log)
  if (params.start_minimization_sigma > 0):
    sites_start = model.get_sites_cart()
    sites_cart, e_start, e_final = ff.minimize(sites_start,
      hydrogens_only = False, sigma = params.start_minimization_sigma)
    update_riding(sites_cart)
    e_final = ff.energy_and_gradients(model.get_sites_cart(), False)[0]
    heavy = ff.selection & ~model.get_hd_selection()
    d = flex.sqrt((model.get_sites_cart() - sites_start).select(heavy).dot())
    print("  restrained minimization (sigma=%g A): E_mm %.0f -> %.0f kJ/mol, "
      "shifts rms %.3f max %.2f A" % (params.start_minimization_sigma,
      e_start, e_final, flex.mean_sq(d)**0.5, flex.max(d)), file=log)
  model.restraints_manager = manager(
    restraints_manager = model.get_restraints_manager(),
    force_field        = ff)
  return ff
