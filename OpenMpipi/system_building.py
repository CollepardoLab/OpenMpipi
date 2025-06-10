import numpy as np

import openmm as mm
import openmm.app as app
import openmm.unit as unit

from scipy.spatial import KDTree
import os
import copy

from .constants import DATA_DIR

def get_harmonic_bonds(positions, topology, globular_indices_dict, IDR_k=8031.):
    """
    Generates a list of harmonic bonds for a given topology. The function handles nucleosome chains, DNA chains,
    and other biomolecules and assigns bonds between atoms based on their positions and chain type.

    Args:
        positions (ndarray): An array of atomic positions.
        topology (Topology): An OpenMM topology object containing the chains and atoms of the system.
        globular_indices_dict (dict): A dictionary mapping chain IDs to lists of globular domain indices.
        dyad_positions (list): A list of dyad positions for nucleosomes.
        constraints (str, optional): Type of nucleosome-DNA constraint ('none', 'dyad', 'inner', 'breathing', 'all'). Default is 'inner'.
        IDR_k (float, optional): The force constant for harmonic bonds in IDRs. Default is 8031.
    
    Returns:
        list: A list of harmonic bonds, where each bond is represented as a tuple (atom1, atom2, distance, force constant).
    """
    
    bonds = []

    all_atoms = list(topology.atoms())  # Fetch all atoms in topology once
    atom_tree = KDTree(positions)

    chains = list(topology.chains())  # Fetch chains only once

    # Iterate over all chains in topology
    for i, chain in enumerate(chains):
        chain_atoms = list(chain.atoms())

        if 'r' in chain_atoms[0].name:
            IDR_d = 0.500
        else:
            IDR_d = 0.381
        
        chain_id = chain.id
        globular_indices_list = globular_indices_dict.get(chain_id, [])  # Use default empty list if not found

        # Identify IDR indices
        all_globular_indices = [i for domain in globular_indices_list for i in domain]
        IDR_indices = [i for i in range(len(chain_atoms)) if i not in all_globular_indices]

        # Add bonds for IDR regions
        for i in range(len(chain_atoms) - 1):
            if i in IDR_indices or i + 1 in IDR_indices:
                bonds.append((chain_atoms[i], chain_atoms[i + 1], IDR_d, IDR_k))

        # Add ENM bonds for globular regions
        for globular_indices in globular_indices_list:
            ENM_atoms = [chain_atoms[i] for i in globular_indices]
            ENM_bonds = get_ENM_bonds(atom_tree, ENM_atoms, all_atoms)
            bonds.extend(ENM_bonds)

    return bonds

def get_ENM_bonds(atom_tree, ENM_atoms, all_atoms, cutoff=0.75, k=8031.):
    """
    Generates a list of Elastic Network Model (ENM) bonds for a given set of atoms based on a distance cutoff.

    Args:
        atom_tree (KDTree): KDTree of atomic positions for efficient neighbor lookup.
        ENM_atoms (list): List of atoms to consider for ENM bonds.
        all_atoms (list): List of all atoms in the system.
        cutoff (float, optional): Distance cutoff for considering a bond (in nm). Default is 0.75 nm.
        k (float, optional): Force constant for the ENM bonds. Default is 8031.
    
    Returns:
        list: A list of ENM bonds, where each bond is represented as a tuple (atom1, atom2, distance, force constant).
    """
    
    bonds = []
    num_atoms = len(ENM_atoms)
    
    # Precompute positions of ENM atoms from the KDTree
    ENM_positions = {atom.index: atom_tree.data[atom.index] for atom in ENM_atoms}
    
    # Iterate through each atom in ENM_atoms
    for i in range(num_atoms):
        atom1 = ENM_atoms[i]
        atom1_pos = ENM_positions[atom1.index]
        
        # Query nearby atoms within the cutoff distance
        nearby_indices = atom_tree.query_ball_point(atom1_pos, cutoff)
        
        # Check each nearby atom and form a bond
        for j in nearby_indices:
            if j != atom1.index and j in ENM_positions.keys():  # Exclude self
                atom2 = all_atoms[j]
                atom2_pos =  ENM_positions[atom2.index]
             
                r = np.linalg.norm(atom1_pos - atom2_pos)
                bonds.append((atom1, atom2, r, k))
    
    return bonds

def calculate_debye_length(T, csx):
    """
    Calculate the Debye length based on the temperature and ionic strength.

    This function calculates the Debye length based on the temperature and ionic strength of the system.
    The Debye length is used to screen electrostatic interactions in the system.

    Args:
        T (float): The temperature of the system in Kelvin.
        csx (float): The ionic strength of the system in mM.

    Returns:
        float: The Debye length in nanometers.
    """
    # Electrolyte solution ionic strength in m^-3
    cs = (csx / 1000) * 6.022e26
    # Relative dielectric constant of the medium
    er = 5321 / T + 233.76 - 0.9297 * T + 0.001417 * T**2 - 0.0000008292 * T**3
    # Bjerrum length in meters
    bjerrum = (1.671e-5) / (er * T)
    # Debye length in nanometers
    debye_length = np.sqrt(1 / (8 * np.pi * bjerrum * cs)) * 1e9

    return debye_length
    
def get_mpipi_system(positions, topology, globular_indices_dict, T, csx, CM_remover=True, periodic=True):
    
    NB_PARAMETERS = np.loadtxt(os.path.join(DATA_DIR, 'recharged_params.txt'))
    mapping_dict = {
                    'pM':[131.20, 0], 'pG':[57.05, 1],  'pK':[128.20,2],  'pT':[101.10,3], 
                    'pR':[156.20, 4], 'pA':[71.08, 5],  'pD':[115.10,6],  'pE':[129.10,7],
                    'pY':[163.20, 8], 'pV':[99.07, 9],  'pL':[113.20,10], 'pQ':[128.10,11],
                    'pW':[186.20, 12],'pF':[147.20,13], 'pS':[87.08, 14], 'pH':[137.10,15],
                    'pN':[114.10, 16],'pP':[97.12, 17], 'pC':[103.10,18], 'pI':[113.20,19], # AMINO ACIDS
                    'rU':[244.20, 20]                                                       # RNA NUCLEOTIDES
                   }
    
    debye_length = calculate_debye_length(T, csx)
    system = mm.System()
    for atom in topology.atoms():
        system.addParticle(mapping_dict[atom.name][0])

        
    harm_bonds = get_harmonic_bonds(positions, topology, globular_indices_dict) 
    
    bond_flag = True
    if len(list(topology.bonds())) > 0:
        bond_flag = False
    
    harm_potential = mm.HarmonicBondForce()
    for bond in harm_bonds:
        a1, a2, d, k = bond  
        
        harm_potential.addBond(a1.index, a2.index, d, k)
        if bond_flag == True:
            topology.addBond(a1, a2)
    

    system.addForce(harm_potential)
    
    wf_string = '''
    glob_factor * step(rc-r)*epsilon*alpha*((sigma/r)^(2*mu)-1)*((rc/r)^(2*mu)-1)^2;
    alpha = 2*(3^(2*mu))*((3)/(2*((3^(2*mu))-1)))^3;
    rc = 3*sigma;

    glob_factor = select(globular1*globular2, 0.7, select(globular1+globular2, sqrt(0.7), 1));
    
    epsilon = wf_table(index1, index2, 0);
    sigma   = wf_table(index1, index2, 1);
    mu =  floor(wf_table(index1, index2, 2));
    '''
    yukawa_string = '''
    (A_table(index1, index2)/r) * exp(-kappa*r);
    '''
    
    wf_potential = mm.CustomNonbondedForce(wf_string)
    yukawa_potential = mm.CustomNonbondedForce(yukawa_string)
    wf_potential.addPerParticleParameter('index')
    wf_potential.addPerParticleParameter('globular')
    
    yukawa_potential.addPerParticleParameter('index')
    yukawa_potential.addGlobalParameter('kappa', 1/debye_length)
    
    if periodic == True:
        wf_potential.setNonbondedMethod(mm.CustomNonbondedForce.CutoffPeriodic)
        yukawa_potential.setNonbondedMethod(mm.CustomNonbondedForce.CutoffPeriodic)
        
    else:
        wf_potential.setNonbondedMethod(mm.CustomNonbondedForce.CutoffNonPeriodic)
        yukawa_potential.setNonbondedMethod(mm.CustomNonbondedForce.CutoffNonPeriodic)
    wf_potential.setCutoffDistance(2.5*unit.nanometer)                        
    
    yukawa_potential.setCutoffDistance(3.5*unit.nanometer)
    yukawa_potential.setForceGroup(1) # to use different cutoff, have to be in different ForceGroup
    
    for chain in topology.chains():
        atoms = list(chain.atoms())
        if 'nuc' in chain.id: 
            
            for i, atom in enumerate(atoms):
                globular = 1 if atom.element.symbol == 'Pt' else 0
                index = mapping_dict[atom.name][1]
             
                wf_potential.addParticle([index, globular])
                yukawa_potential.addParticle([index])
        
        else:
            for i, atom in enumerate(atoms):
                globular = 1 if atom.element.symbol == 'Pt' else 0 
                index = mapping_dict[atom.name][1]
                
                wf_potential.addParticle([index, globular]) 
                yukawa_potential.addParticle([index])
    
                
    wf_potential.createExclusionsFromBonds([(bond[0].index, bond[1].index) for bond in topology.bonds()], 1)
    yukawa_potential.createExclusionsFromBonds([(bond[0].index, bond[1].index) for bond in topology.bonds()], 1)
    
    wf_table = mm.Discrete3DFunction(21, 21, 3, NB_PARAMETERS[:-21*21])
    wf_potential.addTabulatedFunction('wf_table', wf_table)
    
    yukawa_table = mm.Discrete2DFunction(21, 21, NB_PARAMETERS[-21*21:])
    yukawa_potential.addTabulatedFunction('A_table', yukawa_table)
    
    system.addForce(wf_potential)
    system.addForce(yukawa_potential)
    
    if CM_remover == True: 
        system.addForce(mm.CMMotionRemover(1000))
    
    ranges = np.ptp(positions, axis=0)
    max_range_value = ranges[np.argmax(ranges)]
    box_length = max_range_value + 50
    box_vecs = [mm.Vec3(x=box_length, y=0.0, z=0.0), mm.Vec3(x=0.0, y=box_length, z=0.0), mm.Vec3(x=0.0, y=0.0, z=box_length)]*unit.nanometer
    system.setDefaultPeriodicBoxVectors(*box_vecs)
    
    return system
