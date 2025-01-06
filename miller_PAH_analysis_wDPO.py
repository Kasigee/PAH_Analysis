import os
import time
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdDetermineBonds
from multiprocessing import Pool, cpu_count
import networkx as nx
#import matplotlib
#matplotlib.use('Agg')  # Use the 'Agg' backend which is safe for multiprocessing
import matplotlib.pyplot as plt
from collections import deque
import re
from scipy.spatial import ConvexHull

AngleThreshold=25

def extract_energy1(log_file):
    with open(log_file, 'r') as file:
        lines = file.readlines()

    last_scf_line = None
    for line in lines:
        if 'SCF Done' in line:
            last_scf_line = line

    if last_scf_line:
        energy = float(last_scf_line.split()[4])
        energy_kcal_mol = energy * 627.5095
        return energy, energy_kcal_mol

    return None, None
def extract_energy2(log_file):
    with open(log_file, 'r') as file:
        lines = file.readlines()

    last_scf_line = None
    homo_energy = None
    lumo_energy = None
    occ_energies = []
    virt_energies = []
    band_gap_eV = None

    for line in lines:
        # Find SCF Done line (to get final SCF energy)
        if 'SCF Done' in line:
            last_scf_line = line

        # Find occupied molecular orbital energies (HOMO)
        if 'Alpha  occ. eigenvalues' in line or 'Beta  occ. eigenvalues' in line:
            occ_energies.extend([float(x) for x in line.split()[4:]])

        # Find virtual molecular orbital energies (LUMO)
        if 'Alpha virt. eigenvalues' in line or 'Beta virt. eigenvalues' in line:
            virt_energies.extend([float(x) for x in line.split()[4:]])

    # Extract HOMO and LUMO
    if occ_energies:
        homo_energy = max(occ_energies)  # Highest occupied molecular orbital
    if virt_energies:
        lumo_energy = min(virt_energies)  # Lowest unoccupied molecular orbital

    # Calculate band gap in Hartree and convert to eV
    band_gap = None
    if homo_energy is not None and lumo_energy is not None:
        band_gap_hartree = lumo_energy - homo_energy
        band_gap_eV = band_gap_hartree * 27.2114  # Convert Hartree to eV
        band_gap = band_gap_eV

    # Get the last SCF energy in Hartree and convert to kcal/mol
    energy, energy_kcal_mol = None, None
    if last_scf_line:
        energy = float(last_scf_line.split()[4])
        energy_kcal_mol = energy * 627.5095

    return energy, energy_kcal_mol, homo_energy, lumo_energy, band_gap_eV
def extract_energy(log_file):
    with open(log_file, 'r') as file:
        lines = file.readlines()
    
    last_scf_line = None
    homo_energy = None
    lumo_energy = None
    occ_energies = []
    virt_energies = []
    mulliken_charges = []
    collecting_mulliken = False
    total_dipole = None
    electronic_spatial_extent = None
    degrees_of_freedom = None
    dipole_components = {}
    quadrupole_moment = {}
    traceless_quadrupole_moment = {}
    octapole_moment = {}
    hexadecapole_moment = {}

    for i, line in enumerate(lines):
        # Find SCF Done line (to get final SCF energy)
        if 'SCF Done' in line:
            last_scf_line = line

        # Find occupied molecular orbital energies (HOMO)
        if 'Alpha  occ. eigenvalues' in line or 'Beta  occ. eigenvalues' in line:
            occ_energies.extend([float(x) for x in line.split()[4:]])

        # Find virtual molecular orbital energies (LUMO)
        if 'Alpha virt. eigenvalues' in line or 'Beta virt. eigenvalues' in line:
            virt_energies.extend([float(x) for x in line.split()[4:]])

        # Collect Mulliken charges
        if 'Mulliken charges:' in line:
            collecting_mulliken = True
            mulliken_charges = []
            continue  # Go to next line

        if collecting_mulliken:
            if 'Sum of Mulliken charges' in line:
                collecting_mulliken = False
                continue
            else:
                # Skip empty lines or lines with less than 3 elements
                if len(line.strip()) == 0 or len(line.split()) < 3:
                    continue
                # Collect the charges
                parts = line.split()
                try:
                    charge = float(parts[2])
                    mulliken_charges.append(charge)
                except ValueError:
                    pass  # In case of parsing error, skip

        # Electronic spatial extent
        if 'Electronic spatial extent (au):  <R**2>=' in line:
            try:
                electronic_spatial_extent = float(line.split('=')[1].strip())
            except ValueError:
                pass

        # Dipole moment
        if 'Dipole moment' in line:
            # Next line contains the components
            dipole_line = lines[i+1]
            # Use regex to extract the values
            matches = re.findall(r'([XYZTot]+)=\s*([-+]?\d*\.\d+|\d+)', dipole_line)
            for key, value in matches:
                dipole_components[key] = float(value)
            total_dipole = dipole_components.get('Tot', None)
        
        # Degrees of freedom
        if 'Deg. of freedom' in line:
            try:
                degrees_of_freedom = int(line.split()[3])
            except (IndexError, ValueError):
                pass

        # Quadrupole moment
        if 'Quadrupole moment (field-independent basis, Debye-Ang):' in line:
            # Read next two lines
            qm_line1 = lines[i+1]
            qm_line2 = lines[i+2]
            # Parse the lines
            matches1 = re.findall(r'([XYZ]{2})=\s*([-+]?\d*\.\d+|\d+)', qm_line1)
            matches2 = re.findall(r'([XYZ]{2})=\s*([-+]?\d*\.\d+|\d+)', qm_line2)
            for key, value in matches1 + matches2:
                quadrupole_moment[key] = float(value)

        # Traceless Quadrupole moment
        if 'Traceless Quadrupole moment (field-independent basis, Debye-Ang):' in line:
            # Read next two lines
            tqm_line1 = lines[i+1]
            tqm_line2 = lines[i+2]
            # Parse the lines
            matches1 = re.findall(r'([XYZ]{2})=\s*([-+]?\d*\.\d+|\d+)', tqm_line1)
            matches2 = re.findall(r'([XYZ]{2})=\s*([-+]?\d*\.\d+|\d+)', tqm_line2)
            for key, value in matches1 + matches2:
                traceless_quadrupole_moment[key] = float(value)

        # Octapole moment
        if 'Octapole moment (field-independent basis, Debye-Ang**2):' in line:
            # Read next two lines
            oct_line1 = lines[i+1]
            oct_line2 = lines[i+2]
            matches1 = re.findall(r'([XYZ]{3})=\s*([-+]?\d*\.\d+|\d+)', oct_line1)
            matches2 = re.findall(r'([XYZ]{3})=\s*([-+]?\d*\.\d+|\d+)', oct_line2)
            for key, value in matches1 + matches2:
                octapole_moment[key] = float(value)

        # Hexadecapole moment
        if 'Hexadecapole moment (field-independent basis, Debye-Ang**3):' in line:
            # Read next three lines
            hdp_line1 = lines[i+1]
            hdp_line2 = lines[i+2]
            hdp_line3 = lines[i+3]
            matches1 = re.findall(r'([XYZ]{4})=\s*([-+]?\d*\.\d+|\d+)', hdp_line1)
            matches2 = re.findall(r'([XYZ]{4})=\s*([-+]?\d*\.\d+|\d+)', hdp_line2)
            matches3 = re.findall(r'([XYZ]{4})=\s*([-+]?\d*\.\d+|\d+)', hdp_line3)
            for key, value in matches1 + matches2 + matches3:
                hexadecapole_moment[key] = float(value)

    # Extract HOMO and LUMO energies
    if occ_energies:
        homo_energy = max(occ_energies)
    if virt_energies:
        lumo_energy = min(virt_energies)

    # Calculate band gap in eV
    band_gap_eV = None
    if homo_energy is not None and lumo_energy is not None:
        band_gap_hartree = lumo_energy - homo_energy
        band_gap_eV = band_gap_hartree * 27.2114  # Hartree to eV

    # Get the last SCF energy in Hartree and convert to kcal/mol
    energy, energy_kcal_mol = None, None
    if last_scf_line:
        energy = float(last_scf_line.split()[4])
        energy_kcal_mol = energy * 627.5095

    # Get max and min Mulliken charges
    max_mulliken_charge, min_mulliken_charge = None, None
    if mulliken_charges:
        max_mulliken_charge = max(mulliken_charges)
        min_mulliken_charge = min(mulliken_charges)

    # result = {
    #     'energy': energy,
    #     'energy_kcal_mol': energy_kcal_mol,
    #     'homo_energy': homo_energy,
    #     'lumo_energy': lumo_energy,
    #     'band_gap_eV': band_gap_eV,
    #     'total_dipole': total_dipole,
    #     'electronic_spatial_extent': electronic_spatial_extent,
    #     'max_mulliken_charge': max_mulliken_charge,
    #     'min_mulliken_charge': min_mulliken_charge,
    #     'degrees_of_freedom': degrees_of_freedom,
    #     'quadrupole_moment': quadrupole_moment,
    #     'traceless_quadrupole_moment': traceless_quadrupole_moment,
    #     'octapole_moment': octapole_moment,
    #     'hexadecapole_moment': hexadecapole_moment
    # }

    # return result
    return energy, energy_kcal_mol, homo_energy, lumo_energy, band_gap_eV, total_dipole, electronic_spatial_extent, max_mulliken_charge, min_mulliken_charge, degrees_of_freedom, quadrupole_moment, traceless_quadrupole_moment, octapole_moment, hexadecapole_moment
def calculate_dihedral(p1, p2, p3, p4):
    b1 = p2 - p1
    b2 = p3 - p2
    b3 = p4 - p3

    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)

    n1 /= np.linalg.norm(n1)
    n2 /= np.linalg.norm(n2)

    cos_theta = np.dot(n1, n2)
    theta = np.arccos(cos_theta)

    dihedral_angle = np.degrees(theta)

    return dihedral_angle
def calculate_bond_angle(p1, p2, p3):
    v1 = p1 - p2
    v2 = p3 - p2

    v1 /= np.linalg.norm(v1)
    v2 /= np.linalg.norm(v2)

    cos_theta = np.dot(v1, v2)
    theta = np.arccos(cos_theta)

    bond_angle = np.degrees(theta)

    return bond_angle
def find_bonded_carbons(mol):
    bonded_atoms = []
    bonds = mol.GetBonds()

    for bond in bonds:
        atom1 = bond.GetBeginAtomIdx()
        atom2 = bond.GetEndAtomIdx()
        if mol.GetAtomWithIdx(atom1).GetAtomicNum() == 6 and mol.GetAtomWithIdx(atom2).GetAtomicNum() == 6:
            bonded_atoms.append((atom1, atom2))

    return bonded_atoms
def find_dihedrals_and_bond_angles(mol, bonded_carbons):
    bonded_carbons_dict = {a: [] for pair in bonded_carbons for a in pair}

    for a1, a2 in bonded_carbons:
        bonded_carbons_dict[a1].append(a2)
        bonded_carbons_dict[a2].append(a1)

    dihedrals = []
    bond_angles = []
    unique_dihedrals = set()
    sum_less_90 = sum_greater_90 = sum_abs_120_minus_angle = 0
    count_less_90 = count_greater_90 = count_angles = 0

    for a1 in bonded_carbons_dict:
        for a2 in bonded_carbons_dict[a1]:
            for a3 in bonded_carbons_dict[a2]:
                if a3 != a1:
                    # Bond angles
                    p1 = mol.GetConformer().GetAtomPosition(a1)
                    p2 = mol.GetConformer().GetAtomPosition(a2)
                    p3 = mol.GetConformer().GetAtomPosition(a3)
                    bond_angle = calculate_bond_angle(p1, p2, p3)
                    bond_angles.append((a1, a2, a3, bond_angle))
                    sum_abs_120_minus_angle += abs(120 - bond_angle)
                    count_angles += 1

                    for a4 in bonded_carbons_dict[a3]:
                        if a4 != a2 and a4 != a1:
                            dihedral_tuple = (a1, a2, a3, a4)
                            reversed_dihedral_tuple = (a4, a3, a2, a1)

                            if dihedral_tuple not in unique_dihedrals and reversed_dihedral_tuple not in unique_dihedrals:
                                unique_dihedrals.add(dihedral_tuple)

                                p4 = mol.GetConformer().GetAtomPosition(a4)
                                dihedral_angle = calculate_dihedral(p1, p2, p3, p4)
                                dihedrals.append((a1, a2, a3, a4, dihedral_angle))

                                if dihedral_angle < 90:
                                    sum_less_90 += dihedral_angle
                                    count_less_90 += 1
                                else:
                                    sum_greater_90 += 180 - dihedral_angle
                                    count_greater_90 += 1

    return dihedrals, bond_angles, sum_less_90, sum_greater_90, count_less_90, count_greater_90, sum_abs_120_minus_angle, count_angles
def find_hydrogen_distances(mol, cutoff=2.5):
    hydrogen_atoms = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() == 1]
    total_distance = total_distance2 = 0.0
    countH_under2_5 = 0

    for i in range(len(hydrogen_atoms)):
        for j in range(i + 1, len(hydrogen_atoms)):
            h1 = mol.GetConformer().GetAtomPosition(hydrogen_atoms[i])
            h2 = mol.GetConformer().GetAtomPosition(hydrogen_atoms[j])
            dist = np.linalg.norm(np.array(h1) - np.array(h2))
            total_distance += dist
            if dist <= cutoff:
                total_distance2 += dist
                countH_under2_5 += 1

    return total_distance, total_distance2, countH_under2_5
def calculate_rmsd_bond_lengths(mol, bonded_carbons):
    distances = [np.linalg.norm(np.array(mol.GetConformer().GetAtomPosition(a1)) - np.array(mol.GetConformer().GetAtomPosition(a2))) for a1, a2 in bonded_carbons]

    if not distances:
        return float('nan'), float('nan')

    mean_distance = np.mean(distances)
    rmsd = np.sqrt(np.mean((np.array(distances) - mean_distance)**2))

    return mean_distance, rmsd
def detect_aromatic_rings(mol):
    """
    Detect aromatic rings in the molecule.
    
    Parameters:
    mol (RDKit Mol object): The molecule in which aromatic rings are to be detected.

    Returns:
    list: List of aromatic rings where each ring is a list of atom indices.
    """
    ri = mol.GetRingInfo()
    aromatic_rings = [ring for ring in ri.AtomRings() if all(mol.GetAtomWithIdx(atom).GetIsAromatic() for atom in ring)]
    return aromatic_rings
def calculate_dpo_OLD(mol):
   """
   Calculate the Degree of π-Orbital Overlap (DPO) for a PAH structure.
   
   Parameters:
   mol (RDKit Mol object): The molecule for which DPO is to be calculated.

   Returns:
   float: DPO value for the molecule.
   """

    # Step 1: Detect fused rings and identify the reference segment
   aromatic_rings = detect_aromatic_rings(mol)
    
   if not aromatic_rings:
       print("No aromatic rings found in the molecule.")
       return None

   #Assign initial DPO parameters
   a = 0.05
   b = -1 / 4
   c = 1 / 3
   d = 1 / 3


    # Step 1: Determine the reference segment
   ring_info = mol.GetRingInfo()
   atom_rings = ring_info.AtomRings()
   segments = identify_segments(mol, atom_rings)
   print("atom rings =", atom_rings)
   print("segments =", segments)
    # Apply rules to select the reference segment
    
    # Apply Rule (a): Select the segment with the largest number of fused rings
   max_length = max(len(seg) for seg in segments)
   candidate_segments = [seg for seg in segments if len(seg) == max_length]

   if len(candidate_segments) == 1:
       reference_segment = candidate_segments[0]
   else:
       # Apply Rule (b) and (c) as needed
       reference_segment = select_reference_segment(mol, candidate_segments)
   print("reference_segment =",reference_segment)

    # Step 2: Assign DPO values to fused bonds
   DPO_value = assign_DPO_values(mol, segments, a, b, c, d)

   return DPO_value
def calculate_dpo_2(mol, a=0.05, b=-1/4, c=1/3, d=1/3):  # Add default values for a, b, c, d
    aromatic_rings = detect_aromatic_rings(mol)
    if not aromatic_rings:
        print("No aromatic rings found in the molecule.")
        return None

    G, ring_centers = build_ring_graph(mol, aromatic_rings)
    linear_segment = find_longest_linear_path(G, ring_centers)
    
    # print("linear_segment =",linear_segment)
    angles = calculate_angles_to_linear_segment(G, ring_centers, linear_segment)
    
    # visualize_graph(G, ring_centers, angles)
    visualize_graph_with_edge_angles(G, ring_centers, angles, linear_segment)
    
    reference_segment = find_longest_linear_path(G, ring_centers, angle_threshold=15)
    
    print("Reference Segment:", reference_segment)
    
    DPO_total, assigned_bonds = assign_dpo_to_reference_segment(mol, reference_segment, a)
    DPO_total += assign_dpo_to_angulated_segments(mol, G, reference_segment, ring_centers, a, b, c, d, assigned_bonds)
    
    return DPO_total
def calculate_dpo_3(mol, a=0.05, b=-1/4, c=1/3, d=1/3):
    aromatic_rings = detect_aromatic_rings(mol)
    if not aromatic_rings:
        print("No aromatic rings found in the molecule.")
        return None

    G, ring_centers = build_ring_graph(mol, aromatic_rings)
    linear_segment = find_longest_linear_path(G, ring_centers)
    edge_info = calculate_angles_to_linear_segment(G, ring_centers, linear_segment)
    
    
    reference_segment = find_longest_linear_path(G, ring_centers, angle_threshold=angle_threshold)
    
    print("Reference Segment:", reference_segment)
    
    # DPO_total, assigned_bonds = assign_dpo_to_angulated_segments(mol, G, reference_segment, edge_info, a, b, c, d)
    DPO_total, assigned_bonds, edge_categories = assign_dpo_to_angulated_segments(mol, G, reference_segment, edge_info, a, b, c, d)
    # visualize_graph_with_edge_angles(G, ring_centers, {e: info['angle'] for e, info in edge_info.items()}, linear_segment)
    visualize_graph_with_edge_info(G, ring_centers, edge_info, edge_categories, linear_segment)
    
    return DPO_total
def calculate_dpo_4(mol, a=0.05, b=-1/4, c=1/3, d=1/3):
    aromatic_rings = detect_aromatic_rings(mol)
    if not aromatic_rings:
        print("No aromatic rings found in the molecule.")
        return None

    G, ring_centers = build_ring_graph(mol, aromatic_rings)
    linear_segment = find_longest_linear_path(G, ring_centers)
    edge_info = calculate_angles_to_linear_segment(G, ring_centers, linear_segment)
    
    reference_segment = find_longest_linear_path(G, ring_centers, angle_threshold=AngleThreshold)
    
    print("Reference Segment:", reference_segment)
    
    # Calculate DPO for reference segment
    reference_DPO, reference_assigned_bonds = assign_dpo_to_reference_segment(mol, reference_segment, a)
    
    # Calculate DPO for angulated segments
    angulated_DPO, angulated_assigned_bonds, edge_categories = assign_dpo_to_angulated_segments(mol, G, reference_segment, edge_info, ring_centers, a, b, c, d)
    
    # Combine results
    DPO_total = reference_DPO + angulated_DPO
    assigned_bonds = reference_assigned_bonds.union(angulated_assigned_bonds)
    
    # Add reference segment information to edge_categories
    for i in range(len(reference_segment) - 1):
        edge = tuple(sorted((reference_segment[i], reference_segment[i+1])))
        if edge in edge_info:
            edge_categories[edge] = {
                'category': 'a',
                'dpo_value': 1 - i * a,
                'path': ['a'] * (i + 1),
                'angle': edge_info[edge]['angle']
            }
    
    visualize_graph_with_edge_info(G, ring_centers, edge_info, edge_categories, linear_segment, reference_segment)
    
    return DPO_total
def calculate_dpo5(mol, a=0.05, b=-1/4, c=1/3, d=1/3):
    aromatic_rings = detect_aromatic_rings(mol)
    if not aromatic_rings:
        print("No aromatic rings found in the molecule.")
        return None

    G, ring_centers = build_ring_graph(mol, aromatic_rings)
    
    reference_segment = find_reference_segment(G, ring_centers)
    
    print("Reference Segment:", reference_segment)
    
    edge_info = calculate_angles_to_linear_segment(G, ring_centers, reference_segment)
    
    # Calculate DPO for reference segment
    reference_DPO, reference_assigned_bonds = assign_dpo_to_reference_segment(mol, reference_segment, a)
    
    # Calculate DPO for angulated segments
    angulated_DPO, angulated_assigned_bonds, edge_categories = assign_dpo_to_angulated_segments(mol, G, reference_segment, edge_info, ring_centers, a, b, c, d)
    
    # Combine results
    DPO_total = reference_DPO + angulated_DPO
    assigned_bonds = reference_assigned_bonds.union(angulated_assigned_bonds)
    
    # Add reference segment information to edge_categories
    for i in range(len(reference_segment) - 1):
        edge = tuple(sorted((reference_segment[i], reference_segment[i+1])))
        if edge in edge_info:
            edge_categories[edge] = {
                'category': 'a',
                'dpo_value': 1 - i * a,
                'path': ['a'] * (i + 1),
                'angle': edge_info[edge]['angle']
            }
    
    visualize_graph_with_edge_info(G, ring_centers, edge_info, edge_categories, reference_segment, reference_segment)
    
    return DPO_total
def calculate_dpo6(mol, a=0.05, b=-1/4, c=1/3, d=1/3):
    aromatic_rings = detect_aromatic_rings(mol)
    if not aromatic_rings:
        print("No aromatic rings found in the molecule.")
        return None

    G, ring_centers = build_ring_graph(mol, aromatic_rings)
    
    reference_segment = find_reference_segment(G, ring_centers)
    
    print("Reference Segment:", reference_segment)
    print("Reference Segment Length:", len(reference_segment))
    
    edge_info = calculate_angles_to_linear_segment(G, ring_centers, reference_segment)
    
    # Calculate DPO for reference segment
    reference_DPO, reference_assigned_bonds = assign_dpo_to_reference_segment(mol, reference_segment, a)
    
    # Calculate DPO for angulated segments
    angulated_DPO, angulated_assigned_bonds, edge_categories = assign_dpo_to_angulated_segments(mol, G, reference_segment, edge_info, ring_centers, a, b, c, d)
    
    # Combine results
    all_assigned_bonds = reference_assigned_bonds.union(angulated_assigned_bonds)
    
    # Recalculate total DPO based on unique assigned bonds
    DPO_total = sum(edge_categories[edge]['dpo_value'] for edge in edge_categories)
    
    # Add reference segment information to edge_categories
    for i in range(len(reference_segment) - 1):
        edge = tuple(sorted((reference_segment[i], reference_segment[i+1])))
        if edge in edge_info:
            edge_categories[edge] = {
                'category': 'a',
                'dpo_value': 1 - i * a,
                'path': ['a'] * (i + 1),
                'angle': edge_info[edge]['angle']
            }
    
    visualize_graph_with_edge_info(G, ring_centers, edge_info, edge_categories, reference_segment, reference_segment)
    
    print(f"Final DPO_total: {DPO_total}")
    print(f"Total unique assigned bonds: {len(all_assigned_bonds)}")
    
    return DPO_total
def calculate_dpo7(mol, a=0.05, b=-1/4, c=1/3, d=1/3):
    aromatic_rings = detect_aromatic_rings(mol)
    if not aromatic_rings:
        print("No aromatic rings found in the molecule.")
        return None

    G, ring_centers = build_ring_graph(mol, aromatic_rings)
    
    reference_segment = find_reference_segment(G, ring_centers)
    
    print("Reference Segment:", reference_segment)
    print("Reference Segment Length:", len(reference_segment))
    
    edge_info = calculate_angles_to_linear_segment(G, ring_centers, reference_segment)
    
    DPO_total, assigned_bonds, edge_categories = assign_dpo_to_angulated_segments(mol, G, reference_segment, edge_info, ring_centers, a, b, c, d, AngleThreshold)
    
    visualize_graph_with_edge_info(G, ring_centers, edge_info, edge_categories, reference_segment, reference_segment)
    
    print(f"Final DPO_total: {DPO_total}")
    print(f"Total unique assigned bonds: {len(assigned_bonds)}")
    
    return DPO_total
def calculate_dpo(mol, a=0.05, b=-1/4, c=1/3, d=1/3):
    aromatic_rings = detect_aromatic_rings(mol)
    if not aromatic_rings:
        print("No aromatic rings found in the molecule.")
        return None

    G, ring_centers = build_ring_graph(mol, aromatic_rings)
    
    reference_segment = find_reference_segment(G, ring_centers)
    
    print("Reference Segment:", reference_segment)
    print("Reference Segment Length:", len(reference_segment))
    longest_linear_path = len(reference_segment)
    
    # edge_info = calculate_angles_to_linear_segment(G, ring_centers, reference_segment)
    edge_info = calculate_angles_relative_to_reference(G, ring_centers, reference_segment)
    
    #DPO_total, assigned_bonds =assign_dpo_to_reference_segment(mol, reference_segment, a)
    #dpo_temp = DPO_total
    DPO_total, assigned_bonds, edge_categories = assign_dpo_to_angulated_segments(mol, G, reference_segment, edge_info, ring_centers, a, b, c, d)
    #dpo_temp = DPO_total + dpo_temp
    #DPO_total = dpo_temp
#    visualize_graph_with_edge_info(G, ring_centers, edge_info, edge_categories, reference_segment, reference_segment)
    
    print(f"Final DPO_total: {DPO_total}")
    print(f"Total unique assigned bonds: {len(assigned_bonds)}")
    
    return DPO_total, longest_linear_path
def calculate_ring_centers(mol, atom_rings):
    ring_centers = []
    for ring in atom_rings:
        center = np.mean([mol.GetConformer().GetAtomPosition(i) for i in ring], axis=0)
        ring_centers.append(center)
    return ring_centers
def calculate_ring_center2(mol, ring):
    coords = np.array([mol.GetConformer().GetAtomPosition(idx) for idx in ring])
    return np.mean(coords, axis=0)
def calculate_ring_center(mol, ring):
    # Calculate the center of the ring, ignoring non-carbon atoms
    coords = [mol.GetConformer().GetAtomPosition(idx) for idx in ring if mol.GetAtomWithIdx(idx).GetAtomicNum() == 6]
    if not coords:
        coords = [mol.GetConformer().GetAtomPosition(idx) for idx in ring]  # Fallback to all atoms if no carbons
    return np.mean(coords, axis=0)
def angle_between_rings(center1, center2, center3):
    v1 = center1 - center2
    v2 = center3 - center2
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    return np.degrees(angle)
def identify_segments(mol, atom_rings):
    """
    Identify segments based on angles between ring centers.

    Returns:
    - segments: List of tuples (segment_path, segment_category)
    """
    import networkx as nx
    
    G = build_ring_graph(mol, atom_rings)
    ring_centers = calculate_ring_centers(mol, atom_rings)
    segments = []
    
    # Parameters
    angle_threshold_linear = 15  # Threshold for considering angle ~180°
    angle_threshold_120 = 15     # Threshold for considering angle ~120°
    angle_threshold_60 = 15      # Threshold for considering angle ~60°
    
    for start_ring in G.nodes():
        # Start a new path from each ring
        visited_nodes = set()
        path = [start_ring]
        visited_nodes.add(start_ring)
        extend_path(G, path, visited_nodes, ring_centers, segments,
                    angle_threshold_linear, angle_threshold_120, angle_threshold_60)
    
    return segments
def extend_path(G, path, visited_nodes, ring_centers, segments,
                angle_threshold_linear, angle_threshold_120, angle_threshold_60):
    current_ring = path[-1]
    extended = False
    for neighbor in G.neighbors(current_ring):
        if neighbor not in visited_nodes:
            if len(path) >= 2:
                # Calculate the angle at the current ring
                prev_ring = path[-2]
                angle = angle_at_ring_j(ring_centers, prev_ring, current_ring, neighbor)
                
                # Classify the angle
                if abs(angle - 180) <= angle_threshold_linear:
                    category = 'a'
                elif abs(angle - 120) <= angle_threshold_120:
                    category = 'b'
                elif abs(angle - 60) <= angle_threshold_60:
                    category = 'c'
                else:
                    continue  # Angle doesn't match any category, don't extend path
            else:
                # For the first step, assume linear (or handle differently)
                category = 'a'
                angle = None
            
            # Extend the path
            new_path = path + [neighbor]
            visited_nodes.add(neighbor)
            # Store the segment if desired
            segments.append((new_path.copy(), category))
            # Continue extending the path
            extend_path(G, new_path, visited_nodes, ring_centers, segments,
                        angle_threshold_linear, angle_threshold_120, angle_threshold_60)
            extended = True
            visited_nodes.remove(neighbor)  # Backtrack
    # If path cannot be extended further, you may store it here if not already stored
def build_ring_graph_OLD(mol, atom_rings, ring_centers):
   G = nx.Graph()
   for idx, center in enumerate(ring_centers):
       G.add_node(idx, pos=center[:2])  # Use only x and y coordinates
   
   for i in range(len(atom_rings)):
       for j in range(i+1, len(atom_rings)):
           if rings_are_fused(atom_rings[i], atom_rings[j]):
               # Calculate angle between ring normals
               normal_i = calculate_ring_normal(mol, atom_rings[i])
               normal_j = calculate_ring_normal(mol, atom_rings[j])
               angle = angle_between_normals(normal_i, normal_j)
               G.add_edge(i, j, angle=angle)
   
   return G
def build_ring_graph2(mol, atom_rings):
    G = nx.Graph()
    ring_centers = [calculate_ring_center(mol, ring) for ring in atom_rings]
    
    for idx, center in enumerate(ring_centers):
        G.add_node(idx, pos=center[:2])  # Use only x and y coordinates
    
    for i in range(len(atom_rings)):
        for j in range(i+1, len(atom_rings)):
            if rings_are_fused(atom_rings[i], atom_rings[j]):
                G.add_edge(i, j)
    
    # Calculate angles for paths of length 3
    angles = {}
    for node in G.nodes():
        for neighbor1 in G.neighbors(node):
            for neighbor2 in G.neighbors(node):
                if neighbor1 < neighbor2:  # Avoid duplicate paths
                    path = (neighbor1, node, neighbor2)
                    angle = angle_between_rings(ring_centers[neighbor1], ring_centers[node], ring_centers[neighbor2])
                    angles[path] = angle

    nx.set_edge_attributes(G, {(path[1], path[2]): {'angle': angle} for path, angle in angles.items()})

    return G, ring_centers
def build_ring_graph3(mol, atom_rings):
    G = nx.Graph()
    ring_centers = [calculate_ring_center(mol, ring) for ring in atom_rings]
    
    for idx, center in enumerate(ring_centers):
        G.add_node(idx, pos=center[:2])  # Use only x and y coordinates
    
    for i in range(len(atom_rings)):
        for j in range(i+1, len(atom_rings)):
            if rings_are_fused(atom_rings[i], atom_rings[j]):
                G.add_edge(i, j)
    
    # Calculate angles for paths of length 3
    angles = {}
    for node in G.nodes():
        for neighbor1 in G.neighbors(node):
            for neighbor2 in G.neighbors(node):
                if neighbor1 < neighbor2:  # Avoid duplicate paths
                    path = (neighbor1, node, neighbor2)
                    angle = angle_between_rings(ring_centers[neighbor1], ring_centers[node], ring_centers[neighbor2])
                    angles[path] = angle

    # Store angles with direction
    edge_info = {}
    for path, angle in angles.items():
        edge_info[(path[0], path[1])] = {'angle': angle, 'direction': (path[0], path[1])}
        edge_info[(path[1], path[2])] = {'angle': angle, 'direction': (path[1], path[2])}

    return G, ring_centers, edge_info
def build_ring_graph(mol, atom_rings):
    G = nx.Graph()
    ring_centers = [calculate_ring_center(mol, ring) for ring in atom_rings]
    
    for idx, center in enumerate(ring_centers):
        G.add_node(idx, pos=center[:2])  # Use only x and y coordinates
    
    for i in range(len(atom_rings)):
        for j in range(i+1, len(atom_rings)):
            if rings_are_fused(atom_rings[i], atom_rings[j]):
                G.add_edge(i, j)
    
    return G, ring_centers
def calculate_angles_relative_to_reference(G, ring_centers, reference_segment):
    reference_vector = ring_centers[reference_segment[-1]] - ring_centers[reference_segment[0]]
    reference_vector /= np.linalg.norm(reference_vector)  # Normalize the reference vector
    
    edge_info = {}
    for node in G.nodes():
        for neighbor in G.neighbors(node):
            edge_vector = ring_centers[neighbor] - ring_centers[node]
            edge_vector /= np.linalg.norm(edge_vector)  # Normalize the edge vector
            
            # Calculate the angle between the reference vector and the edge vector
            angle = np.degrees(np.arccos(np.clip(np.dot(reference_vector, edge_vector), -1.0, 1.0)))
            
            # Ensure directionality: if the angle is greater than 90 degrees, reverse the direction
            # if angle > 90:
            #     angle = 180 - angle
            #     edge_vector = -edge_vector  # Reverse the edge vector
            # if neighbor > node:
            #     angle = 180 - angle
            #     edge_vector = -edge_vector  # Reverse the edge vector

            # print("calculate_angles_relative_to_reference","node: ", node, "neighbor: ", neighbor, "angle: ", angle)
            edge_info[(node, neighbor)] = {'angle': angle, 'direction': (node, neighbor), 'vector': edge_vector}
    
    return edge_info
def visualize_graph_OLD(G, ring_centers, angles):
    plt.figure(figsize=(12, 10))
    pos = nx.get_node_attributes(G, 'pos')
    
    # Draw the graph
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=12, font_weight='bold')
    
    # Add angle labels with offset
    angle_labels = {node: f"{angle:.1f}°" for node, angle in angles.items()}
    label_pos = {}
    for node, coords in pos.items():
        label_pos[node] = (coords[0], coords[1] + 0.1)  # Offset labels slightly above nodes
    
    nx.draw_networkx_labels(G, label_pos, labels=angle_labels, font_size=8, font_color='red')
    
    # Draw edges
    nx.draw_networkx_edges(G, pos)
    
    plt.title("Ring Graph with Angles")
    plt.axis('equal')
    plt.axis('off')  # Turn off axis
    plt.tight_layout()
    plt.show()
    # plt.savefig('ring_graph_with_angles.png')
    # plt.close()
def visualize_graph_with_edge_angles(G, ring_centers, angles, linear_segment):
    plt.figure(figsize=(12, 10))
    pos = {node: center[:2] for node, center in enumerate(ring_centers)}
    
    # Draw the graph
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=12, font_weight='bold')
    
    # Highlight the linear segment
    nx.draw_networkx_edges(G, pos, edgelist=list(zip(linear_segment, linear_segment[1:])), 
                           edge_color='r', width=2)
    
    # Add angle labels to edges
    edge_labels = {edge: f"{angle:.1f}°" for edge, angle in angles.items()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    
    plt.title("Ring Graph with Edge Angles")
    plt.axis('equal')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
def visualize_graph_with_edge_info2(G, ring_centers, edge_info, edge_categories, linear_segment):
    plt.figure(figsize=(12, 10))
    pos = {node: center[:2] for node, center in enumerate(ring_centers)}
    
    # Draw the graph
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=12, font_weight='bold')
    
    # Highlight the linear segment
    nx.draw_networkx_edges(G, pos, edgelist=list(zip(linear_segment, linear_segment[1:])), 
                           edge_color='r', width=2)
    
    # Add edge labels with angle, category, and DPO value
    edge_labels = {}
    for edge, data in edge_info.items():
        angle = data['angle']
        if edge in edge_categories:
            category = edge_categories[edge]['category']
            dpo_value = edge_categories[edge]['dpo_value']
            edge_labels[edge] = f"{angle:.1f}°\n{category}\n{dpo_value:.3f}"
        else:
            edge_labels[edge] = f"{angle:.1f}°"
    
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    
    plt.title("Ring Graph with Edge Angles, Categories, and DPO Values")
    plt.axis('equal')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
def visualize_graph_with_edge_info3(G, ring_centers, edge_info, edge_categories, linear_segment):
    plt.figure(figsize=(12, 10))
    pos = {node: center[:2] for node, center in enumerate(ring_centers)}
    
    # Draw the graph
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=12, font_weight='bold')
    
    # Highlight the linear segment
    nx.draw_networkx_edges(G, pos, edgelist=list(zip(linear_segment, linear_segment[1:])), 
                           edge_color='r', width=2)
    
    # Add edge labels with angle, category, and DPO value
    edge_labels = {}
    for edge, data in edge_categories.items():
        angle = data['angle']
        category = data['category']
        dpo_value = data['dpo_value']
        path = '->'.join(data['path'])
        edge_labels[edge] = f"{angle:.1f}°\n{category}\n{dpo_value:.3f}\n{path}"
    
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    
    plt.title("Ring Graph with Edge Angles, Categories, and DPO Values")
    plt.axis('equal')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
def visualize_graph_with_edge_info4(G, ring_centers, edge_info, edge_categories, linear_segment, reference_segment):
    plt.figure(figsize=(12, 10))
    pos = {node: center[:2] for node, center in enumerate(ring_centers)}
    
    # Draw the graph
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=12, font_weight='bold')
    
    # Highlight the linear segment
    nx.draw_networkx_edges(G, pos, edgelist=list(zip(linear_segment, linear_segment[1:])), 
                           edge_color='r', width=2)
    
    # Highlight the reference segment
    nx.draw_networkx_edges(G, pos, edgelist=list(zip(reference_segment, reference_segment[1:])), 
                           edge_color='g', width=2)
    
    # Add edge labels with angle, category, and DPO value
    edge_labels = {}
    for edge, data in edge_categories.items():
        angle = data['angle']
        category = data['category']
        dpo_value = data['dpo_value']
        path = '->'.join(data['path'])
        edge_labels[edge] = f"{angle:.1f}°\n{category}\n{dpo_value:.3f}\n{path}"
    
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    
    plt.title("Ring Graph with Edge Angles, Categories, and DPO Values")
    plt.axis('equal')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
def visualize_graph_with_edge_info(G, ring_centers, edge_info, edge_categories, linear_segment, reference_segment):
    plt.figure(figsize=(12, 10))
    pos = {node: center[:2] for node, center in enumerate(ring_centers)}
    
    # Draw the graph
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=12, font_weight='bold')
    
    # Highlight the linear segment
    nx.draw_networkx_edges(G, pos, edgelist=list(zip(linear_segment, linear_segment[1:])), 
                           edge_color='r', width=2)
    
    # Highlight the reference segment
    nx.draw_networkx_edges(G, pos, edgelist=list(zip(reference_segment, reference_segment[1:])), 
                           edge_color='g', width=2)
    
    # Add edge labels with angle, category, and DPO value
    edge_labels = {}
    for edge, data in edge_categories.items():
        angle = data['angle']
        category = data['category']
        dpo_value = data['dpo_value']
        path = '->'.join(data['path'])
        edge_labels[edge] = f"{angle:.1f}°\n{category}\n{dpo_value:.3f}\n{path}"
    
    # Debugging information
    print("Edge labels:", edge_labels)
    
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    
    plt.title("Ring Graph with Edge Angles, Categories, and DPO Values")
    plt.axis('equal')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
def print_graph_path(G):
    def dfs_path(node, visited, path):
        visited.add(node)
        path.append(node)
        print(f"Visiting node: {node}")
        
        for neighbor in G.neighbors(node):
            if neighbor not in visited:
                dfs_path(neighbor, visited, path)
        
        return path

    # Find a terminal node (node with only one neighbor)
    terminal_nodes = [node for node in G.nodes() if G.degree(node) == 1]
    
    if not terminal_nodes:
        print("No terminal nodes found. Starting from an arbitrary node.")
        start_node = list(G.nodes())[0]
    else:
        start_node = terminal_nodes[0]
    
    print(f"Starting from node: {start_node}")
    
    visited = set()
    full_path = dfs_path(start_node, visited, [])
    
    print("\nFull path:")
    print(" -> ".join(map(str, full_path)))
    
    # Check if all nodes were visited
    if len(visited) != len(G.nodes()):
        print("\nWarning: Not all nodes were visited. The graph might be disconnected.")
        unvisited = set(G.nodes()) - visited
        print(f"Unvisited nodes: {unvisited}")
def print_unique_paths(G, reference_segment):
    def dfs_path(node, visited, path, is_reference):
        visited.add(node)
        path.append(node)
        
        print(f"Visiting node: {node}, Is reference: {is_reference}")  # Debug print
        
        if node in reference_segment:
            is_reference = True
        
        for neighbor in G.neighbors(node):
            print(f"  Checking neighbor: {neighbor}")  # Debug print
            if neighbor not in visited:
                if is_reference or neighbor in reference_segment:
                    new_path = dfs_path(neighbor, visited.copy(), path.copy(), is_reference)
                    if new_path:
                        yield new_path
            elif neighbor in reference_segment and not is_reference:
                # We've found a path back to the reference segment
                yield path + [neighbor]
        
        if node == reference_segment[-1]:
            yield path

    start_node = reference_segment[0]
    print(f"Starting from node: {start_node}")  # Debug print
    print(f"Reference segment: {reference_segment}")  # Debug print
    print(f"Graph nodes: {list(G.nodes())}")  # Debug print
    print(f"Graph edges: {list(G.edges())}")  # Debug print

    all_paths = list(dfs_path(start_node, set(), [], True))

    print(f"\nFound {len(all_paths)} unique paths:")
    for i, path in enumerate(all_paths, 1):
        print(f"Path {i}: {' -> '.join(map(str, path))}")

    # Check for unvisited nodes
    all_visited = set(node for path in all_paths for node in path)
    unvisited = set(G.nodes()) - all_visited
    if unvisited:
        print(f"\nWarning: {len(unvisited)} nodes were not visited: {unvisited}")
    else:
        print("\nAll nodes were visited.")
def rings_are_fused(ring1, ring2):
    shared_atoms = set(ring1).intersection(set(ring2))
    return len(shared_atoms) >= 2
def angle_between_normals_OLD(normal1, normal2):
   cos_theta = np.dot(normal1, normal2) / (np.linalg.norm(normal1) * np.linalg.norm(normal2))
   angle = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
   if angle > 90:
       angle = 180 - angle
   return angle
def angle_between_normals(normal1, normal2):
    dot_product = np.dot(normal1, normal2)
    angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
    return np.degrees(angle)
def calculate_ring_normals_OLD(mol, atom_rings):
   """
   Calculate the normal vector for each ring.

   Returns:
   - ring_normals: List of normal vectors for each ring.
   """
   ring_normals = []
   conformer = mol.GetConformer()

   for ring in atom_rings:
       # Get the positions of the atoms in the ring
       coords = np.array([conformer.GetAtomPosition(idx) for idx in ring])
       coords = coords - coords.mean(axis=0)  # Center the coordinates

       # Perform PCA to find the normal vector
       cov = np.cov(coords.T)
       eigenvalues, eigenvectors = np.linalg.eigh(cov)
       normal_vector = eigenvectors[:, 0]  # The normal vector corresponds to the smallest eigenvalue

       ring_normals.append(normal_vector)

   return ring_normals
def calculate_ring_normal(mol, ring):
    # Calculate the normal vector of the ring plane
    coords = [mol.GetConformer().GetAtomPosition(idx) for idx in ring]
    center = np.mean(coords, axis=0)
    centered_coords = coords - center
    _, _, vh = np.linalg.svd(centered_coords)
    return vh[2]  # The last row of vh is the normal vector
def extend_linear_path(G, path, current_node, visited_edges, segments, ring_normals, angle_threshold):
    """
    Recursively extend the linear path by adding fused and aligned rings.
    """
    extended = False
    for neighbor in G.neighbors(current_node):
        if neighbor not in path:
            # Check if the ring is aligned with the previous ring
            prev_node = path[-1]
            angle = angle_between_normals(ring_normals[prev_node], ring_normals[neighbor])
            if angle <= angle_threshold:
                edge = (current_node, neighbor)
                if edge not in visited_edges:
                    print(f"Extending path {path} with neighbor {neighbor}, angle {angle:.2f}")
                    visited_edges.add(edge)
                    new_path = path + [neighbor]
                    extend_linear_path(G, new_path, neighbor, visited_edges, segments, ring_normals, angle_threshold)
                    extended = True
    if not extended:
        # Path cannot be extended further; add it to segments
        print(f"Finalized segment: {path}")
        segments.append(path)
def determine_layer(mol, reference_ring_idx, neighbor_ring_idx):
    """
    Determine if the neighbor ring is an overlayer above or below the reference ring.
    Returns:
    - layer: Positive integer indicating the layer number.
    """
    # Implement logic to determine the layer based on atom positions
    # For simplicity, you can calculate the average Z-coordinate of each ring
    conformer = mol.GetConformer()
    reference_ring_atoms = ring_info.AtomRings()[reference_ring_idx]
    neighbor_ring_atoms = ring_info.AtomRings()[neighbor_ring_idx]

    reference_z = np.mean([conformer.GetAtomPosition(idx).z for idx in reference_ring_atoms])
    neighbor_z = np.mean([conformer.GetAtomPosition(idx).z for idx in neighbor_ring_atoms])

    layer = int(abs(neighbor_z - reference_z) / layer_height)  # Define layer_height as needed
    return layer
def find_neighboring_rings(mol, reference_segment):
    ring_info = mol.GetRingInfo()
    atom_rings = ring_info.AtomRings()
    neighboring_rings = set()

    for ring_idx in reference_segment:
        ring_atoms = set(atom_rings[ring_idx])
        for neighbor_idx, neighbor_atoms in enumerate(atom_rings):
            if neighbor_idx in reference_segment:
                continue
            if len(ring_atoms.intersection(neighbor_atoms)) >= 2:
                neighboring_rings.add(neighbor_idx)

    return neighboring_rings
def find_longest_path_in_subgraph(subgraph):
    longest_path = []
    for node in subgraph.nodes():
        path = dfs_longest_path(subgraph, node, visited=set())
        if len(path) > len(longest_path):
            longest_path = path
    return longest_path
def dfs_longest_path(graph, node, visited):
    visited.add(node)
    max_path = [node]
    for neighbor in graph.neighbors(node):
        if neighbor not in visited:
            path = dfs_longest_path(graph, neighbor, visited)
            if len(path) + 1 > len(max_path):
                max_path = [node] + path
    visited.remove(node)  # Backtrack to allow other paths
    return max_path
def rings_are_fused_and_aligned(ring1, ring2, normal1, normal2, angle_threshold=AngleThreshold):
    """
    Determine if two rings are fused (share a bond) and are aligned (planes are nearly parallel).

    Parameters:
    - ring1, ring2: Lists of atom indices for each ring.
    - normal1, normal2: Normal vectors of the rings.
    - angle_threshold: Maximum angle (in degrees) between ring planes to be considered aligned.

    Returns:
    - True if rings are fused and aligned, False otherwise.
    """
    shared_atoms = set(ring1).intersection(set(ring2))
    if len(shared_atoms) >= 2:
        # Rings are fused
        # Calculate the angle between the normal vectors
        cos_theta = np.dot(normal1, normal2) / (np.linalg.norm(normal1) * np.linalg.norm(normal2))
        angle = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
        if angle > 90:
            angle = 180 - angle  # Ensure angle is between 0 and 90 degrees
        if angle <= angle_threshold:
            return True
    return False
def select_reference_segment(segments):
    """
    Select the reference segment based on the given rules.
    """
    # Rule (a): Segment with the largest number of fused rings
    max_length = max(len(seg) for seg in segments)
    candidate_segments = [seg for seg in segments if len(seg) == max_length]

    if len(candidate_segments) == 1:
        return candidate_segments[0]
    else:
        # Apply Rule (b)
        # For simplicity, select the first candidate
        # You can implement further logic based on the number of parallel fused C–C bonds
        return candidate_segments[0]
def assign_DPO_values(mol, segments, a, b, c, d):
    DPO_total = 0.0
    assigned_bonds = set()
    
    for segment, category in segments:
        if category == 'a':
            # Linear segment
            position = 0
            for i in range(len(segment) - 1):
                ring_idx = segment[i]
                next_ring_idx = segment[i + 1]
                DPO_value = 1 - position * a
                DPO_contribution = assign_dpo_to_fused_bonds(mol, [ring_idx, next_ring_idx], DPO_value, assigned_bonds)
                DPO_total += DPO_contribution
                position += 1
        elif category == 'b':
            # 120° angled segment
            DPO_value = b
            DPO_contribution = assign_dpo_to_fused_bonds(mol, segment, DPO_value, assigned_bonds)
            DPO_total += DPO_contribution
        elif category == 'c':
            # 60° angled segment
            DPO_value = c
            DPO_contribution = assign_dpo_to_fused_bonds(mol, segment, DPO_value, assigned_bonds)
            DPO_total += DPO_contribution
        # Handle overlayers (category 'd') as needed
        print("DPO contribution:",DPO_contribution, "position:",position, "segment",segment, "category",category)
    return DPO_total
def assign_dpo_to_fused_bonds(mol, segment, DPO_value, assigned_bonds):
    """
    Assign DPO values to fused bonds between rings in the segment and return the total DPO contribution from this segment.
    """
    DPO_contribution = 0.0
    ring_info = mol.GetRingInfo()
    bond_rings = ring_info.BondRings()
    for i in range(len(segment) - 1):
        ring_idx = segment[i]
        next_ring_idx = segment[i + 1]
        fused_bonds = get_fused_bonds_between_rings(mol, ring_idx, next_ring_idx, bond_rings)
        for bond_idx in fused_bonds:
            if bond_idx not in assigned_bonds:
                assigned_bonds.add(bond_idx)
                DPO_contribution += DPO_value
    return DPO_contribution
def assign_dpo_to_reference_segment2(mol, reference_segment, a):
    """
    Assign DPO values to fused bonds in the reference segment.
    """
    DPO_total = 0.0
    assigned_bonds = set()
    ring_info = mol.GetRingInfo()
    bond_rings = ring_info.BondRings()
    
    for position, ring_idx in enumerate(reference_segment):
        fused_bonds = get_fused_bonds_in_ring(mol, ring_idx, bond_rings)
        for bond_idx in fused_bonds:
            if bond_idx not in assigned_bonds:
                DPO_value = 1 - position * a
                print("DPO_reference_segment_value",DPO_value,"position",position, "bond",bond_idx,"ring",ring_idx)
                DPO_total += DPO_value
                assigned_bonds.add(bond_idx)
    
    return DPO_total, assigned_bonds
def assign_dpo_to_reference_segment(mol, reference_segment, a):
    """
    Assign DPO values to fused bonds in the reference segment.
    """
    DPO_total = 0.0
    assigned_bonds = set()
    ring_info = mol.GetRingInfo()
    bond_rings = ring_info.BondRings()
    
    for i in range(len(reference_segment) - 1):
        ring_idx = reference_segment[i]
        next_ring_idx = reference_segment[i + 1]
        fused_bonds = get_fused_bonds_between_rings(mol, ring_idx, next_ring_idx, bond_rings)
        
        for bond_idx in fused_bonds:
            if bond_idx not in assigned_bonds:
                DPO_value = 1 - i * a
                print(f"DPO_reference_segment_value: {DPO_value:.4f}, position: {i}, bond: {bond_idx}, rings: {ring_idx}-{next_ring_idx}")
                DPO_total += DPO_value
                assigned_bonds.add(bond_idx)
    
    return DPO_total, assigned_bonds
def assign_dpo_to_angulated_segments_OLD(mol, G, reference_segment, ring_centers, a, b, c, d, assigned_bonds):
    """
    Assign DPO values to angled segments and overlayers.
    """
    DPO_total = 0.0
    ring_info = mol.GetRingInfo()
    bond_rings = ring_info.BondRings()
    
    # Identify neighboring rings not in the reference segment
    neighboring_rings = set()
    for ring_idx in reference_segment:
        neighbors = list(G.neighbors(ring_idx))
        for neighbor in neighbors:
            if neighbor not in reference_segment:
                neighboring_rings.add(neighbor)
    
    for neighbor_ring in neighboring_rings:
        # Find the ring in the reference segment that it's fused to
        fused_with = [ring for ring in reference_segment if rings_are_fused(mol.GetRingInfo().AtomRings()[ring], mol.GetRingInfo().AtomRings()[neighbor_ring])]
        if not fused_with:
            continue  # No fused connection found
        reference_fused_ring = fused_with[0]
        # Find the next ring in the reference segment
        reference_ring_index = reference_segment.index(reference_fused_ring)
        if reference_ring_index + 1 < len(reference_segment):
            previous_reference_ring = reference_segment[reference_ring_index + 1]
        elif reference_ring_index > 0:
            previous_reference_ring = reference_segment[reference_ring_index - 1]
        else:
            continue  # Not enough rings in the reference segment to calculate angle
        
        # Calculate the angle between the reference segment vector and the vector to the neighbor ring
        angle = angle_at_ring_j(ring_centers, previous_reference_ring, reference_fused_ring,  neighbor_ring)
        
        # Determine the category based on the angle
        if abs(angle - 180) <= 15:
            category = 'a'
            DPO_value = 1  # Starting value
            # Further position-based assignments can be handled here
        elif abs(angle - 120) <= 15:
            category = 'b'
            DPO_value = b
        elif abs(angle - 60) <= 15:
            category = 'c'
            DPO_value = c
        else:
            print("angle does not match any category",angle, "for ring",neighbor_ring, "fused with ring",reference_fused_ring)
            continue  # Angle doesn't match any category
        print(f"Angle between reference segment (({previous_reference_ring},{reference_fused_ring}) and neighbor ring {neighbor_ring}: {angle:.2f} degrees, therefore category {category}")
        # Assign DPO to fused bonds between reference_fused_ring and neighbor_ring
        fused_bonds = get_fused_bonds_between_rings(mol, reference_fused_ring, neighbor_ring, bond_rings)
        for bond_idx in fused_bonds:
            if bond_idx not in assigned_bonds:
                DPO_total += DPO_value
                assigned_bonds.add(bond_idx)
        
        # Handle overlayers by applying multiplicative factor d if needed
        # This part requires additional logic based on how overlayers are defined
        # For simplicity, it's omitted here and can be implemented as needed
    
    return DPO_total
def assign_dpo_to_angulated_segments2(mol, G, reference_segment, edge_info, a, b, c, d):
    DPO_total = 0.0
    assigned_bonds = set()
    ring_info = mol.GetRingInfo()
    bond_rings = ring_info.BondRings()
    
    def get_category(angle):
        if abs(angle - 60) <= 15:
            return 'b'
        elif abs(angle - 120) <= 15:
            return 'c'
        elif abs(angle - 0) <= 15 or abs(angle - 180) <= 15:
            return 'a'
        else:
            return None
    
    def calculate_dpo_value(category, path):
        if category == 'a':
            return 1 - len(path) * a
        elif category == 'b':
            return b * (d ** path.count('b'))
        elif category == 'c':
            return c * (d ** path.count('c'))
        else:
            return 0
    
    def assign_dpo_recursive(edge, path):
        nonlocal DPO_total, assigned_bonds
        
        edge_data = edge_info[edge]
        category = get_category(edge_data['angle'])
        
        if category is None:
            return
        
        new_path = path + [category]
        dpo_value = calculate_dpo_value(category, new_path)
        
        fused_bonds = get_fused_bonds_between_rings(mol, edge_data['direction'][0], edge_data['direction'][1], bond_rings)
        for bond_idx in fused_bonds:
            if bond_idx not in assigned_bonds:
                DPO_total += dpo_value
                assigned_bonds.add(bond_idx)
        
        # Recursively process neighboring edges
        for neighbor in G.neighbors(edge_data['direction'][1]):
            next_edge = tuple(sorted((edge_data['direction'][1], neighbor)))
            if next_edge in edge_info and next_edge not in assigned_bonds:
                assign_dpo_recursive(next_edge, new_path)
    
    # Start the recursive assignment from edges connected to the reference segment
    for i in range(len(reference_segment) - 1):
        for neighbor in G.neighbors(reference_segment[i]):
            if neighbor not in reference_segment:
                edge = tuple(sorted((reference_segment[i], neighbor)))
                if edge in edge_info:
                    assign_dpo_recursive(edge, [])
    
    return DPO_total, assigned_bonds
def assign_dpo_to_angulated_segments3(mol, G, reference_segment, edge_info, a, b, c, d):
    DPO_total = 0.0
    assigned_bonds = set()
    edge_categories = {}
    ring_info = mol.GetRingInfo()
    bond_rings = ring_info.BondRings()
    
    def get_category(angle):
        if abs(angle - 60) <= 15:
            return 'b'
        elif abs(angle - 120) <= 15:
            return 'c'
        elif abs(angle - 0) <= 15 or abs(angle - 180) <= 15:
            return 'a'
        else:
            return None
    
    def calculate_dpo_value(category, path):
        if category == 'a':
            return 1 - len(path) * a
        elif category == 'b':
            return b * (d ** path.count('b'))
        elif category == 'c':
            return c * (d ** path.count('c'))
        else:
            return 0
    
    def assign_dpo_recursive(edge, path):
        nonlocal DPO_total, assigned_bonds, edge_categories
        
        edge_data = edge_info[edge]
        category = get_category(edge_data['angle'])
        
        if category is None:
            return
        
        new_path = path + [category]
        dpo_value = calculate_dpo_value(category, new_path)
        
        edge_categories[edge] = {
            'category': category,
            'dpo_value': dpo_value,
            'path': new_path
        }
        
        fused_bonds = get_fused_bonds_between_rings(mol, edge_data['direction'][0], edge_data['direction'][1], bond_rings)
        for bond_idx in fused_bonds:
            if bond_idx not in assigned_bonds:
                DPO_total += dpo_value
                assigned_bonds.add(bond_idx)
        
        # Recursively process neighboring edges
        for neighbor in G.neighbors(edge_data['direction'][1]):
            next_edge = tuple(sorted((edge_data['direction'][1], neighbor)))
            if next_edge in edge_info and next_edge not in assigned_bonds:
                assign_dpo_recursive(next_edge, new_path)
    
    # Start the recursive assignment from edges connected to the reference segment
    for i in range(len(reference_segment) - 1):
        for neighbor in G.neighbors(reference_segment[i]):
            if neighbor not in reference_segment:
                edge = tuple(sorted((reference_segment[i], neighbor)))
                if edge in edge_info:
                    assign_dpo_recursive(edge, [])
    
    return DPO_total, assigned_bonds, edge_categories
def assign_dpo_to_angulated_segments4(mol, G, reference_segment, edge_info, a, b, c, d):
    DPO_total = 0.0
    assigned_bonds = set()
    edge_categories = {}
    ring_info = mol.GetRingInfo()
    bond_rings = ring_info.BondRings()
    
    def get_category(angle):
        if abs(angle - 60) <= 15:
            return 'b'
        elif abs(angle - 120) <= 15:
            return 'c'
        elif abs(angle - 0) <= 15 or abs(angle - 180) <= 15:
            return 'a'
        else:
            return None
    
    def calculate_dpo_value(category, path):
        if category == 'a':
            return 1 - len(path) * a
        elif category == 'b':
            return b * (d ** path.count('b'))
        elif category == 'c':
            return c * (d ** path.count('c'))
        else:
            return 0
    
    # Initialize a queue with edges connected to the reference segment
    edge_queue = []
    for i in range(len(reference_segment) - 1):
        for neighbor in G.neighbors(reference_segment[i]):
            if neighbor not in reference_segment:
                edge = tuple(sorted((reference_segment[i], neighbor)))
                if edge in edge_info:
                    edge_queue.append((edge, []))
    
    # Process edges iteratively
    while edge_queue:
        edge, path = edge_queue.pop(0)
        
        if edge in assigned_bonds:
            continue
        
        edge_data = edge_info[edge]
        category = get_category(edge_data['angle'])
        
        if category is None:
            continue
        
        new_path = path + [category]
        dpo_value = calculate_dpo_value(category, new_path)
        
        edge_categories[edge] = {
            'category': category,
            'dpo_value': dpo_value,
            'path': new_path
        }
        
        fused_bonds = get_fused_bonds_between_rings(mol, edge_data['direction'][0], edge_data['direction'][1], bond_rings)
        for bond_idx in fused_bonds:
            if bond_idx not in assigned_bonds:
                DPO_total += dpo_value
                assigned_bonds.add(bond_idx)
        
        # Add neighboring edges to the queue
        for neighbor in G.neighbors(edge_data['direction'][1]):
            next_edge = tuple(sorted((edge_data['direction'][1], neighbor)))
            if next_edge in edge_info and next_edge not in assigned_bonds:
                edge_queue.append((next_edge, new_path))
    
    return DPO_total, assigned_bonds, edge_categories
def assign_dpo_to_angulated_segments5(mol, G, reference_segment, edge_info, a, b, c, d, max_iterations=1000):
    DPO_total = 0.0
    assigned_bonds = set()
    edge_categories = {}
    ring_info = mol.GetRingInfo()
    bond_rings = ring_info.BondRings()
    
    def get_category(angle):
        if abs(angle - 60) <= 15:
            return 'b'
        elif abs(angle - 120) <= 15:
            return 'c'
        elif abs(angle - 0) <= 15 or abs(angle - 180) <= 15:
            return 'a'
        else:
            return None
    
    def calculate_dpo_value(category, path):
        if category == 'a':
            return 1 - len(path) * a
        elif category == 'b':
            return b * (d ** path.count('b'))
        elif category == 'c':
            return c * (d ** path.count('c'))
        else:
            return 0
    
    # Initialize a queue with edges connected to the reference segment
    edge_queue = []
    for i in range(len(reference_segment) - 1):
        for neighbor in G.neighbors(reference_segment[i]):
            if neighbor not in reference_segment:
                edge = tuple(sorted((reference_segment[i], neighbor)))
                if edge in edge_info:
                    edge_queue.append((edge, []))
    
    # Process edges iteratively
    iteration_count = 0
    while edge_queue and iteration_count < max_iterations:
        iteration_count += 1
        edge, path = edge_queue.pop(0)
        
        print(f"Processing edge {edge}, path: {path}")  # Debug print
        
        if edge in assigned_bonds:
            continue
        
        edge_data = edge_info[edge]
        category = get_category(edge_data['angle'])
        
        if category is None:
            continue
        
        new_path = path + [category]
        dpo_value = calculate_dpo_value(category, new_path)
        
        edge_categories[edge] = {
            'category': category,
            'dpo_value': dpo_value,
            'path': new_path
        }
        
        fused_bonds = get_fused_bonds_between_rings(mol, edge_data['direction'][0], edge_data['direction'][1], bond_rings)
        for bond_idx in fused_bonds:
            if bond_idx not in assigned_bonds:
                DPO_total += dpo_value
                assigned_bonds.add(bond_idx)
        
        # Add neighboring edges to the queue
        for neighbor in G.neighbors(edge_data['direction'][1]):
            next_edge = tuple(sorted((edge_data['direction'][1], neighbor)))
            if next_edge in edge_info and next_edge not in assigned_bonds:
                edge_queue.append((next_edge, new_path))
        
        print(f"Processed edge {edge}, DPO_total: {DPO_total}")  # Debug print
    
    if iteration_count >= max_iterations:
        print(f"Warning: Reached maximum iterations ({max_iterations}). The molecule might be too complex or there might be an issue with the graph structure.")
    
    print(f"Total iterations: {iteration_count}")  # Debug print
    return DPO_total, assigned_bonds, edge_categories
def assign_dpo_to_angulated_segments6(mol, G, reference_segment, edge_info, a, b, c, d, max_iterations=1000):
    DPO_total = 0.0
    assigned_bonds = set()
    edge_categories = {}
    processed_edges = set()  # Keep track of processed edges
    ring_info = mol.GetRingInfo()
    bond_rings = ring_info.BondRings()
    
    def get_category(angle):
        if abs(angle - 60) <= 15:
            return 'b'
        elif abs(angle - 120) <= 15:
            return 'c'
        elif abs(angle - 0) <= 15 or abs(angle - 180) <= 15:
            return 'a'
        else:
            return None
    
    def calculate_dpo_value(category, path):
        if category == 'a':
            return 1 - len(path) * a
        elif category == 'b':
            return b * (d ** path.count('b'))
        elif category == 'c':
            return c * (d ** path.count('c'))
        else:
            return 0
    
    # Initialize a queue with edges connected to the reference segment
    edge_queue = []
    for i in range(len(reference_segment) - 1):
        for neighbor in G.neighbors(reference_segment[i]):
            if neighbor not in reference_segment:
                edge = tuple(sorted((reference_segment[i], neighbor)))
                if edge in edge_info and edge not in processed_edges:
                    edge_queue.append((edge, []))
                    processed_edges.add(edge)
    
    # Process edges iteratively
    iteration_count = 0
    while edge_queue and iteration_count < max_iterations:
        iteration_count += 1
        edge, path = edge_queue.pop(0)
        
        print(f"Processing edge {edge}, path: {path}")  # Debug print
        
        edge_data = edge_info[edge]
        category = get_category(edge_data['angle'])
        
        if category is None:
            continue
        
        new_path = path + [category]
        dpo_value = calculate_dpo_value(category, new_path)
        
        edge_categories[edge] = {
            'category': category,
            'dpo_value': dpo_value,
            'path': new_path
        }
        
        fused_bonds = get_fused_bonds_between_rings(mol, edge_data['direction'][0], edge_data['direction'][1], bond_rings)
        for bond_idx in fused_bonds:
            if bond_idx not in assigned_bonds:
                DPO_total += dpo_value
                assigned_bonds.add(bond_idx)
        
        # Add neighboring edges to the queue
        for neighbor in G.neighbors(edge_data['direction'][1]):
            next_edge = tuple(sorted((edge_data['direction'][1], neighbor)))
            if next_edge in edge_info and next_edge not in processed_edges:
                edge_queue.append((next_edge, new_path))
                processed_edges.add(next_edge)
        
        print(f"Processed edge {edge}, DPO_total: {DPO_total}")  # Debug print
    
    if iteration_count >= max_iterations:
        print(f"Warning: Reached maximum iterations ({max_iterations}). The molecule might be too complex or there might be an issue with the graph structure.")
    
    print(f"Total iterations: {iteration_count}")  # Debug print
    return DPO_total, assigned_bonds, edge_categories
def assign_dpo_to_angulated_segments7(mol, G, reference_segment, edge_info, a, b, c, d, max_iterations=1000):
    DPO_total = 0.0
    assigned_bonds = set()
    edge_categories = {}
    processed_edges = set()
    ring_info = mol.GetRingInfo()
    bond_rings = ring_info.BondRings()
    
    def get_category(angle):
        if abs(angle - 60) <= 15:
            return 'b'
        elif abs(angle - 120) <= 15:
            return 'c'
        elif abs(angle - 0) <= 15 or abs(angle - 180) <= 15:
            return 'a'
        else:
            return 'd'  # Default category for angles that don't fit a, b, or c
    
    def calculate_dpo_value(category, path):
        print("category",category)
        print("path",path)
        if category == 'a':
            return 1 - len(path) * a
        elif category == 'b':
            return b * (d ** path.count('b'))
        elif category == 'c':
            return c * (d ** path.count('c'))
        else:
            return d  # Default DPO value for category 'd'
    
    # Initialize queue with all edges in edge_info
    edge_queue = [(edge, []) for edge in edge_info.keys()]
    
    # Process edges iteratively
    iteration_count = 0
    while edge_queue and iteration_count < max_iterations:
        iteration_count += 1
        edge, path = edge_queue.pop(0)
        
        if edge in processed_edges:
            continue
        
        processed_edges.add(edge)
        
        print(f"Processing edge {edge}, path: {path}")  # Debug print
        
        edge_data = edge_info[edge]
        category = get_category(edge_data['angle'])
        
        new_path = path + [category]
        dpo_value = calculate_dpo_value(category, new_path)
        
        edge_categories[edge] = {
            'category': category,
            'dpo_value': dpo_value,
            'path': new_path,
            'angle': edge_data['angle']
        }
        
        fused_bonds = get_fused_bonds_between_rings(mol, edge_data['direction'][0], edge_data['direction'][1], bond_rings)
        for bond_idx in fused_bonds:
            if bond_idx not in assigned_bonds:
                DPO_total += dpo_value
                assigned_bonds.add(bond_idx)
        
        # Add neighboring edges to the queue
        for neighbor in G.neighbors(edge_data['direction'][1]):
            next_edge = tuple(sorted((edge_data['direction'][1], neighbor)))
            if next_edge in edge_info and next_edge not in processed_edges:
                edge_queue.append((next_edge, new_path))
        
        print(f"Processed edge {edge}, category: {category}, DPO_value: {dpo_value}, DPO_total: {DPO_total}")  # Debug print
    
    if iteration_count >= max_iterations:
        print(f"Warning: Reached maximum iterations ({max_iterations}). The molecule might be too complex or there might be an issue with the graph structure.")
    
    print(f"Total iterations: {iteration_count}")  # Debug print
    print(f"Processed edges: {len(processed_edges)}, Total edges in edge_info: {len(edge_info)}")  # Debug print
    
    # Check for unprocessed edges
    unprocessed_edges = set(edge_info.keys()) - processed_edges
    if unprocessed_edges:
        print(f"Warning: {len(unprocessed_edges)} edges were not processed: {unprocessed_edges}")
    
    return DPO_total, assigned_bonds, edge_categories
def assign_dpo_to_angulated_segments8(mol, G, reference_segment, edge_info, a, b, c, d):
    DPO_total = 0.0
    assigned_bonds = set()
    edge_categories = {}
    ring_info = mol.GetRingInfo()
    bond_rings = ring_info.BondRings()
    
    def get_category(angle):
        if abs(angle - 60) <= 15:
            return 'b'
        elif abs(angle - 120) <= 15:
            return 'c'
        elif abs(angle - 0) <= 15 or abs(angle - 180) <= 15:
            return 'a'
        else:
            return 'd'  # Default category for angles that don't fit a, b, or c
    
    def calculate_dpo_value(category, path):
        if category == 'a':
            return 1 - len(path) * a
        elif category == 'b':
            return b * (d ** path.count('b'))
        elif category == 'c':
            return c * (d ** path.count('c'))
        else:
            return d  # Default DPO value for category 'd'
    
    # Initialize BFS queue with edges connected to the reference segment
    queue = deque()
    visited_edges = set()
    for i in range(len(reference_segment) - 1):
        edge = tuple(sorted((reference_segment[i], reference_segment[i+1])))
        if edge in edge_info:
            queue.append((edge, []))
            visited_edges.add(edge)
    
    while queue:
        current_edge, path = queue.popleft()
        edge_data = edge_info[current_edge]
        category = get_category(edge_data['angle'])
        new_path = path + [category]
        
        dpo_value = calculate_dpo_value(category, new_path)
        edge_categories[current_edge] = {
            'category': category,
            'dpo_value': dpo_value,
            'path': new_path,
            'angle': edge_data['angle']
        }
        
        fused_bonds = get_fused_bonds_between_rings(mol, edge_data['direction'][0], edge_data['direction'][1], bond_rings)
        for bond_idx in fused_bonds:
            if bond_idx not in assigned_bonds:
                DPO_total += dpo_value
                assigned_bonds.add(bond_idx)
        
        # Add neighboring edges to the queue
        for node in edge_data['direction']:
            for neighbor in G.neighbors(node):
                next_edge = tuple(sorted((node, neighbor)))
                if next_edge in edge_info and next_edge not in visited_edges:
                    queue.append((next_edge, new_path))
                    visited_edges.add(next_edge)
        
        print(f"Processed edge {current_edge}, category: {category}, path: {new_path}, DPO_value: {dpo_value}, DPO_total: {DPO_total}")
    
    # Check for unprocessed edges
    unprocessed_edges = set(edge_info.keys()) - visited_edges
    if unprocessed_edges:
        print(f"Warning: {len(unprocessed_edges)} edges were not processed: {unprocessed_edges}")
        # Process unconnected edges
        for edge in unprocessed_edges:
            edge_data = edge_info[edge]
            category = get_category(edge_data['angle'])
            dpo_value = calculate_dpo_value(category, [category])
            edge_categories[edge] = {
                'category': category,
                'dpo_value': dpo_value,
                'path': [category],
                'angle': edge_data['angle']
            }
            print(f"Processed unconnected edge {edge}, category: {category}, path: [{category}], DPO_value: {dpo_value}")
    
    return DPO_total, assigned_bonds, edge_categories
def assign_dpo_to_angulated_segments9(mol, G, reference_segment, edge_info, a, b, c, d):
    DPO_total = 0.0
    assigned_bonds = set()
    edge_categories = {}
    ring_info = mol.GetRingInfo()
    bond_rings = ring_info.BondRings()
    
    def get_base_category(angle):
        if abs(angle - 60) <= 15:
            return 'b'
        elif abs(angle - 120) <= 15:
            return 'c'
        elif abs(angle - 0) <= 15 or abs(angle - 180) <= 15:
            return 'a'
        else:
            return 'd'  # Default category for angles that don't fit a, b, or c
    
    def calculate_dpo_value(category, path):
        if category == 'a':
            return 1 - len([c for c in path if c != 'a']) * a
        elif category.startswith('b'):
            return b * (d ** (int(category[1:]) if len(category) > 1 else 0))
        elif category.startswith('c'):
            return c * (d ** (int(category[1:]) if len(category) > 1 else 0))
        else:
            return d  # Default DPO value for category 'd'
    
    # Initialize BFS queue with nodes from the reference segment
    queue = deque([(node, []) for node in reference_segment])
    visited_nodes = set(reference_segment)
    
    while queue:
        current_node, path = queue.popleft()
        
        for neighbor in G.neighbors(current_node):
            if neighbor in visited_nodes:
                continue
            
            visited_nodes.add(neighbor)
            edge = tuple(sorted((current_node, neighbor)))
            
            if edge not in edge_info:
                continue
            
            edge_data = edge_info[edge]
            base_category = get_base_category(edge_data['angle'])
            
            if base_category in ['b', 'c']:
                category = f"{base_category}{path.count('b') + path.count('c')}"
            else:
                category = base_category
            
            new_path = path + [category]
            dpo_value = calculate_dpo_value(category, new_path)
            
            edge_categories[edge] = {
                'category': category,
                'dpo_value': dpo_value,
                'path': new_path,
                'angle': edge_data['angle']
            }
            
            fused_bonds = get_fused_bonds_between_rings(mol, current_node, neighbor, bond_rings)
            for bond_idx in fused_bonds:
                if bond_idx not in assigned_bonds:
                    DPO_total += dpo_value
                    assigned_bonds.add(bond_idx)
            
            queue.append((neighbor, new_path))
            
            print(f"Processed edge {edge}, category: {category}, path: {new_path}, DPO_value: {dpo_value}, DPO_total: {DPO_total}")
    
    # Check for unprocessed edges
    unprocessed_edges = set(edge_info.keys()) - set(edge_categories.keys())
    if unprocessed_edges:
        print(f"Warning: {len(unprocessed_edges)} edges were not processed: {unprocessed_edges}")
        # Process unconnected edges
        for edge in unprocessed_edges:
            edge_data = edge_info[edge]
            category = get_base_category(edge_data['angle'])
            dpo_value = calculate_dpo_value(category, [category])
            edge_categories[edge] = {
                'category': category,
                'dpo_value': dpo_value,
                'path': [category],
                'angle': edge_data['angle']
            }
            print(f"Processed unconnected edge {edge}, category: {category}, path: [{category}], DPO_value: {dpo_value}")
    
    return DPO_total, assigned_bonds, edge_categories
def assign_dpo_to_angulated_segments10(mol, G, reference_segment, edge_info, a, b, c, d):
    DPO_total = 0.0
    assigned_bonds = set()
    edge_categories = {}
    ring_info = mol.GetRingInfo()
    bond_rings = ring_info.BondRings()
    
    def get_base_category(angle):
        if abs(angle - 60) <= 15:
            return 'b'
        elif abs(angle - 120) <= 15:
            return 'c'
        elif abs(angle - 0) <= 15 or abs(angle - 180) <= 15:
            return 'a'
        else:
            return 'd'  # Default category for angles that don't fit a, b, or c
    
    def calculate_dpo_value(category, path):
        if category == 'a':
            return 1 - len([c for c in path if c != 'a']) * a
        elif category.startswith('b'):
            b_count = sum(1 for c in path if c.startswith('b'))
            return b * (d ** (b_count - 1))  # Apply d for each preceding 'b'
        elif category.startswith('c'):
            c_count = sum(1 for c in path if c.startswith('c'))
            return c * (d ** (c_count - 1))  # Apply d for each preceding 'c'
        else:
            return d  # Default DPO value for category 'd'
    
    # Initialize BFS queue with nodes from the reference segment
    queue = deque([(node, []) for node in reference_segment])
    visited_nodes = set(reference_segment)
    
    while queue:
        current_node, path = queue.popleft()
        
        for neighbor in G.neighbors(current_node):
            if neighbor in visited_nodes:
                continue
            
            visited_nodes.add(neighbor)
            edge = tuple(sorted((current_node, neighbor)))
            
            if edge not in edge_info:
                continue
            
            edge_data = edge_info[edge]
            base_category = get_base_category(edge_data['angle'])
            
            if base_category in ['b', 'c']:
                category = f"{base_category}{path.count('b') + path.count('c')}"
            else:
                category = base_category
            
            new_path = path + [category]
            dpo_value = calculate_dpo_value(category, new_path)
            
            edge_categories[edge] = {
                'category': category,
                'dpo_value': dpo_value,
                'path': new_path,
                'angle': edge_data['angle']
            }
            
            fused_bonds = get_fused_bonds_between_rings(mol, current_node, neighbor, bond_rings)
            for bond_idx in fused_bonds:
                if bond_idx not in assigned_bonds:
                    DPO_total += dpo_value
                    assigned_bonds.add(bond_idx)
            
            queue.append((neighbor, new_path))
            
            print(f"Processed edge {edge}, category: {category}, path: {new_path}, DPO_value: {dpo_value}, DPO_total: {DPO_total}")
    
    # Check for unprocessed edges
    unprocessed_edges = set(edge_info.keys()) - set(edge_categories.keys())
    if unprocessed_edges:
        print(f"Warning: {len(unprocessed_edges)} edges were not processed: {unprocessed_edges}")
        # Process unconnected edges
        for edge in unprocessed_edges:
            edge_data = edge_info[edge]
            category = get_base_category(edge_data['angle'])
            dpo_value = calculate_dpo_value(category, [category])
            edge_categories[edge] = {
                'category': category,
                'dpo_value': dpo_value,
                'path': [category],
                'angle': edge_data['angle']
            }
            print(f"Processed unconnected edge {edge}, category: {category}, path: [{category}], DPO_value: {dpo_value}")
    
    return DPO_total, assigned_bonds, edge_categories
def assign_dpo_to_angulated_segments11(mol, G, reference_segment, edge_info, a, b, c, d):
    DPO_total = 0.0
    assigned_bonds = set()
    edge_categories = {}
    ring_info = mol.GetRingInfo()
    bond_rings = ring_info.BondRings()
    
    def get_base_category(angle):
        if abs(angle - 60) <= 15:
            return 'b'
        elif abs(angle - 120) <= 15:
            return 'c'
        elif abs(angle - 0) <= 15 or abs(angle - 180) <= 15:
            return 'a'
        else:
            return 'd'  # Default category for angles that don't fit a, b, or c
    
    def calculate_dpo_value(category, path):
        if category.startswith('a'):
            # Count 'a' segments, resetting after each 'b' or 'c'
            a_count = 0
            for c in reversed(path):
                if c.startswith('a'):
                    a_count += 1
                elif c.startswith('b') or c.startswith('c'):
                    break
            
            # Count 'b' and 'c' segments
            bc_count = sum(1 for c in path if c.startswith('b') or c.startswith('c'))
            
            return (1 - (a_count - 1) * a) * (d ** bc_count)
        elif category.startswith('b'):
            b_count = sum(1 for c in path if c.startswith('b'))
            return b * (d ** (b_count - 1))
        elif category.startswith('c'):
            c_count = sum(1 for c in path if c.startswith('c'))
            return c * (d ** (c_count - 1))
        else:
            return d  # Default DPO value for category 'd'
    
    # Process the reference segment first
    for i in range(len(reference_segment) - 1):
        edge = tuple(sorted((reference_segment[i], reference_segment[i+1])))
        if edge in edge_info:
            edge_data = edge_info[edge]
            category = 'a0'  # Reference segment is always category 'a'
            dpo_value = 1  # First 'a' segment always has a DPO value of 1
            edge_categories[edge] = {
                'category': category,
                'dpo_value': dpo_value,
                'path': [category],
                'angle': edge_data['angle']
            }
            fused_bonds = get_fused_bonds_between_rings(mol, reference_segment[i], reference_segment[i+1], bond_rings)
            for bond_idx in fused_bonds:
                if bond_idx not in assigned_bonds:
                    DPO_total += dpo_value
                    assigned_bonds.add(bond_idx)
            print(f"Processed reference edge {edge}, category: {category}, path: [{category}], DPO_value: {dpo_value}, DPO_total: {DPO_total}")
    
    # Initialize BFS queue with nodes from the reference segment
    queue = deque([(node, ['a0']) for node in reference_segment])
    visited_nodes = set(reference_segment)
    
    while queue:
        current_node, path = queue.popleft()
        
        for neighbor in G.neighbors(current_node):
            if neighbor in visited_nodes:
                continue
            
            visited_nodes.add(neighbor)
            edge = tuple(sorted((current_node, neighbor)))
            
            if edge not in edge_info:
                continue
            
            edge_data = edge_info[edge]
            base_category = get_base_category(edge_data['angle'])
            
            if base_category in ['b', 'c']:
                category = f"{base_category}{path.count('b') + path.count('c')}"
            else:
                category = f"{base_category}{path.count('a')}"
            
            new_path = path + [category]
            dpo_value = calculate_dpo_value(category, new_path)
            
            edge_categories[edge] = {
                'category': category,
                'dpo_value': dpo_value,
                'path': new_path,
                'angle': edge_data['angle']
            }
            
            fused_bonds = get_fused_bonds_between_rings(mol, current_node, neighbor, bond_rings)
            for bond_idx in fused_bonds:
                if bond_idx not in assigned_bonds:
                    DPO_total += dpo_value
                    assigned_bonds.add(bond_idx)
            
            queue.append((neighbor, new_path))
            
            print(f"Processed edge {edge}, category: {category}, path: {new_path}, DPO_value: {dpo_value}, DPO_total: {DPO_total}")
    
    # Check for unprocessed edges
    unprocessed_edges = set(edge_info.keys()) - set(edge_categories.keys())
    if unprocessed_edges:
        print(f"Warning: {len(unprocessed_edges)} edges were not processed: {unprocessed_edges}")
        # Process unconnected edges
        for edge in unprocessed_edges:
            edge_data = edge_info[edge]
            category = get_base_category(edge_data['angle'])
            dpo_value = calculate_dpo_value(category, [category])
            edge_categories[edge] = {
                'category': category,
                'dpo_value': dpo_value,
                'path': [category],
                'angle': edge_data['angle']
            }
            print(f"Processed unconnected edge {edge}, category: {category}, path: [{category}], DPO_value: {dpo_value}")
    
    return DPO_total, assigned_bonds, edge_categories
def assign_dpo_to_angulated_segments12(mol, G, reference_segment, edge_info, a, b, c, d):
    DPO_total = 0.0
    assigned_bonds = set()
    edge_categories = {}
    ring_info = mol.GetRingInfo()
    bond_rings = ring_info.BondRings()
    
    def get_base_category(angle):
        if abs(angle - 60) <= 15:
            return 'b'
        elif abs(angle - 120) <= 15:
            return 'c'
        elif abs(angle - 0) <= 15 or abs(angle - 180) <= 15:
            return 'a'
        else:
            return 'd'  # Default category for angles that don't fit a, b, or c
    
    def calculate_dpo_value(category, path):
        if category.startswith('a'):
            # Count 'a' segments, resetting after each 'b' or 'c'
            a_count = 0
            for c in reversed(path):
                if c.startswith('a'):
                    a_count += 1
                elif c.startswith('b') or c.startswith('c'):
                    break
            
            # Count 'b' and 'c' segments
            bc_count = sum(1 for c in path if c.startswith('b') or c.startswith('c'))
            
            return (1 - (a_count - 1) * a) * (d ** bc_count)
        elif category.startswith('b'):
            b_count = sum(1 for c in path if c.startswith('b'))
            return b * (d ** (b_count - 1))
        elif category.startswith('c'):
            c_count = sum(1 for c in path if c.startswith('c'))
            return c * (d ** (c_count - 1))
        else:
            return d  # Default DPO value for category 'd'
    
    # Process the reference segment first
    for i in range(len(reference_segment) - 1):
        edge = tuple(sorted((reference_segment[i], reference_segment[i+1])))
        if edge in edge_info:
            edge_data = edge_info[edge]
            category = 'a0'  # Reference segment is always category 'a'
            dpo_value = 1  # First 'a' segment always has a DPO value of 1
            edge_categories[edge] = {
                'category': category,
                'dpo_value': dpo_value,
                'path': [category],
                'angle': edge_data['angle']
            }
            fused_bonds = get_fused_bonds_between_rings(mol, reference_segment[i], reference_segment[i+1], bond_rings)
            for bond_idx in fused_bonds:
                if bond_idx not in assigned_bonds:
                    DPO_total += dpo_value
                    assigned_bonds.add(bond_idx)
            print(f"Processed reference edge {edge}, category: {category}, path: [{category}], DPO_value: {dpo_value}, DPO_total: {DPO_total}")
    
    # Initialize BFS queue with nodes from the reference segment
    queue = deque([(node, ['a0']) for node in reference_segment])
    visited_nodes = set(reference_segment)
    
    while queue:
        current_node, path = queue.popleft()
        
        for neighbor in G.neighbors(current_node):
            if neighbor in visited_nodes:
                continue
            
            visited_nodes.add(neighbor)
            edge = tuple(sorted((current_node, neighbor)))
            
            if edge not in edge_info:
                continue
            
            edge_data = edge_info[edge]
            base_category = get_base_category(edge_data['angle'])
            
            if base_category in ['b', 'c']:
                category = f"{base_category}{path.count('b') + path.count('c')}"
            else:
                category = f"{base_category}{path.count('a')}"
            
            new_path = path + [category]
            dpo_value = calculate_dpo_value(category, new_path)
            
            edge_categories[edge] = {
                'category': category,
                'dpo_value': dpo_value,
                'path': new_path,
                'angle': edge_data['angle']
            }
            
            fused_bonds = get_fused_bonds_between_rings(mol, current_node, neighbor, bond_rings)
            for bond_idx in fused_bonds:
                if bond_idx not in assigned_bonds:
                    DPO_total += dpo_value
                    assigned_bonds.add(bond_idx)
            
            queue.append((neighbor, new_path))
            
            print(f"Processed edge {edge}, category: {category}, path: {new_path}, DPO_value: {dpo_value}, DPO_total: {DPO_total}")
    
    # Check for unprocessed edges
    unprocessed_edges = set(edge_info.keys()) - set(edge_categories.keys())
    if unprocessed_edges:
        print(f"Warning: {len(unprocessed_edges)} edges were not processed: {unprocessed_edges}")
        # Process unconnected edges
        for edge in unprocessed_edges:
            edge_data = edge_info[edge]
            category = get_base_category(edge_data['angle'])
            dpo_value = calculate_dpo_value(category, [category])
            edge_categories[edge] = {
                'category': category,
                'dpo_value': dpo_value,
                'path': [category],
                'angle': edge_data['angle']
            }
            print(f"Processed unconnected edge {edge}, category: {category}, path: [{category}], DPO_value: {dpo_value}")
    
    return DPO_total, assigned_bonds, edge_categories
def assign_dpo_to_angulated_segments13(mol, G, reference_segment, edge_info, a, b, c, d):
    DPO_total = 0.0
    assigned_bonds = set()
    edge_categories = {}
    ring_info = mol.GetRingInfo()
    bond_rings = ring_info.BondRings()
    
    def get_base_category(angle):
        if abs(angle - 60) <= 15:
            return 'b'
        elif abs(angle - 120) <= 15:
            return 'c'
        elif abs(angle - 0) <= 15 or abs(angle - 180) <= 15:
            return 'a'
        else:
            return 'd'  # Default category for angles that don't fit a, b, or c
    
    def calculate_dpo_value(category, path, a, b, c, d):
        if category.startswith('a'):
            # Count 'a' segments, resetting after each 'b' or 'c'
            a_count = 0
            for cat in reversed(path):
                if cat.startswith('a'):
                    a_count += 1
                elif cat.startswith('b') or cat.startswith('c'):
                    break
            
            # Count 'b' and 'c' segments
            bc_count = sum(1 for cat in path if cat.startswith('b') or cat.startswith('c'))
            
            return (1 - (a_count - 1) * a) * (d ** bc_count)
        elif category.startswith('b'):
            b_count = sum(1 for cat in path if cat.startswith('b'))
            return b * (d ** (b_count - 1))
        elif category.startswith('c'):
            c_count = sum(1 for cat in path if cat.startswith('c'))
            return c * (d ** (c_count - 1))
        else:
            return d  # Default DPO value for category 'd'
    
    # Process the reference segment first
    for i in range(len(reference_segment) - 1):
        edge = tuple(sorted((reference_segment[i], reference_segment[i+1])))
        if edge in edge_info:
            edge_data = edge_info[edge]
            category = 'a0'  # Reference segment is always category 'a'
            dpo_value = 1  # First 'a' segment always has a DPO value of 1
            edge_categories[edge] = {
                'category': category,
                'dpo_value': dpo_value,
                'path': [category],
                'angle': edge_data['angle']
            }
            fused_bonds = get_fused_bonds_between_rings(mol, reference_segment[i], reference_segment[i+1], bond_rings)
            for bond_idx in fused_bonds:
                if bond_idx not in assigned_bonds:
                    DPO_total += dpo_value
                    assigned_bonds.add(bond_idx)
            print(f"Processed reference edge {edge}, category: {category}, path: [{category}], DPO_value: {dpo_value}, DPO_total: {DPO_total}")
    
    # Initialize BFS queue with nodes from the reference segment
    queue = deque([(node, ['a0']) for node in reference_segment])
    visited_nodes = set(reference_segment)
    
    while queue:
        current_node, path = queue.popleft()
        
        for neighbor in G.neighbors(current_node):
            if neighbor in visited_nodes:
                continue
            
            visited_nodes.add(neighbor)
            edge = tuple(sorted((current_node, neighbor)))
            
            if edge not in edge_info:
                continue
            
            edge_data = edge_info[edge]
            base_category = get_base_category(edge_data['angle'])
            
            if base_category in ['b', 'c']:
                category = f"{base_category}{path.count('b') + path.count('c')}"
            else:
                category = f"{base_category}{path.count('a')}"
            
            new_path = path + [category]
            dpo_value = calculate_dpo_value(category, new_path, a, b, c, d)
            
            edge_categories[edge] = {
                'category': category,
                'dpo_value': dpo_value,
                'path': new_path,
                'angle': edge_data['angle']
            }
            
            fused_bonds = get_fused_bonds_between_rings(mol, current_node, neighbor, bond_rings)
            for bond_idx in fused_bonds:
                if bond_idx not in assigned_bonds:
                    DPO_total += dpo_value
                    assigned_bonds.add(bond_idx)
            
            queue.append((neighbor, new_path))
            
            print(f"Processed edge {edge}, category: {category}, path: {new_path}, DPO_value: {dpo_value}, DPO_total: {DPO_total}")
    
    # Check for unprocessed edges
    unprocessed_edges = set(edge_info.keys()) - set(edge_categories.keys())
    if unprocessed_edges:
        print(f"Warning: {len(unprocessed_edges)} edges were not processed: {unprocessed_edges}")
        # Process unconnected edges
        for edge in unprocessed_edges:
            edge_data = edge_info[edge]
            category = get_base_category(edge_data['angle'])
            dpo_value = calculate_dpo_value(category, [category], a, b, c, d)
            edge_categories[edge] = {
                'category': category,
                'dpo_value': dpo_value,
                'path': [category],
                'angle': edge_data['angle']
            }
            print(f"Processed unconnected edge {edge}, category: {category}, path: [{category}], DPO_value: {dpo_value}")
    
    return DPO_total, assigned_bonds, edge_categories
def assign_dpo_to_angulated_segments14(mol, G, reference_segment, edge_info, ring_centers, a, b, c, d):
    DPO_total = 0.0
    assigned_bonds = set()
    edge_categories = {}
    ring_info = mol.GetRingInfo()
    bond_rings = ring_info.BondRings()
    
    def get_base_category(angle, path_direction, edge_direction):
        # Adjust angle based on path and edge directions
        if np.dot(path_direction, edge_direction) < 0:
            angle = 180 - angle  # Flip the angle if directions are opposite
        
        if abs(angle - 60) <= 15:
            return 'b'
        elif abs(angle - 120) <= 15:
            return 'c'
        elif abs(angle - 0) <= 15 or abs(angle - 180) <= 15:
            return 'a'
        else:
            return 'd'  # Default category for angles that don't fit a, b, or c
    
    def calculate_dpo_value(category, path, a, b, c, d):
        if category.startswith('a'):
            # Count 'a' segments, resetting after each 'b' or 'c'
            a_count = 0
            for cat in reversed(path):
                if cat.startswith('a'):
                    a_count += 1
                elif cat.startswith('b') or cat.startswith('c'):
                    break
            
            # Count 'b' and 'c' segments
            bc_count = sum(1 for cat in path if cat.startswith('b') or cat.startswith('c'))
            
            return (1 - (a_count - 1) * a) * (d ** bc_count)
        elif category.startswith('b'):
            b_count = sum(1 for cat in path if cat.startswith('b'))
            return b * (d ** (b_count - 1))
        elif category.startswith('c'):
            c_count = sum(1 for cat in path if cat.startswith('c'))
            return c * (d ** (c_count - 1))
        else:
            return d  # Default DPO value for category 'd'

    # Process the reference segment first
    for i in range(len(reference_segment) - 1):
        edge = tuple(sorted((reference_segment[i], reference_segment[i+1])))
        if edge in edge_info:
            edge_data = edge_info[edge]
            category = 'a0'  # Reference segment is always category 'a'
            dpo_value = 1  # First 'a' segment always has a DPO value of 1
            edge_categories[edge] = {
                'category': category,
                'dpo_value': dpo_value,
                'path': [category],
                'angle': edge_data['angle']
            }
            fused_bonds = get_fused_bonds_between_rings(mol, reference_segment[i], reference_segment[i+1], bond_rings)
            for bond_idx in fused_bonds:
                if bond_idx not in assigned_bonds:
                    DPO_total += dpo_value
                    assigned_bonds.add(bond_idx)
            print(f"Processed reference edge {edge}, category: {category}, path: [{category}], DPO_value: {dpo_value}, DPO_total: {DPO_total}")
    
    # Initialize BFS queue with nodes from the reference segment
    queue = deque([(node, ['a0'], reference_segment.index(node)) for node in reference_segment])
    visited_nodes = set(reference_segment)
    
    while queue:
        current_node, path, path_index = queue.popleft()
        
        for neighbor in G.neighbors(current_node):
            if neighbor in visited_nodes:
                continue
            
            visited_nodes.add(neighbor)
            edge = tuple(sorted((current_node, neighbor)))
            
            if edge not in edge_info:
                continue
            
            edge_data = edge_info[edge]
            
            # Calculate path direction
            if path_index > 0:
                prev_node = reference_segment[path_index - 1]
                path_direction = ring_centers[current_node] - ring_centers[prev_node]
            else:
                next_node = reference_segment[path_index + 1]
                path_direction = ring_centers[next_node] - ring_centers[current_node]
            
            # Calculate edge direction
            edge_direction = ring_centers[neighbor] - ring_centers[current_node]
            
            base_category = get_base_category(edge_data['angle'], path_direction, edge_direction)
            
            if base_category in ['b', 'c']:
                category = f"{base_category}{path.count('b') + path.count('c')}"
            else:
                category = f"{base_category}{path.count('a')}"
            
            new_path = path + [category]
            dpo_value = calculate_dpo_value(category, new_path, a, b, c, d)
            
            edge_categories[edge] = {
                'category': category,
                'dpo_value': dpo_value,
                'path': new_path,
                'angle': edge_data['angle']
            }
            
            fused_bonds = get_fused_bonds_between_rings(mol, current_node, neighbor, bond_rings)
            for bond_idx in fused_bonds:
                if bond_idx not in assigned_bonds:
                    DPO_total += dpo_value
                    assigned_bonds.add(bond_idx)
            
            queue.append((neighbor, new_path, -1))  # -1 indicates it's not part of the reference segment
            
            print(f"Processed edge {edge}, category: {category}, path: {new_path}, DPO_value: {dpo_value}, DPO_total: {DPO_total}")
    
    # Check for unprocessed edges
    unprocessed_edges = set(edge_info.keys()) - set(edge_categories.keys())
    if unprocessed_edges:
        print(f"Warning: {len(unprocessed_edges)} edges were not processed: {unprocessed_edges}")
        # Process unconnected edges
        for edge in unprocessed_edges:
            edge_data = edge_info[edge]
            category = get_base_category(edge_data['angle'])
            dpo_value = calculate_dpo_value(category, [category], a, b, c, d)
            edge_categories[edge] = {
                'category': category,
                'dpo_value': dpo_value,
                'path': [category],
                'angle': edge_data['angle']
            }
            print(f"Processed unconnected edge {edge}, category: {category}, path: [{category}], DPO_value: {dpo_value}")
    

    return DPO_total, assigned_bonds, edge_categories
def assign_dpo_to_angulated_segments15(mol, G, reference_segment, edge_info, ring_centers, a, b, c, d):
    DPO_total = 0.0
    assigned_bonds = set()
    edge_categories = {}
    ring_info = mol.GetRingInfo()
    bond_rings = ring_info.BondRings()
    
    def get_base_category(angle, path_direction=None, edge_direction=None):
        if path_direction is not None and edge_direction is not None:
            # Adjust angle based on path and edge directions
            print(np.dot(path_direction, edge_direction))
            if np.dot(path_direction, edge_direction) < 0:
                angle = 180 - angle  # Flip the angle if directions are opposite
                print("flipped angle",angle,path_direction,edge_direction)
                
        
        if abs(angle - 60) <= 15:
            return 'b', angle
        elif abs(angle - 120) <= 15:
            return 'c', angle
        elif abs(angle - 0) <= 15 or abs(angle - 180) <= 15:
            return 'a', angle
        else:
            return 'd', angle  # Default category for angles that don't fit a, b, or c
    
    def calculate_dpo_value(category, path, a, b, c, d):
        if category.startswith('a'):
            # Count 'a' segments, resetting after each 'b' or 'c'
            a_count = 0
            for cat in reversed(path):
                if cat.startswith('a'):
                    a_count += 1
                elif cat.startswith('b') or cat.startswith('c'):
                    break
            
            # Count 'b' and 'c' segments
            bc_count = sum(1 for cat in path if cat.startswith('b') or cat.startswith('c'))
            
            return (1 - (a_count - 1) * a) * (d ** bc_count)
        elif category.startswith('b'):
            b_count = sum(1 for cat in path if cat.startswith('b'))
            return b * (d ** (b_count - 1))
        elif category.startswith('c'):
            c_count = sum(1 for cat in path if cat.startswith('c'))
            return c * (d ** (c_count - 1))
        else:
            return d  # Default DPO value for category 'd'

    # Process the reference segment first
    for i in range(len(reference_segment) - 1):
        edge = tuple(sorted((reference_segment[i], reference_segment[i+1])))
        if edge in edge_info:
            edge_data = edge_info[edge]
            category = 'a0'  # Reference segment is always category 'a'
            dpo_value = 1  # First 'a' segment always has a DPO value of 1
            edge_categories[edge] = {
                'category': category,
                'dpo_value': dpo_value,
                'path': [category],
                'angle': edge_data['angle']
            }
            fused_bonds = get_fused_bonds_between_rings(mol, reference_segment[i], reference_segment[i+1], bond_rings)
            for bond_idx in fused_bonds:
                if bond_idx not in assigned_bonds:
                    DPO_total += dpo_value
                    assigned_bonds.add(bond_idx)
            print(f"Processed reference edge {edge}, category: {category}, path: [{category}], DPO_value: {dpo_value}, DPO_total: {DPO_total}")
    
    # Initialize BFS queue with nodes from the reference segment
    queue = deque([(node, ['a0'], reference_segment.index(node)) for node in reference_segment])
    visited_nodes = set(reference_segment)
    
    while queue:
        current_node, path, path_index = queue.popleft()
        
        for neighbor in G.neighbors(current_node):
            if neighbor in visited_nodes:
                continue
            
            visited_nodes.add(neighbor)
            edge = tuple(sorted((current_node, neighbor)))
            
            if edge not in edge_info:
                continue
            
            edge_data = edge_info[edge]
            
            # Calculate path direction
            print("visited_nodes",visited_nodes,"path_index",path_index)
            if path_index > 0:
                prev_node = reference_segment[path_index - 1]
                path_direction = ring_centers[current_node] - ring_centers[prev_node]
            else:
                next_node = reference_segment[path_index + 1]
                path_direction = ring_centers[next_node] - ring_centers[current_node]
            
            # Calculate edge direction
            edge_direction = ring_centers[neighbor] - ring_centers[current_node]
            
            base_category, edge_data['angle'] = get_base_category(edge_data['angle'], path_direction, edge_direction)
            
            if base_category in ['b', 'c']:
                category = f"{base_category}{path.count('b') + path.count('c')}"
            else:
                category = f"{base_category}{path.count('a')}"
            
            new_path = path + [category]
            dpo_value = calculate_dpo_value(category, new_path, a, b, c, d)
            
            edge_categories[edge] = {
                'category': category,
                'dpo_value': dpo_value,
                'path': new_path,
                'angle': edge_data['angle']
            }
            
            fused_bonds = get_fused_bonds_between_rings(mol, current_node, neighbor, bond_rings)
            for bond_idx in fused_bonds:
                if bond_idx not in assigned_bonds:
                    DPO_total += dpo_value
                    assigned_bonds.add(bond_idx)
            
            queue.append((neighbor, new_path, -1))  # -1 indicates it's not part of the reference segment
            
            print(f"Processed edge {edge}, category: {category}, path: {new_path}, angle: {edge_data['angle']}, DPO_value: {dpo_value}, DPO_total: {DPO_total}")
    
    # Check for unprocessed edges
    unprocessed_edges = set(edge_info.keys()) - set(edge_categories.keys())
    if unprocessed_edges:
        print(f"Warning: {len(unprocessed_edges)} edges were not processed: {unprocessed_edges}")
        # Process unconnected edges
        for edge in unprocessed_edges:
            edge_data = edge_info[edge]
            category, edge_data['angle'] = get_base_category(edge_data['angle']) # No direction information for unprocessed edges
            dpo_value = calculate_dpo_value(category, [category], a, b, c, d)
            edge_categories[edge] = {
                'category': category,
                'dpo_value': dpo_value,
                'path': [category],
                'angle': edge_data['angle']
            }
            print(f"Processed unconnected edge {edge}, category: {category}, path: [{category}], DPO_value: {dpo_value}")
    

    return DPO_total, assigned_bonds, edge_categories
def assign_dpo_to_angulated_segments16(mol, G, reference_segment, edge_info, ring_centers, a, b, c, d):
    DPO_total = 0.0
    assigned_bonds = set()
    edge_categories = {}
    ring_info = mol.GetRingInfo()
    bond_rings = ring_info.BondRings()
    
    def get_base_category(angle, path_direction, edge_direction):
        # Adjust angle based on path and edge directions
        if np.dot(path_direction, edge_direction) < 0:
            angle = 180 - angle  # Flip the angle if directions are opposite
            print("flipped angle",angle,path_direction,edge_direction)
        
        if abs(angle - 60) <= 15:
            return 'b', angle
        elif abs(angle - 120) <= 15:
            return 'c', angle
        elif abs(angle - 0) <= 15 or abs(angle - 180) <= 15:
            return 'a', angle
        else:
            return 'd', angle  # Default category for angles that don't fit a, b, or c
    
    def calculate_dpo_value(category, path, a, b, c, d):
        if category.startswith('a'):
            # Count 'a' segments, resetting after each 'b' or 'c'
            a_count = 0
            for cat in reversed(path):
                if cat.startswith('a'):
                    a_count += 1
                elif cat.startswith('b') or cat.startswith('c'):
                    break
            
            # Count 'b' and 'c' segments
            bc_count = sum(1 for cat in path if cat.startswith('b') or cat.startswith('c'))
            
            return (1 - (a_count - 1) * a) * (d ** bc_count)
        elif category.startswith('b'):
            b_count = sum(1 for cat in path if cat.startswith('b'))
            return b * (d ** (b_count - 1))
        elif category.startswith('c'):
            c_count = sum(1 for cat in path if cat.startswith('c'))
            return c * (d ** (c_count - 1))
        else:
            return d  # Default DPO value for category 'd'
    
    # Process the reference segment first
    for i in range(len(reference_segment) - 1):
        current_node = reference_segment[i]
        next_node = reference_segment[i + 1]
        edge = tuple(sorted((current_node, next_node)))
        if edge in edge_info:
            edge_data = edge_info[edge]
            category = 'a0'  # Reference segment is always category 'a'
            dpo_value = 1  # First 'a' segment always has a DPO value of 1
            edge_categories[edge] = {
                'category': category,
                'dpo_value': dpo_value,
                'path': [category],
                'angle': edge_data['angle']
            }
            fused_bonds = get_fused_bonds_between_rings(mol, current_node, next_node, bond_rings)
            for bond_idx in fused_bonds:
                if bond_idx not in assigned_bonds:
                    DPO_total += dpo_value
                    assigned_bonds.add(bond_idx)
            print(f"Processed reference edge {edge}, category: {category}, path: [{category}], DPO_value: {dpo_value}, DPO_total: {DPO_total}")
    
    # Initialize BFS queue with nodes from the reference segment
    queue = deque([(node, ['a0'], i) for i, node in enumerate(reference_segment)])
    visited_nodes = set(reference_segment)
    
    while queue:
        current_node, path, path_index = queue.popleft()
        print
        for neighbor in G.neighbors(current_node):
            print("current_node",current_node,"path",path,"path_index",path_index,"neighbor",neighbor)
            if neighbor in visited_nodes:
                continue
            
            visited_nodes.add(neighbor)
            edge = tuple(sorted((current_node, neighbor)))
            
            if edge not in edge_info:
                continue
            
            edge_data = edge_info[edge]
            
            # Calculate path direction
            if path_index > 0:
                prev_node = reference_segment[path_index - 1]
                path_direction = ring_centers[current_node] - ring_centers[prev_node]
            else:
                next_node = reference_segment[path_index + 1]
                path_direction = ring_centers[next_node] - ring_centers[current_node]
            
            # Calculate edge direction
            edge_direction = ring_centers[neighbor] - ring_centers[current_node]
            
            base_category, adjusted_angle = get_base_category(edge_data['angle'], path_direction, edge_direction)
            
            if base_category in ['b', 'c']:
                category = f"{base_category}{path.count('b') + path.count('c')}"
            else:
                category = f"{base_category}{path.count('a')}"
            
            new_path = path + [category]
            dpo_value = calculate_dpo_value(category, new_path, a, b, c, d)
            
            edge_categories[edge] = {
                'category': category,
                'dpo_value': dpo_value,
                'path': new_path,
                'angle': adjusted_angle
            }
            
            fused_bonds = get_fused_bonds_between_rings(mol, current_node, neighbor, bond_rings)
            for bond_idx in fused_bonds:
                if bond_idx not in assigned_bonds:
                    DPO_total += dpo_value
                    assigned_bonds.add(bond_idx)
            
            queue.append((neighbor, new_path, -1))  # -1 indicates it's not part of the reference segment
            
            print(f"Processed edge {edge}, category: {category}, path: {new_path}, angle: {adjusted_angle}, DPO_value: {dpo_value}, DPO_total: {DPO_total}")
    
    # ... (keep the existing code for unprocessed edges)
    # Check for unprocessed edges
    unprocessed_edges = set(edge_info.keys()) - set(edge_categories.keys())
    if unprocessed_edges:
        print(f"Warning: {len(unprocessed_edges)} edges were not processed: {unprocessed_edges}")
        # Process unconnected edges
        for edge in unprocessed_edges:
            edge_data = edge_info[edge]
            category, edge_data['angle'] = get_base_category(edge_data['angle']) # No direction information for unprocessed edges
            dpo_value = calculate_dpo_value(category, [category], a, b, c, d)
            edge_categories[edge] = {
                'category': category,
                'dpo_value': dpo_value,
                'path': [category],
                'angle': edge_data['angle']
            }
            print(f"Processed unconnected edge {edge}, category: {category}, path: [{category}], DPO_value: {dpo_value}")

    
    return DPO_total, assigned_bonds, edge_categories
def assign_dpo_to_angulated_segments17bad(mol, G, reference_segment, edge_info, ring_centers, a, b, c, d):
    DPO_total = 0.0
    assigned_bonds = set()
    edge_categories = {}
    ring_info = mol.GetRingInfo()
    bond_rings = ring_info.BondRings()

    def get_base_category(angle, path_direction, edge_direction):
        dot_product = np.dot(path_direction, edge_direction)
        angle = np.degrees(np.arccos(np.clip(dot_product / (np.linalg.norm(path_direction) * np.linalg.norm(edge_direction)), -1.0, 1.0)))
        
        if abs(angle - 60) <= 15:
            return 'b', angle
        elif abs(angle - 120) <= 15:
            return 'c', angle
        elif abs(angle - 0) <= 15 or abs(angle - 180) <= 15:
            return 'a', angle
        else:
            return 'd', angle

    def calculate_dpo_value(category, path, a, b, c, d):
        if category.startswith('a'):
            # Count 'a' segments, resetting after each 'b' or 'c'
            a_count = 0
            for cat in reversed(path):
                if cat.startswith('a'):
                    a_count += 1
                elif cat.startswith('b') or cat.startswith('c'):
                    break
            
            # Count 'b' and 'c' segments
            bc_count = sum(1 for cat in path if cat.startswith('b') or cat.startswith('c'))
            
            return (1 - (a_count - 1) * a) * (d ** bc_count)
        elif category.startswith('b'):
            b_count = sum(1 for cat in path if cat.startswith('b'))
            return b * (d ** (b_count - 1))
        elif category.startswith('c'):
            c_count = sum(1 for cat in path if cat.startswith('c'))
            return c * (d ** (c_count - 1))
        else:
            return d  # Default DPO value for category 'd'

    # Find a terminal node to start from
    terminal_nodes = [node for node in G.nodes() if G.degree(node) == 1]
    if not terminal_nodes:
        raise ValueError("No terminal nodes found in the graph.")
    start_node = terminal_nodes[0]

    # Create a directed tree from the start node
    tree = nx.bfs_tree(G, start_node)

    # Process the reference segment first
    for i in range(len(reference_segment) - 1):
        current_node = reference_segment[i]
        next_node = reference_segment[i + 1]
        edge = (current_node, next_node)
        if edge in edge_info:
            edge_data = edge_info[edge]
            category = 'a0'
            dpo_value = 1
            edge_categories[edge] = {
                'category': category,
                'dpo_value': dpo_value,
                'path': [category],
                'angle': edge_data['angle']
            }
            fused_bonds = get_fused_bonds_between_rings(mol, current_node, next_node, bond_rings)
            for bond_idx in fused_bonds:
                if bond_idx not in assigned_bonds:
                    DPO_total += dpo_value
                    assigned_bonds.add(bond_idx)
            print(f"Processed reference edge {edge}, category: {category}, path: [{category}], DPO_value: {dpo_value}, DPO_total: {DPO_total}")

    # Process the entire tree
    for edge in nx.bfs_edges(tree, start_node):
        current_node, neighbor = edge
        if edge not in edge_info:
            continue

        edge_data = edge_info[edge]

        # Calculate path direction (from parent to current node)
        parent = next(tree.predecessors(current_node), None)
        if parent is not None:
            path_direction = ring_centers[current_node] - ring_centers[parent]
        else:
            # For the start node, use the direction to its first child
            child = next(tree.successors(current_node))
            path_direction = ring_centers[child] - ring_centers[current_node]

        # Calculate edge direction
        edge_direction = ring_centers[neighbor] - ring_centers[current_node]

        base_category, adjusted_angle = get_base_category(edge_data['angle'], path_direction, edge_direction)

        # Determine the path to this edge
        path_to_edge = nx.shortest_path(tree, start_node, current_node)
        path = [edge_categories.get((path_to_edge[i], path_to_edge[i+1]), {'category': 'a0'})['category'] for i in range(len(path_to_edge)-1)]

        if base_category in ['b', 'c']:
            category = f"{base_category}{path.count('b') + path.count('c')}"
        else:
            category = f"{base_category}{path.count('a')}"

        new_path = path + [category]
        dpo_value = calculate_dpo_value(category, new_path, a, b, c, d)

        edge_categories[edge] = {
            'category': category,
            'dpo_value': dpo_value,
            'path': new_path,
            'angle': adjusted_angle
        }

        fused_bonds = get_fused_bonds_between_rings(mol, current_node, neighbor, bond_rings)
        for bond_idx in fused_bonds:
            if bond_idx not in assigned_bonds:
                DPO_total += dpo_value
                assigned_bonds.add(bond_idx)

        print(f"Processed edge {edge}, category: {category}, path: {new_path}, angle: {adjusted_angle}, DPO_value: {dpo_value}, DPO_total: {DPO_total}")
   

    # Print the entire graph path
    print("\nEntire Graph Path:")
    for edge in nx.bfs_edges(tree, start_node):
        if edge in edge_categories:
            print(f"Edge: {edge}, Category: {edge_categories[edge]['category']}, Path: {edge_categories[edge]['path']}, Angle: {edge_categories[edge]['angle']:.2f}")
        else:
            print(f"Edge: {edge}, Not processed (not in edge_info)")

    # Check for edges in edge_info that were not processed
    unprocessed_edges = set(edge_info.keys()) - set(edge_categories.keys())
    if unprocessed_edges:
        print("\nUnprocessed edges:")
        for edge in unprocessed_edges:
            print(f"Edge: {edge}, Angle: {edge_info[edge]['angle']:.2f}")

    return DPO_total, assigned_bonds, edge_categories
def assign_dpo_to_angulated_segments18MissingUnprocessed(mol, G, reference_segment, edge_info, ring_centers, a, b, c, d):
    DPO_total = 0.0
    assigned_bonds = set()
    edge_categories = {}
    ring_info = mol.GetRingInfo()
    bond_rings = ring_info.BondRings()

    def get_base_category(angle, reference_direction, edge_direction):
        dot_product = np.dot(reference_direction, edge_direction)
        angle = np.degrees(np.arccos(np.clip(dot_product / (np.linalg.norm(reference_direction) * np.linalg.norm(edge_direction)), -1.0, 1.0)))
        
        if abs(angle - 60) <= 15:
            return 'b', angle
        elif abs(angle - 120) <= 15:
            return 'c', angle
        elif abs(angle - 0) <= 15 or abs(angle - 180) <= 15:
            return 'a', angle
        else:
            return 'd', angle

    def calculate_dpo_value(category, path, a, b, c, d):
        if category.startswith('a'):
            # Count 'a' segments, resetting after each 'b' or 'c'
            a_count = 0
            for cat in reversed(path):
                if cat.startswith('a'):
                    a_count += 1
                elif cat.startswith('b') or cat.startswith('c'):
                    break
            
            # Count 'b' and 'c' segments
            bc_count = sum(1 for cat in path if cat.startswith('b') or cat.startswith('c'))
            
            return (1 - (a_count - 1) * a) * (d ** bc_count)
        elif category.startswith('b'):
            b_count = sum(1 for cat in path if cat.startswith('b'))
            return b * (d ** (b_count - 1))
        elif category.startswith('c'):
            c_count = sum(1 for cat in path if cat.startswith('c'))
            return c * (d ** (c_count - 1))
        else:
            return d  # Default DPO value for category 'd'

    # Find a terminal node to start from
    terminal_nodes = [node for node in G.nodes() if G.degree(node) == 1]
    if not terminal_nodes:
        raise ValueError("No terminal nodes found in the graph.")
    start_node = terminal_nodes[0]

    # Create a directed tree from the start node
    tree = nx.bfs_tree(G, start_node)

    # Calculate reference segment direction
    ref_start = reference_segment[0]
    ref_end = reference_segment[-1]
    reference_direction = ring_centers[ref_end] - ring_centers[ref_start]

    # Process the reference segment first
    for i in range(len(reference_segment) - 1):
        current_node = reference_segment[i]
        next_node = reference_segment[i + 1]
        edge = (current_node, next_node)
        if edge in edge_info:
            edge_data = edge_info[edge]
            category = 'a0'
            dpo_value = 1
            edge_categories[edge] = {
                'category': category,
                'dpo_value': dpo_value,
                'path': [category],
                'angle': 0  # Angle with reference segment is 0
            }
            fused_bonds = get_fused_bonds_between_rings(mol, current_node, next_node, bond_rings)
            for bond_idx in fused_bonds:
                if bond_idx not in assigned_bonds:
                    DPO_total += dpo_value
                    assigned_bonds.add(bond_idx)
            print(f"Processed reference edge {edge}, category: {category}, path: [{category}], DPO_value: {dpo_value}, DPO_total: {DPO_total}")

    # Process the entire tree
    for edge in nx.bfs_edges(tree, start_node):
        current_node, neighbor = edge
        if edge not in edge_info:
            continue

        edge_data = edge_info[edge]

        # Calculate edge direction
        edge_direction = ring_centers[neighbor] - ring_centers[current_node]

        base_category, adjusted_angle = get_base_category(edge_data['angle'], reference_direction, edge_direction)

        # Determine the path to this edge
        path_to_edge = nx.shortest_path(tree, start_node, current_node)
        path = [edge_categories.get((path_to_edge[i], path_to_edge[i+1]), {'category': 'a0'})['category'] for i in range(len(path_to_edge)-1)]

        if base_category in ['b', 'c']:
            category = f"{base_category}{path.count('b') + path.count('c')}"
        else:
            category = f"{base_category}{path.count('a')}"

        new_path = path + [category]
        dpo_value = calculate_dpo_value(category, new_path, a, b, c, d)

        edge_categories[edge] = {
            'category': category,
            'dpo_value': dpo_value,
            'path': new_path,
            'angle': adjusted_angle
        }

        fused_bonds = get_fused_bonds_between_rings(mol, current_node, neighbor, bond_rings)
        for bond_idx in fused_bonds:
            if bond_idx not in assigned_bonds:
                DPO_total += dpo_value
                assigned_bonds.add(bond_idx)

        print(f"Processed edge {edge}, category: {category}, path: {new_path}, angle: {adjusted_angle}, DPO_value: {dpo_value}, DPO_total: {DPO_total}")

    # Print the entire graph path
    print("\nEntire Graph Path:")
    for edge in nx.bfs_edges(tree, start_node):
        if edge in edge_categories:
            print(f"Edge: {edge}, Category: {edge_categories[edge]['category']}, Path: {edge_categories[edge]['path']}, Angle: {edge_categories[edge]['angle']:.2f}")
        else:
            print(f"Edge: {edge}, Not processed (not in edge_info)")

    # Check for edges in edge_info that were not processed
    unprocessed_edges = set(edge_info.keys()) - set(edge_categories.keys())
    if unprocessed_edges:
        print("\nUnprocessed edges:")
        for edge in unprocessed_edges:
            print(f"Edge: {edge}, Angle: {edge_info[edge]['angle']:.2f}")
    

    return DPO_total, assigned_bonds, edge_categories
def assign_dpo_to_angulated_segments19WrongDirectionOnHalf(mol, G, reference_segment, edge_info, ring_centers, a, b, c, d):
    DPO_total = 0.0
    assigned_bonds = set()
    edge_categories = {}
    ring_info = mol.GetRingInfo()
    bond_rings = ring_info.BondRings()

    def get_base_category(angle, reference_direction, edge_direction):
        dot_product = np.dot(reference_direction, edge_direction)
        angle = np.degrees(np.arccos(np.clip(dot_product / (np.linalg.norm(reference_direction) * np.linalg.norm(edge_direction)), -1.0, 1.0)))
        
        if abs(angle - 60) <= 15:
            return 'b', angle
        elif abs(angle - 120) <= 15:
            return 'c', angle
        elif abs(angle - 0) <= 15 or abs(angle - 180) <= 15:
            return 'a', angle
        else:
            return 'd', angle

    def calculate_dpo_value(category, path, a, b, c, d):
        if category.startswith('a'):
            # Count 'a' segments, resetting after each 'b' or 'c'
            a_count = 0
            for cat in reversed(path):
                if cat.startswith('a'):
                    a_count += 1
                elif cat.startswith('b') or cat.startswith('c'):
                    break
            
            # Count 'b' and 'c' segments
            bc_count = sum(1 for cat in path if cat.startswith('b') or cat.startswith('c'))
            
            return (1 - (a_count - 1) * a) * (d ** bc_count)
        elif category.startswith('b'):
            b_count = sum(1 for cat in path if cat.startswith('b'))
            return b * (d ** (b_count - 1))
        elif category.startswith('c'):
            c_count = sum(1 for cat in path if cat.startswith('c'))
            return c * (d ** (c_count - 1))
        else:
            return d  # Default DPO value for category 'd'


    # Find a terminal node to start from
    terminal_nodes = [node for node in G.nodes() if G.degree(node) == 1]
    if not terminal_nodes:
        raise ValueError("No terminal nodes found in the graph.")
    start_node = terminal_nodes[0]

    # Create a directed tree from the start node
    tree = nx.bfs_tree(G, start_node)

    # Calculate reference segment direction
    ref_start = reference_segment[0]
    ref_end = reference_segment[-1]
    reference_direction = ring_centers[ref_end] - ring_centers[ref_start]

    # Process the reference segment first
    for i in range(len(reference_segment) - 1):
        current_node = reference_segment[i]
        next_node = reference_segment[i + 1]
        edge = (current_node, next_node)
        if edge in edge_info:
            edge_data = edge_info[edge]
            category = 'a0'
            dpo_value = 1
            edge_categories[edge] = {
                'category': category,
                'dpo_value': dpo_value,
                'path': [category],
                'angle': 0  # Angle with reference segment is 0
            }
            fused_bonds = get_fused_bonds_between_rings(mol, current_node, next_node, bond_rings)
            for bond_idx in fused_bonds:
                if bond_idx not in assigned_bonds:
                    DPO_total += dpo_value
                    assigned_bonds.add(bond_idx)
            print(f"Processed reference edge {edge}, category: {category}, path: [{category}], DPO_value: {dpo_value}, DPO_total: {DPO_total}")

    # Process the entire tree
    for edge in nx.bfs_edges(tree, start_node):
        current_node, neighbor = edge
        if edge not in edge_info:
            continue

        edge_data = edge_info[edge]

        # Calculate edge direction
        edge_direction = ring_centers[neighbor] - ring_centers[current_node]

        base_category, adjusted_angle = get_base_category(edge_data['angle'], reference_direction, edge_direction)

        # Determine the path to this edge
        path_to_edge = nx.shortest_path(tree, start_node, current_node)
        path = [edge_categories.get((path_to_edge[i], path_to_edge[i+1]), {'category': 'a0'})['category'] for i in range(len(path_to_edge)-1)]

        if base_category in ['b', 'c']:
            category = f"{base_category}{path.count('b') + path.count('c')}"
        else:
            category = f"{base_category}{path.count('a')}"

        new_path = path + [category]
        dpo_value = calculate_dpo_value(category, new_path, a, b, c, d)

        edge_categories[edge] = {
            'category': category,
            'dpo_value': dpo_value,
            'path': new_path,
            'angle': adjusted_angle
        }

        fused_bonds = get_fused_bonds_between_rings(mol, current_node, neighbor, bond_rings)
        for bond_idx in fused_bonds:
            if bond_idx not in assigned_bonds:
                DPO_total += dpo_value
                assigned_bonds.add(bond_idx)

        print(f"Processed edge {edge}, category: {category}, path: {new_path}, angle: {adjusted_angle}, DPO_value: {dpo_value}, DPO_total: {DPO_total}")

    # Print the entire graph path
    print("\nEntire Graph Path:")
    for edge in nx.bfs_edges(tree, start_node):
        if edge in edge_categories:
            print(f"Edge: {edge}, Category: {edge_categories[edge]['category']}, Path: {edge_categories[edge]['path']}, Angle: {edge_categories[edge]['angle']:.2f}")
        else:
            print(f"Edge: {edge}, Not processed (not in edge_info)")

    # Process unprocessed edges
    unprocessed_edges = set(edge_info.keys()) - set(edge_categories.keys())
    for edge in unprocessed_edges:
        current_node, neighbor = edge
        edge_data = edge_info[edge]

        # Calculate edge direction
        edge_direction = ring_centers[neighbor] - ring_centers[current_node]

        base_category, adjusted_angle = get_base_category(edge_data['angle'], reference_direction, edge_direction)

        # Determine the path to this edge
        try:
            path_to_edge = nx.shortest_path(tree, start_node, current_node)
            path = [edge_categories.get((path_to_edge[i], path_to_edge[i+1]), {'category': 'a0'})['category'] for i in range(len(path_to_edge)-1)]
        except nx.NetworkXNoPath:
            # If there's no path in the tree, start a new path
            path = []

        if base_category in ['b', 'c']:
            category = f"{base_category}{path.count('b') + path.count('c')}"
        else:
            category = f"{base_category}{path.count('a')}"

        new_path = path + [category]
        dpo_value = calculate_dpo_value(category, new_path, a, b, c, d)

        edge_categories[edge] = {
            'category': category,
            'dpo_value': dpo_value,
            'path': new_path,
            'angle': adjusted_angle
        }

        fused_bonds = get_fused_bonds_between_rings(mol, current_node, neighbor, bond_rings)
        for bond_idx in fused_bonds:
            if bond_idx not in assigned_bonds:
                DPO_total += dpo_value
                assigned_bonds.add(bond_idx)

        print(f"Processed unprocessed edge {edge}, category: {category}, path: {new_path}, angle: {adjusted_angle}, DPO_value: {dpo_value}, DPO_total: {DPO_total}")

    # Print the entire graph path
    print("\nEntire Graph Path:")
    for edge in edge_categories:
        print(f"Edge: {edge}, Category: {edge_categories[edge]['category']}, Path: {edge_categories[edge]['path']}, Angle: {edge_categories[edge]['angle']:.2f}")

    return DPO_total, assigned_bonds, edge_categories
def assign_dpo_to_angulated_segments20RareCaseWorksFailsAgainOnOthers(mol, G, reference_segment, edge_info, ring_centers, a, b, c, d):
    DPO_total = 0.0
    assigned_bonds = set()
    edge_categories = {}
    ring_info = mol.GetRingInfo()
    bond_rings = ring_info.BondRings()

    def get_base_category(angle, reference_direction, edge_direction):
        dot_product = np.dot(reference_direction, edge_direction)
        angle = np.degrees(np.arccos(np.clip(dot_product / (np.linalg.norm(reference_direction) * np.linalg.norm(edge_direction)), -1.0, 1.0)))
        
        if abs(angle - 60) <= 15:
            return 'b', angle
        elif abs(angle - 120) <= 15:
            return 'c', angle
        elif abs(angle - 0) <= 15 or abs(angle - 180) <= 15:
            return 'a', angle
        else:
            return 'd', angle

    
    def calculate_dpo_value(category, path, a, b, c, d):
        if category.startswith('a'):
            # Count 'a' segments, resetting after each 'b' or 'c'
            a_count = 0
            for cat in reversed(path):
                if cat.startswith('a'):
                    a_count += 1
                elif cat.startswith('b') or cat.startswith('c'):
                    break
            
            # Count 'b' and 'c' segments
            bc_count = sum(1 for cat in path if cat.startswith('b') or cat.startswith('c'))
            
            return (1 - (a_count - 1) * a) * (d ** bc_count)
        elif category.startswith('b'):
            b_count = sum(1 for cat in path if cat.startswith('b'))
            return b * (d ** (b_count - 1))
        elif category.startswith('c'):
            c_count = sum(1 for cat in path if cat.startswith('c'))
            return c * (d ** (c_count - 1))
        else:
            return d  # Default DPO value for category 'd'



    # Find a terminal node to start from
    terminal_nodes = [node for node in G.nodes() if G.degree(node) == 1]
    if not terminal_nodes:
        raise ValueError("No terminal nodes found in the graph.")
    start_node = terminal_nodes[0]

    # Create a directed tree from the start node
    tree = nx.bfs_tree(G, start_node)

    # Calculate reference segment direction
    ref_start = reference_segment[0]
    ref_end = reference_segment[-1]
    reference_direction = ring_centers[ref_end] - ring_centers[ref_start]

    # Process the reference segment first
    for i in range(len(reference_segment) - 1):
        current_node = reference_segment[i]
        next_node = reference_segment[i + 1]
        edge = (current_node, next_node)
        if edge in edge_info:
            edge_data = edge_info[edge]
            category = 'a0'
            dpo_value = 1
            edge_categories[edge] = {
                'category': category,
                'dpo_value': dpo_value,
                'path': [category],
                'angle': 0  # Angle with reference segment is 0
            }
            fused_bonds = get_fused_bonds_between_rings(mol, current_node, next_node, bond_rings)
            for bond_idx in fused_bonds:
                if bond_idx not in assigned_bonds:
                    DPO_total += dpo_value
                    assigned_bonds.add(bond_idx)
            print(f"Processed reference edge {edge}, category: {category}, path: [{category}], DPO_value: {dpo_value}, DPO_total: {DPO_total}")

    # Process the entire tree
    for edge in nx.bfs_edges(tree, start_node):
        current_node, neighbor = edge
        if edge not in edge_info:
            continue

        edge_data = edge_info[edge]

        # Calculate edge direction
        edge_direction = ring_centers[neighbor] - ring_centers[current_node]

        base_category, adjusted_angle = get_base_category(edge_data['angle'], reference_direction, edge_direction)

        # Determine the path to this edge
        path_to_edge = nx.shortest_path(tree, start_node, current_node)
        path = [edge_categories.get((path_to_edge[i], path_to_edge[i+1]), {'category': 'a0'})['category'] for i in range(len(path_to_edge)-1)]

        if base_category in ['b', 'c']:
            category = f"{base_category}{path.count('b') + path.count('c')}"
        else:
            category = f"{base_category}{path.count('a')}"

        new_path = path + [category]
        dpo_value = calculate_dpo_value(category, new_path, a, b, c, d)

        edge_categories[edge] = {
            'category': category,
            'dpo_value': dpo_value,
            'path': new_path,
            'angle': adjusted_angle
        }

        fused_bonds = get_fused_bonds_between_rings(mol, current_node, neighbor, bond_rings)
        for bond_idx in fused_bonds:
            if bond_idx not in assigned_bonds:
                DPO_total += dpo_value
                assigned_bonds.add(bond_idx)

        print(f"Processed edge {edge}, category: {category}, path: {new_path}, angle: {adjusted_angle}, DPO_value: {dpo_value}, DPO_total: {DPO_total}")


    # Process unprocessed edges
    unprocessed_edges = set(edge_info.keys()) - set(edge_categories.keys())
    for edge in unprocessed_edges:
        current_node, neighbor = edge
        edge_data = edge_info[edge]

        # Calculate edge direction
        edge_direction = ring_centers[neighbor] - ring_centers[current_node]

        # Check directionality
        if np.dot(reference_direction, edge_direction) < 0:
            # If the edge direction is opposite to the reference direction, swap the nodes
            current_node, neighbor = neighbor, current_node
            edge_direction = -edge_direction

        base_category, adjusted_angle = get_base_category(edge_data['angle'], reference_direction, edge_direction)

        # Determine the path to this edge
        try:
            path_to_edge = nx.shortest_path(tree, start_node, current_node)
            path = [edge_categories.get((path_to_edge[i], path_to_edge[i+1]), {'category': 'a0'})['category'] for i in range(len(path_to_edge)-1)]
        except nx.NetworkXNoPath:
            # If there's no path in the tree, start a new path
            path = []

        if base_category in ['b', 'c']:
            category = f"{base_category}{path.count('b') + path.count('c')}"
        else:
            category = f"{base_category}{path.count('a')}"

        new_path = path + [category]
        dpo_value = calculate_dpo_value(category, new_path, a, b, c, d)

        edge_categories[edge] = {
            'category': category,
            'dpo_value': dpo_value,
            'path': new_path,
            'angle': adjusted_angle
        }

        fused_bonds = get_fused_bonds_between_rings(mol, current_node, neighbor, bond_rings)
        for bond_idx in fused_bonds:
            if bond_idx not in assigned_bonds:
                DPO_total += dpo_value
                assigned_bonds.add(bond_idx)

        print(f"Processed unprocessed edge {edge}, category: {category}, path: {new_path}, angle: {adjusted_angle}, DPO_value: {dpo_value}, DPO_total: {DPO_total}")

    # Calculate DPO_total based on edge_categories
    DPO_total = sum(info['dpo_value'] for info in edge_categories.values())

    # # Print the entire graph path
    # print("\nEntire Graph Path:")
    # for edge in edge_categories:
    #     print(f"Edge: {edge}, Category: {edge_categories[edge]['category']}, Path: {edge_categories[edge]['path']}, Angle: {edge_categories[edge]['angle']:.2f}")

    # # Print information about the graph structure
    # print("\nGraph Structure Information:")
    # print(f"Total nodes in G: {G.number_of_nodes()}")
    # print(f"Total edges in G: {G.number_of_edges()}")
    # print(f"Total edges in tree: {tree.number_of_edges()}")
    # print(f"Total edges in edge_info: {len(edge_info)}")
    # print(f"Total edges processed: {len(edge_categories)}")
    # print(f"Unprocessed edges: {len(unprocessed_edges)}")

    return DPO_total, assigned_bonds, edge_categories
def assign_dpo_to_angulated_segments21GetsSomeVectorsWrongIBelieve(mol, G, reference_segment, edge_info, ring_centers, a, b, c, d):
    DPO_total = 0.0
    assigned_bonds = set()
    edge_categories = {}
    ring_info = mol.GetRingInfo()
    bond_rings = ring_info.BondRings()

    def get_base_category(angle, reference_direction, edge_direction):
        #dot_product = np.dot(reference_direction, edge_direction)
        #angle = np.degrees(np.arccos(np.clip(dot_product / (np.linalg.norm(reference_direction) * np.linalg.norm(edge_direction)), -1.0, 1.0)))
        
        if abs(angle - 60) <= 15:
            return 'b', angle
        elif abs(angle - 120) <= 15:
            return 'c', angle
        elif abs(angle - 0) <= 15 or abs(angle - 180) <= 15:
            return 'a', angle
        else:
            return 'd', angle
        

    # def calculate_dpo_value(category, path, a, b, c, d):
    #     print("category",category,"what is happening here")
    #     if category.startswith('a'):
    #         return 1 - int(category[1:]) * a
    #     elif category.startswith('b'):
    #         return (1 - int(category[1:]) * b) * d**path.count('d')
    #     elif category.startswith('c'):
    #         return (1 - int(category[1:]) * c) * d**path.count('d')
    #     elif category.startswith('d'):
    #         return d**path.count('d')
    #     else:
    #         return 0
    def calculate_dpo_value(category, path, a, b, c, d):
        if category.startswith('a'):
            # Count 'a' segments, resetting after each 'b' or 'c'
            a_count = 0
            for cat in reversed(path):
                if cat.startswith('a'):
                    a_count += 1
                elif cat.startswith('b') or cat.startswith('c'):
                    break
            
            # Count 'b' and 'c' segments
            bc_count = sum(1 for cat in path if cat.startswith('b') or cat.startswith('c'))
            
            return (1 - (a_count - 1) * a) * (d ** bc_count)
        elif category.startswith('b'):
            b_count = sum(1 for cat in path if cat.startswith('b'))
            return b * (d ** (b_count - 1))
        elif category.startswith('c'):
            c_count = sum(1 for cat in path if cat.startswith('c'))
            return c * (d ** (c_count - 1))
        else:
            return d  # Default DPO value for category 'd'

    # Calculate reference segment direction
    ref_start = reference_segment[0]
    ref_end = reference_segment[-1]
    reference_direction = ring_centers[ref_end] - ring_centers[ref_start]

    # Process the reference segment first, in the given order
    for i in range(len(reference_segment) - 1):
        current_node = reference_segment[i]
        next_node = reference_segment[i + 1]
        edge = (current_node, next_node)
        reverse_edge = (next_node, current_node)
        
        if edge in edge_info:
            use_edge = edge
        elif reverse_edge in edge_info:
            use_edge = reverse_edge
        else:
            print(f"Warning: Edge {edge} not found in edge_info")
            continue

        category = f'a{i}'
        dpo_value = 1 - i * a
        edge_categories[use_edge] = {
            'category': category,
            'dpo_value': dpo_value,
            'path': ['a'] * (i + 1),
            'angle': 0  # Angle with reference segment is 0
        }
        fused_bonds = get_fused_bonds_between_rings(mol, current_node, next_node, bond_rings)
        for bond_idx in fused_bonds:
            if bond_idx not in assigned_bonds:
                DPO_total += dpo_value
                assigned_bonds.add(bond_idx)
        print(f"Processed reference edge {use_edge}, category: {category}, path: {edge_categories[use_edge]['path']}, DPO_value: {dpo_value}, DPO_total: {DPO_total}")

    # Process edges outward from the reference segment
    queue = deque(reference_segment)
    processed_nodes = set(reference_segment)

    # print("Contents of edge_info:")
    # for edge, info in edge_info.items():
    #     print(f"Edge {edge}: {info}")
    # print_unique_paths(G, reference_segment)
    
    while queue:
        current_node = queue.popleft()
        
        for neighbor in G.neighbors(current_node):
            if neighbor in processed_nodes:
                continue
            
            edge = (current_node, neighbor)
            reverse_edge = (neighbor, current_node)
            
            
            if edge in edge_info:
                use_edge = edge
            elif reverse_edge in edge_info:
                use_edge = reverse_edge
            else:
                print(f"Warning: Edge {edge} not found in edge_info")
                continue
            
            edge_data = edge_info[use_edge]
            edge_direction = ring_centers[neighbor] - ring_centers[current_node]
            
            # Determine the path to this edge
            path_to_edge = nx.shortest_path(G, reference_segment[0], current_node)
            path = [edge_categories.get((path_to_edge[i], path_to_edge[i+1]), 
                    edge_categories.get((path_to_edge[i+1], path_to_edge[i]), {'category': 'a0'}))['category'] 
                    for i in range(len(path_to_edge)-1)]
            
            # Calculate edge direction aligned with the path
            path_direction = ring_centers[path_to_edge[-1]] - ring_centers[path_to_edge[0]]
            edge_direction = align_vector_with_path(ring_centers[neighbor] - ring_centers[current_node], path_direction)
            print(edge,reverse_edge,edge_direction,use_edge,edge_data['angle'])

            base_category, adjusted_angle = get_base_category(edge_data['angle'], reference_direction, edge_direction)
                  
        
            if base_category in ['b', 'c']:
                category = f"{base_category}{path.count('b') + path.count('c')}"
            else:
                category = f"{base_category}{path.count('a')}"
            
            new_path = path + [category]
            dpo_value = calculate_dpo_value(category, new_path, a, b, c, d)
            
            edge_categories[use_edge] = {
                'category': category,
                'dpo_value': dpo_value,
                'path': new_path,
                'angle': adjusted_angle
            }
            
            fused_bonds = get_fused_bonds_between_rings(mol, current_node, neighbor, bond_rings)
            for bond_idx in fused_bonds:
                if bond_idx not in assigned_bonds:
                    DPO_total += dpo_value
                    assigned_bonds.add(bond_idx)
            
            print(f"Processed edge {use_edge}, category: {category}, path: {new_path}, angle: {adjusted_angle}, DPO_value: {dpo_value}, DPO_total: {DPO_total}")
            
            processed_nodes.add(neighbor)
            queue.append(neighbor)

    # Check for any unprocessed edges
    unprocessed_edges = set(edge_info.keys()) - set(edge_categories.keys())
    if unprocessed_edges:
        print("Warning: Some edges were not processed in the main loop.")
        for edge in unprocessed_edges:
            print(f"Unprocessed edge: {edge}")

    return DPO_total, assigned_bonds, edge_categories
def assign_dpo_to_angulated_segments22bad(mol, G, reference_segment, edge_info, ring_centers, a, b, c, d):
    DPO_total = 0.0
    assigned_bonds = set()
    edge_categories = {}
    ring_info = mol.GetRingInfo()
    bond_rings = ring_info.BondRings()

    def get_vector(start, end):
        return ring_centers[end] - ring_centers[start]

    def get_base_category(angle):
        if abs(angle - 60) <= 15:
            return 'b', angle
        elif abs(angle - 120) <= 15:
            return 'c', angle
        elif abs(angle - 0) <= 15 or abs(angle - 180) <= 15:
            return 'a', angle
        else:
            return 'd', angle

    def calculate_dpo_value(category, path, a, b, c, d):
        if category.startswith('a'):
            a_count = 0
            for cat in reversed(path):
                if cat.startswith('a'):
                    a_count += 1
                elif cat.startswith('b') or cat.startswith('c'):
                    break
            bc_count = sum(1 for cat in path if cat.startswith('b') or cat.startswith('c'))
            return (1 - (a_count - 1) * a) * (d ** bc_count)
        elif category.startswith('b'):
            b_count = sum(1 for cat in path if cat.startswith('b'))
            return b * (d ** (b_count - 1))
        elif category.startswith('c'):
            c_count = sum(1 for cat in path if cat.startswith('c'))
            return c * (d ** (c_count - 1))
        else:
            return d

    # Calculate reference vector
    ref_vector = get_vector(reference_segment[0], reference_segment[1])

    # Process the reference segment first
    for i in range(len(reference_segment) - 1):
        current_node = reference_segment[i]
        next_node = reference_segment[i + 1]
        edge = (current_node, next_node)

        category = f'a{i}'
        dpo_value = 1 - i * a
        edge_categories[edge] = {
            'category': category,
            'dpo_value': dpo_value,
            'path': ['a'] * (i + 1),
            'angle': 0
        }
        fused_bonds = get_fused_bonds_between_rings(mol, current_node, next_node, bond_rings)
        for bond_idx in fused_bonds:
            if bond_idx not in assigned_bonds:
                DPO_total += dpo_value
                assigned_bonds.add(bond_idx)
        print(f"Processed reference edge {edge}, category: {category}, path: {edge_categories[edge]['path']}, DPO_value: {dpo_value}, DPO_total: {DPO_total}")

    # Process edges outward from the reference segment
    queue = deque(reference_segment)
    processed_nodes = set(reference_segment)

    while queue:
        current_node = queue.popleft()
        
        for neighbor in G.neighbors(current_node):
            if neighbor in processed_nodes:
                continue
            
            edge = tuple(sorted((current_node, neighbor)))
            
            if edge not in edge_info:
                print(f"Warning: Edge {edge} not found in edge_info")
                continue
            
            edge_data = edge_info[edge]
            
            # Determine the correct vector direction
            if current_node in reference_segment and neighbor in reference_segment:
                if reference_segment.index(current_node) > reference_segment.index(neighbor):
                    vector = get_vector(neighbor, current_node)
                else:
                    vector = get_vector(current_node, neighbor)
            elif current_node in reference_segment:
                vector = get_vector(current_node, neighbor)
            elif neighbor in reference_segment:
                vector = get_vector(neighbor, current_node)
            else:
                vector = get_vector(min(current_node, neighbor), max(current_node, neighbor))

            # Calculate angle with reference vector
            cos_angle = np.dot(ref_vector, vector) / (np.linalg.norm(ref_vector) * np.linalg.norm(vector))
            angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
            
            base_category, adjusted_angle = get_base_category(angle)
            
            # Determine the path to this edge
            path_to_edge = nx.shortest_path(G, reference_segment[0], current_node)
            path = [edge_categories.get((path_to_edge[i], path_to_edge[i+1]), 
                    edge_categories.get((path_to_edge[i+1], path_to_edge[i]), {'category': 'a0'}))['category'] 
                    for i in range(len(path_to_edge)-1)]
            
            if base_category in ['b', 'c']:
                category = f"{base_category}{path.count('b') + path.count('c')}"
            else:
                category = f"{base_category}{path.count('a')}"
            
            new_path = path + [category]
            dpo_value = calculate_dpo_value(category, new_path, a, b, c, d)
            
            edge_categories[edge] = {
                'category': category,
                'dpo_value': dpo_value,
                'path': new_path,
                'angle': adjusted_angle
            }
            
            fused_bonds = get_fused_bonds_between_rings(mol, current_node, neighbor, bond_rings)
            for bond_idx in fused_bonds:
                if bond_idx not in assigned_bonds:
                    DPO_total += dpo_value
                    assigned_bonds.add(bond_idx)
            
            print(f"Processed edge {edge}, category: {category}, path: {new_path}, angle: {adjusted_angle}, DPO_value: {dpo_value}, DPO_total: {DPO_total}")
            
            processed_nodes.add(neighbor)
            queue.append(neighbor)

    # Check for any unprocessed edges
    unprocessed_edges = set(edge_info.keys()) - set(edge_categories.keys())
    if unprocessed_edges:
        print("Warning: Some edges were not processed in the main loop.")
        for edge in unprocessed_edges:
            print(f"Unprocessed edge: {edge}")

    return DPO_total, assigned_bonds, edge_categories
def assign_dpo_to_angulated_segments23bad(mol, G, reference_segment, edge_info, ring_centers, a, b, c, d):
    DPO_total = 0.0
    assigned_bonds = set()
    edge_categories = {}
    ring_info = mol.GetRingInfo()
    bond_rings = ring_info.BondRings()

    def get_vector(start, end):
        return ring_centers[end] - ring_centers[start]

    def get_base_category(angle):
        if abs(angle - 60) <= 15:
            return 'b', angle
        elif abs(angle - 120) <= 15:
            return 'c', angle
        elif abs(angle - 0) <= 15 or abs(angle - 180) <= 15:
            return 'a', angle
        else:
            return 'd', angle

    def calculate_dpo_value(category, path, a, b, c, d):
        if category.startswith('a'):
            a_count = 0
            for cat in reversed(path):
                if cat.startswith('a'):
                    a_count += 1
                elif cat.startswith('b') or cat.startswith('c'):
                    break
            bc_count = sum(1 for cat in path if cat.startswith('b') or cat.startswith('c'))
            return (1 - (a_count - 1) * a) * (d ** bc_count)
        elif category.startswith('b'):
            b_count = sum(1 for cat in path if cat.startswith('b'))
            return b * (d ** (b_count - 1))
        elif category.startswith('c'):
            c_count = sum(1 for cat in path if cat.startswith('c'))
            return c * (d ** (c_count - 1))
        else:
            return d

    # Calculate reference vector
    ref_vector = get_vector(reference_segment[0], reference_segment[-1])

    # Process the reference segment first
    for i in range(len(reference_segment) - 1):
        current_node = reference_segment[i]
        next_node = reference_segment[i + 1]
        edge = (current_node, next_node)

        category = f'a{i}'
        dpo_value = 1 - i * a
        edge_categories[edge] = {
            'category': category,
            'dpo_value': dpo_value,
            'path': ['a'] * (i + 1),
            'angle': 0
        }
        fused_bonds = get_fused_bonds_between_rings(mol, current_node, next_node, bond_rings)
        for bond_idx in fused_bonds:
            if bond_idx not in assigned_bonds:
                DPO_total += dpo_value
                assigned_bonds.add(bond_idx)
        print(f"Processed reference edge {edge}, category: {category}, path: {edge_categories[edge]['path']}, DPO_value: {dpo_value}, DPO_total: {DPO_total}")

    # Process edges outward from the reference segment
    queue = deque(reference_segment)
    processed_nodes = set(reference_segment)

    while queue:
        current_node = queue.popleft()
        
        # Get neighbors and filter based on directionality
        for neighbor in G.neighbors(current_node):
            if neighbor in processed_nodes:
                continue
            
            # Determine the correct edge direction
            if current_node in reference_segment:
                # Ensure we are processing edges that go away from the reference segment
                if neighbor not in reference_segment:
                    edge = (current_node, neighbor)
                    reverse_edge = (neighbor, current_node)
                else:
                    continue  # Skip if the neighbor is part of the reference segment
            else:
                edge = (neighbor, current_node)
                reverse_edge = (current_node, neighbor)

            if edge in edge_info:
                use_edge = edge
            elif reverse_edge in edge_info:
                use_edge = reverse_edge
            else:
                print(f"Warning: Edge {edge} not found in edge_info")
                continue
            
            edge_data = edge_info[use_edge]
            
            # Determine the correct vector direction based on the path
            vector = get_vector(current_node, neighbor)

            # Calculate angle with reference vector
            cos_angle = np.dot(ref_vector, vector) / (np.linalg.norm(ref_vector) * np.linalg.norm(vector))
            angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
            
            base_category, adjusted_angle = get_base_category(angle)
            
            # Determine the path to this edge
            path_to_edge = nx.shortest_path(G, reference_segment[0], current_node)
            path = [edge_categories.get((path_to_edge[i], path_to_edge[i+1]), 
                    edge_categories.get((path_to_edge[i+1], path_to_edge[i]), {'category': 'a0'}))['category'] 
                    for i in range(len(path_to_edge)-1)]
            
            if base_category in ['b', 'c']:
                category = f"{base_category}{path.count('b') + path.count('c')}"
            else:
                category = f"{base_category}{path.count('a')}"
            
            new_path = path + [category]
            dpo_value = calculate_dpo_value(category, new_path, a, b, c, d)
            
            edge_categories[use_edge] = {
                'category': category,
                'dpo_value': dpo_value,
                'path': new_path,
                'angle': adjusted_angle
            }
            
            fused_bonds = get_fused_bonds_between_rings(mol, current_node, neighbor, bond_rings)
            for bond_idx in fused_bonds:
                if bond_idx not in assigned_bonds:
                    DPO_total += dpo_value
                    assigned_bonds.add(bond_idx)
            
            print(f"Processed edge {use_edge}, category: {category}, path: {new_path}, angle: {adjusted_angle}, DPO_value: {dpo_value}, DPO_total: {DPO_total}")
            
            processed_nodes.add(neighbor)
            queue.append(neighbor)

    # Check for any unprocessed edges
    unprocessed_edges = set(edge_info.keys()) - set(edge_categories.keys())
    if unprocessed_edges:
        print("Warning: Some edges were not processed in the main loop.")
        for edge in unprocessed_edges:
            print(f"Unprocessed edge: {edge}")

    return DPO_total, assigned_bonds, edge_categories
def assign_dpo_to_angulated_segments24good(mol, G, reference_segment, edge_info, ring_centers, a, b, c, d):
    DPO_total = 0.0
    assigned_bonds = set()
    edge_categories = {}
    ring_info = mol.GetRingInfo()
    bond_rings = ring_info.BondRings()

    def get_vector(start, end):
        return ring_centers[end] - ring_centers[start]

    def get_base_category(angle):
        if abs(angle - 60) <= 15:
            return 'b', angle
        elif abs(angle - 120) <= 15:
            return 'c', angle
        elif abs(angle - 0) <= 15 or abs(angle - 180) <= 15:
            return 'a', angle
        else:
            return 'd', angle

    def calculate_dpo_value(category, path, a, b, c, d):
        if category.startswith('a'):
            a_count = 0
            for cat in reversed(path):
                if cat.startswith('a'):
                    a_count += 1
                elif cat.startswith('b') or cat.startswith('c'):
                    break
            bc_count = sum(1 for cat in path if cat.startswith('b') or cat.startswith('c'))
            return (1 - (a_count - 1) * a) * (d ** bc_count)
        elif category.startswith('b'):
            b_count = sum(1 for cat in path if cat.startswith('b'))
            return b * (d ** (b_count - 1))
        elif category.startswith('c'):
            c_count = sum(1 for cat in path if cat.startswith('c'))
            return c * (d ** (c_count - 1))
        else:
            return d

    # Calculate reference vector
    ref_vector = get_vector(reference_segment[0], reference_segment[-1])

    # Process the reference segment first
    for i in range(len(reference_segment) - 1):
        current_node = reference_segment[i]
        next_node = reference_segment[i + 1]
        edge = (current_node, next_node)

        category = f'a{i}'
        dpo_value = 1 - i * a
        edge_categories[edge] = {
            'category': category,
            'dpo_value': dpo_value,
            'path': ['a'] * (i + 1),
            'angle': 0
        }
        fused_bonds = get_fused_bonds_between_rings(mol, current_node, next_node, bond_rings)
        for bond_idx in fused_bonds:
            if bond_idx not in assigned_bonds:
                DPO_total += dpo_value
                assigned_bonds.add(bond_idx)
        print(f"Processed reference edge {edge}, category: {category}, path: {edge_categories[edge]['path']}, DPO_value: {dpo_value}, DPO_total: {DPO_total}")

    # Find all paths starting from the reference segment
    all_paths = []
    for start_node in reference_segment:
        for end_node in G.nodes():
            if start_node != end_node:
                paths = list(nx.all_simple_paths(G, start_node, end_node))
                all_paths.extend(paths)

    # Process each path
    for path in all_paths:
        print(f"Processing path: {path}")
        for i in range(len(path) - 1):
            current_node = path[i]
            next_node = path[i + 1]
            edge = (current_node, next_node)
            reverse_edge = (next_node, current_node)

            if edge in edge_info:
                use_edge = edge
            elif reverse_edge in edge_info:
                use_edge = reverse_edge
            else:
                print(f"Warning: Edge {edge} not found in edge_info")
                continue

            edge_data = edge_info[use_edge]

            # Determine the correct vector direction based on the path
            vector = get_vector(current_node, next_node)

            # Calculate angle with reference vector
            cos_angle = np.dot(ref_vector, vector) / (np.linalg.norm(ref_vector) * np.linalg.norm(vector))
            angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

            base_category, adjusted_angle = get_base_category(angle)

            # Determine the path to this edge
            path_to_edge = path[:i + 1]
            path_categories = [edge_categories.get((path_to_edge[j], path_to_edge[j + 1]),
                                                  edge_categories.get((path_to_edge[j + 1], path_to_edge[j]), {'category': 'a0'}))['category']
                               for j in range(len(path_to_edge) - 1)]

            if base_category in ['b', 'c']:
                category = f"{base_category}{path_categories.count('b') + path_categories.count('c')}"
            else:
                category = f"{base_category}{path_categories.count('a')}"

            new_path = path_categories + [category]
            dpo_value = calculate_dpo_value(category, new_path, a, b, c, d)

            edge_categories[use_edge] = {
                'category': category,
                'dpo_value': dpo_value,
                'path': new_path,
                'angle': adjusted_angle
            }

            fused_bonds = get_fused_bonds_between_rings(mol, current_node, next_node, bond_rings)
            for bond_idx in fused_bonds:
                if bond_idx not in assigned_bonds:
                    DPO_total += dpo_value
                    assigned_bonds.add(bond_idx)

            print(f"Processed edge {use_edge}, category: {category}, path: {new_path}, angle: {adjusted_angle}, DPO_value: {dpo_value}, DPO_total: {DPO_total}")

    # Check for any unprocessed edges
    unprocessed_edges = set(edge_info.keys()) - set(edge_categories.keys())
    if unprocessed_edges:
        print("Warning: Some edges were not processed in the main loop.")
        for edge in unprocessed_edges:
            print(f"Unprocessed edge: {edge}")

    return DPO_total, assigned_bonds, edge_categories
def assign_dpo_to_angulated_segments25good(mol, G, reference_segment, edge_info, ring_centers, a, b, c, d, angle_threshold):
    DPO_total = 0.0
    assigned_bonds = set()
    edge_categories = {}
    ring_info = mol.GetRingInfo()
    bond_rings = ring_info.BondRings()

    def get_vector(start, end):
        return ring_centers[end] - ring_centers[start]

    def get_base_category(angle):
        if abs(angle - 60) <= angle_threshold:
            return 'b', angle
        elif abs(angle - 120) <= angle_threshold:
            return 'c', angle
        elif abs(angle - 0) <= angle_threshold or abs(angle - 180) <= angle_threshold:
            return 'a', angle
        else:
            return 'd', angle

    def calculate_dpo_value(category, path, a, b, c, d):
        if category.startswith('a'):
            a_count = 0
            for cat in reversed(path):
                if cat.startswith('a'):
                    a_count += 1
                elif cat.startswith('b') or cat.startswith('c'):
                    break
            bc_count = sum(1 for cat in path if cat.startswith('b') or cat.startswith('c'))
            return (1 - (a_count - 1) * a) * (d ** bc_count)
        elif category.startswith('b'):
            b_count = sum(1 for cat in path if cat.startswith('b'))
            return b * (d ** (b_count - 1))
        elif category.startswith('c'):
            c_count = sum(1 for cat in path if cat.startswith('c'))
            return c * (d ** (c_count - 1))
        else:
            return d

    # Calculate reference vector
    ref_vector = get_vector(reference_segment[0], reference_segment[-1])

    # Process the reference segment first
    for i in range(len(reference_segment) - 1):
        current_node = reference_segment[i]
        next_node = reference_segment[i + 1]
        edge = (current_node, next_node)

        category = f'a{i}'
        dpo_value = 1 - i * a
        edge_categories[edge] = {
            'category': category,
            'dpo_value': dpo_value,
            'path': ['a'] * (i + 1),
            'angle': 0
        }
        fused_bonds = get_fused_bonds_between_rings(mol, current_node, next_node, bond_rings)
        for bond_idx in fused_bonds:
            if bond_idx not in assigned_bonds:
                DPO_total += dpo_value
                assigned_bonds.add(bond_idx)
        print(f"Processed reference edge {edge}, category: {category}, path: {edge_categories[edge]['path']}, DPO_value: {dpo_value}, DPO_total: {DPO_total}")

    # Find all paths starting from the reference segment
    all_paths = []
    for start_node in reference_segment:
        for end_node in G.nodes():
            if start_node != end_node:
                paths = list(nx.all_simple_paths(G, start_node, end_node))
                all_paths.extend(paths)

    # Process each path
    for path in all_paths:
        print(f"Processing path: {path}")
        for i in range(len(path) - 1):
            current_node = path[i]
            next_node = path[i + 1]
            edge = (current_node, next_node)
            reverse_edge = (next_node, current_node)

            if edge in edge_info:
                use_edge = edge
            elif reverse_edge in edge_info:
                use_edge = reverse_edge
            else:
                print(f"Warning: Edge {edge} not found in edge_info")
                continue

            edge_data = edge_info[use_edge]

            # Determine the correct vector direction based on the path
            vector = get_vector(current_node, next_node)

            # Calculate angle with reference vector
            cos_angle = np.dot(ref_vector, vector) / (np.linalg.norm(ref_vector) * np.linalg.norm(vector))
            angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

            base_category, adjusted_angle = get_base_category(angle)

            # Determine the path to this edge
            path_to_edge = path[:i + 1]
            path_categories = [edge_categories.get((path_to_edge[j], path_to_edge[j + 1]),
                                                  edge_categories.get((path_to_edge[j + 1], path_to_edge[j]), {'category': 'a0'}))['category']
                               for j in range(len(path_to_edge) - 1)]

            if base_category in ['b', 'c']:
                category = f"{base_category}{path_categories.count('b') + path_categories.count('c')}"
            else:
                category = f"{base_category}{path_categories.count('a')}"

            new_path = path_categories + [category]
            dpo_value = calculate_dpo_value(category, new_path, a, b, c, d)

            edge_categories[use_edge] = {
                'category': category,
                'dpo_value': dpo_value,
                'path': new_path,
                'angle': adjusted_angle
            }

            fused_bonds = get_fused_bonds_between_rings(mol, current_node, next_node, bond_rings)
            for bond_idx in fused_bonds:
                if bond_idx not in assigned_bonds:
                    DPO_total += dpo_value
                    assigned_bonds.add(bond_idx)

            print(f"Processed edge {use_edge}, category: {category}, path: {new_path}, angle: {adjusted_angle}, DPO_value: {dpo_value}, DPO_total: {DPO_total}")

    # Check for any unprocessed edges
    unprocessed_edges = set(edge_info.keys()) - set(edge_categories.keys())
    if unprocessed_edges:
        print("Warning: Some edges were not processed in the main loop.")
        for edge in unprocessed_edges:
            print(f"Unprocessed edge: {edge}")

    return DPO_total, assigned_bonds, edge_categories
def assign_dpo_to_angulated_segments26(mol, G, reference_segment, edge_info, a, b, c, d):
    DPO_total = 0.0
    assigned_bonds = set()
    edge_categories = {}
    ring_info = mol.GetRingInfo()
    bond_rings = ring_info.BondRings()
    
    def get_base_category(angle):
        if abs(angle - 60) <= 15:
            return 'b'
        elif abs(angle - 120) <= 15:
            return 'c'
        elif abs(angle - 0) <= 15 or abs(angle - 180) <= 15:
            return 'a'
        else:
            return 'd'  # Default category for angles that don't fit a, b, or c
    
    def calculate_dpo_value(category, path):
        if category == 'a':
            return 1 - len([c for c in path if c != 'a']) * a
        elif category.startswith('b'):
            return b * (d ** (int(category[1:]) if len(category) > 1 else 0))
        elif category.startswith('c'):
            return c * (d ** (int(category[1:]) if len(category) > 1 else 0))
        else:
            return d  # Default DPO value for category 'd'
    
    # Initialize BFS queue with nodes from the reference segment
    queue = deque([(node, []) for node in reference_segment])
    visited_nodes = set(reference_segment)
    
    while queue:
        current_node, path = queue.popleft()
        
        for neighbor in G.neighbors(current_node):
            if neighbor in visited_nodes:
                continue
            
            visited_nodes.add(neighbor)
            edge = tuple(sorted((current_node, neighbor)))
            
            if edge not in edge_info:
                continue
            
            edge_data = edge_info[edge]
            base_category = get_base_category(edge_data['angle'])
            
            if base_category in ['b', 'c']:
                category = f"{base_category}{path.count('b') + path.count('c')}"
            else:
                category = base_category
            
            new_path = path + [category]
            dpo_value = calculate_dpo_value(category, new_path)
            
            edge_categories[edge] = {
                'category': category,
                'dpo_value': dpo_value,
                'path': new_path,
                'angle': edge_data['angle']
            }
            
            fused_bonds = get_fused_bonds_between_rings(mol, edge_data['direction'][0], edge_data['direction'][1], bond_rings)
            for bond_idx in fused_bonds:
                if bond_idx not in assigned_bonds:
                    DPO_total += dpo_value
                    assigned_bonds.add(bond_idx)
            
            queue.append((neighbor, new_path))
            
            print(f"Processed edge {edge}, category: {category}, path: {new_path}, DPO_value: {dpo_value}, DPO_total: {DPO_total}")
    
    # Check for unprocessed edges
    unprocessed_edges = set(edge_info.keys()) - set(edge_categories.keys())
    if unprocessed_edges:
        print(f"Warning: {len(unprocessed_edges)} edges were not processed: {unprocessed_edges}")
        # Process unconnected edges
        for edge in unprocessed_edges:
            edge_data = edge_info[edge]
            category = get_base_category(edge_data['angle'])
            dpo_value = calculate_dpo_value(category, [category])
            edge_categories[edge] = {
                'category': category,
                'dpo_value': dpo_value,
                'path': [category],
                'angle': edge_data['angle']
            }
            print(f"Processed unconnected edge {edge}, category: {category}, path: [{category}], DPO_value: {dpo_value}")
    
    return DPO_total, assigned_bonds, edge_categories
def assign_dpo_to_angulated_segments27(mol, G, reference_segment, edge_info, a, b, c, d):
    DPO_total = 0.0
    assigned_bonds = set()
    edge_categories = {}
    ring_info = mol.GetRingInfo()
    bond_rings = ring_info.BondRings()
    
    def get_base_category(angle):
        if abs(angle - 60) <= 15:
            return 'b'
        elif abs(angle - 120) <= 15:
            return 'c'
        elif abs(angle - 0) <= 15 or abs(angle - 180) <= 15:
            return 'a'
        else:
            return 'd'  # Default category for angles that don't fit a, b, or c
    
    def calculate_dpo_value(category, path):
        # if category == 'a':
        #     return 1 - len([c for c in path if c != 'a']) * a
        if category.startswith('a'):
            a_count = 0
            for cat in reversed(path):
                if cat.startswith('a'):
                    a_count += 1
                elif cat.startswith('b') or cat.startswith('c'):
                    break
            bc_count = sum(1 for cat in path if cat.startswith('b') or cat.startswith('c'))
            return (1 - (a_count - 1) * a) * (d ** bc_count)
        elif category.startswith('b'):
            return b * (d ** (int(category[1:]) if len(category) > 1 else 0))
        elif category.startswith('c'):
            return c * (d ** (int(category[1:]) if len(category) > 1 else 0))
        else:
            return d  # Default DPO value for category 'd'
    
    # Initialize BFS queue with nodes from the reference segment
    queue = deque([(node, []) for node in reference_segment])
    visited_nodes = set(reference_segment)
    
    while queue:
        current_node, path = queue.popleft()
        
        for neighbor in G.neighbors(current_node):
            if neighbor in visited_nodes:
                continue
            
            visited_nodes.add(neighbor)
            edge = tuple(sorted((current_node, neighbor)))
            
            if edge not in edge_info:
                continue
            
            edge_data = edge_info[edge]
            base_category = get_base_category(edge_data['angle'])
            
            if base_category in ['b', 'c']:
                category = f"{base_category}{path.count('b') + path.count('c')}"
            else:
                category = base_category
            
            new_path = path + [category]
            dpo_value = calculate_dpo_value(category, new_path)
            print("path",path,"category",category,"new_path",new_path,"dpo_value",dpo_value)

            edge_categories[edge] = {
                'category': category,
                'dpo_value': dpo_value,
                'path': new_path,
                'angle': edge_data['angle']
            }
            
            fused_bonds = get_fused_bonds_between_rings(mol, edge_data['direction'][0], edge_data['direction'][1], bond_rings)
            for bond_idx in fused_bonds:
                if bond_idx not in assigned_bonds:
                    DPO_total += dpo_value
                    assigned_bonds.add(bond_idx)
            
            queue.append((neighbor, new_path))
            
            print(f"Processed edge {edge}, category: {category}, path: {new_path}, DPO_value: {dpo_value}, DPO_total: {DPO_total}")
    
    # Check for unprocessed edges
    unprocessed_edges = set(edge_info.keys()) - set(edge_categories.keys())
    if unprocessed_edges:
        print(f"Warning: {len(unprocessed_edges)} edges were not processed: {unprocessed_edges}")
        # Process unconnected edges
        for edge in unprocessed_edges:
            edge_data = edge_info[edge]
            category = get_base_category(edge_data['angle'])
            dpo_value = calculate_dpo_value(category, [category])
            edge_categories[edge] = {
                'category': category,
                'dpo_value': dpo_value,
                'path': [category],
                'angle': edge_data['angle']
            }
            print(f"Processed unconnected edge {edge}, category: {category}, path: [{category}], DPO_value: {dpo_value}")
    
    return DPO_total, assigned_bonds, edge_categories
def assign_dpo_to_angulated_segments(mol, G, reference_segment, edge_info, ring_centers, a, b, c, d):
    DPO_total = 0.0
    assigned_bonds = set()
    edge_categories = {}
    ring_info = mol.GetRingInfo()
    bond_rings = ring_info.BondRings()

    def get_vector(start, end):
        return ring_centers[end] - ring_centers[start]

    # def get_base_category(angle):
    #     if abs(angle - 60) <= 15:
    #         return 'b', angle
    #     elif abs(angle - 120) <= 15:
    #         return 'c', angle
    #     elif abs(angle - 0) <= 15 or abs(angle - 180) <= 15:
    #         return 'a', angle
    #     else:
    #         return 'd', angle
    def get_base_category(edge, reference_segment, angle):
        if edge[0] in reference_segment and edge[1] in reference_segment:
            return 'ref', angle
        elif abs(angle - 60) <= AngleThreshold:
            return 'b', angle
        elif abs(angle - 120) <= AngleThreshold:
            return 'c', angle
        elif abs(angle - 0) <= AngleThreshold or abs(angle - 180) <= AngleThreshold:
            return 'a', angle
        else:
            return 'd', angle


    def calculate_dpo_value(category, path, a, b, c, d):
        if category.startswith('ref'):
            a_count = 0
            for cat in reversed(path):
                if cat.startswith('ref'):
                    a_count += 1
                    # b_count = 0 #reset count
                    # c_count = 0 #reset count
                elif cat.startswith('b') or cat.startswith('c'):
                    break
            bc_count = sum(1 for cat in path if cat.startswith('b') or cat.startswith('c'))
            return (1 - (a_count - 1) * a) 
        if category.startswith('a'):
            a_count = 0
            for cat in reversed(path):
                if cat.startswith('a'):
                    a_count += 1
                elif cat.startswith('b') or cat.startswith('c'):
                    break
            bc_count = sum(1 for cat in path if cat.startswith('b') or cat.startswith('c'))
            return (1 - (a_count - 1) * a) * (d ** bc_count)
        elif category.startswith('b'):
            b_count = sum(1 for cat in path if cat.startswith('b'))
            return b * (d ** (b_count - 1))
        elif category.startswith('c'):
            c_count = sum(1 for cat in path if cat.startswith('c'))
            return c * (d ** (c_count - 1))
        else:
            return d

    

    # Process the reference segment first
    for i in range(len(reference_segment) - 1):
        current_node = reference_segment[i]
        next_node = reference_segment[i + 1]
        edge = (current_node, next_node)

        category = f'a{i}'
        dpo_value = 1 - i * a
        edge_categories[edge] = {
            'category': category,
            'dpo_value': dpo_value,
            'path': ['a'] * (i + 1),
            'angle': 0
        }
        fused_bonds = get_fused_bonds_between_rings(mol, current_node, next_node, bond_rings)
        for bond_idx in fused_bonds:
            if bond_idx not in assigned_bonds:
                DPO_total += dpo_value
                assigned_bonds.add(bond_idx)
        # print(f"Processed reference edge {edge}, category: {category}, path: {edge_categories[edge]['path']}, DPO_value: {dpo_value}, DPO_total: {DPO_total}")

    
    # Initialize a set to keep track of processed edges
    processed_edges = set()
    
    # # # Find all paths starting from the reference segment
    # all_paths = []
    # for start_node in reference_segment:
    #     for end_node in G.nodes():
    #         if start_node != end_node:
    #             paths = list(nx.all_simple_paths(G, start_node, end_node))
    #             all_paths.extend(paths)
    # Find all paths that contain the reference segment
    all_forward_paths = []
    for start_node in reference_segment:
        for end_node in G.nodes():
            if start_node != end_node:
                paths = list(nx.all_simple_paths(G, start_node, end_node))
                for path in paths:
                    if all(node in path for node in reference_segment):
                        # Check if the reference segment appears as a contiguous subpath
                        ref_indices = [path.index(node) for node in reference_segment if node in path]
                        if ref_indices == list(range(min(ref_indices), max(ref_indices) + 1)):
                            all_forward_paths.append(path)
    all_backward_paths = []
    for start_node in G.nodes():
        for end_node in reference_segment:
            if start_node != end_node:
                paths = list(nx.all_simple_paths(G, start_node, end_node))
                for path in paths:
                    if all(node in path for node in reference_segment):
                        # Check if the nodes in the reference segment appear in the same order
                        ref_indices = [path.index(node) for node in reference_segment if node in path]
                        if ref_indices == sorted(ref_indices):
                            if ref_indices == list(range(min(ref_indices), max(ref_indices) + 1)):
                                all_backward_paths.append(path)
    # all_backward_paths = []
    # reverse_reference_segment = reference_segment[::-1]  # Reverse the reference segment
    # for start_node in reverse_reference_segment:
    #     for end_node in G.nodes():
    #         if start_node != end_node:
    #             paths = list(nx.all_simple_paths(G, start_node, end_node))
    #             for path in paths:
    #                 if all(node in path for node in reverse_reference_segment):
    #                     # Check if the reference segment appears as a contiguous subpath
    #                     ref_indices = [path.index(node) for node in reverse_reference_segment if node in path]
    #                     if ref_indices == list(range(min(ref_indices), max(ref_indices) + 1)):
    #                         all_forward_paths.append(path)
    # for start_node in G.nodes():
    #     for end_node in reference_segment:
    #         if start_node != end_node:
    #             paths = list(nx.all_simple_paths(G, start_node, end_node))
    #             for path in paths:
    #                 if all(node in path for node in reverse_reference_segment):
    #                     # Check if the reverse reference segment appears as a contiguous subpath
    #                     for i in range(len(path) - len(reverse_reference_segment) + 1):
    #                         if path[i:i + len(reverse_reference_segment)] == reverse_reference_segment:
    #                             all_backward_paths.append(path)
    #                             break
    
    # Create a set of all_backward_paths for quick lookup
    backward_paths_set = {tuple(path) for path in all_backward_paths}
    # Combine both sets of paths
    all_paths = all_forward_paths + [path[::-1] for path in all_backward_paths]

    # Calculate reference vector
    ref_vector = get_vector(reference_segment[0], reference_segment[-1])

    # Process each path
    for path in all_paths:
        # print(f"Processing path: {path}")
        for i in range(len(path) - 1):
            current_node = path[i]
            next_node = path[i + 1]
            # edge = (current_node, next_node)
            # reverse_edge = (next_node, current_node)

            # if edge in edge_info:
            #     use_edge = edge
            # elif reverse_edge in edge_info:
            #     use_edge = reverse_edge
            # else:
            #     print(f"Warning: Edge {edge} not found in edge_info")
            #     continue
            
            if tuple(path[::-1]) in backward_paths_set:
                use_edge = (next_node, current_node)
                unused_edge = (current_node, next_node)
                # Determine the correct vector direction based on the path
                vector = get_vector(next_node, current_node)
                # ref_vector = get_vector(reverse_reference_segment[0], reverse_reference_segment[-1])
                print("path",path,"current_node",next_node,"next_node",current_node)
            else:
                print("path",path,"current_node",current_node,"next_node",next_node)
                use_edge = (current_node, next_node)
                unused_edge = (next_node, current_node)
                # Determine the correct vector direction based on the path
                vector = get_vector(current_node, next_node)
                # ref_vector = get_vector(reference_segment[0], reference_segment[-1])
                

            # if current_node > next_node:
            #     use_edge = (current_node, next_node)
            # else:
            #     use_edge = (next_node, current_node)

            # Skip already processed edges
            if use_edge in processed_edges:
                continue

            edge_data = edge_info[use_edge]

            # # Determine the correct vector direction based on the path
            # vector = get_vector(current_node, next_node)

            # Calculate angle with reference vector
            cos_angle = np.dot(ref_vector, vector) / (np.linalg.norm(ref_vector) * np.linalg.norm(vector))
            angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

            # base_category, adjusted_angle = get_base_category(angle)
            base_category, adjusted_angle = get_base_category(use_edge, reference_segment, angle)

            # Determine the path to this edge
            path_to_edge = path[:i + 1]
            path_categories = [edge_categories.get((path_to_edge[j], path_to_edge[j + 1]),
                                                  edge_categories.get((path_to_edge[j + 1], path_to_edge[j]), {'category': 'a0'}))['category']
                               for j in range(len(path_to_edge) - 1)]

            if base_category in ['b', 'c']:
                category = f"{base_category}{path_categories.count('b') + path_categories.count('c')}"
            else:
                category = f"{base_category}{path_categories.count('a')}"

            new_path = path_categories + [category]
            dpo_value = calculate_dpo_value(category, new_path, a, b, c, d)

            edge_categories[use_edge] = {
                'category': category,
                'dpo_value': dpo_value,
                'path': new_path,
                'angle': adjusted_angle
            }

            fused_bonds = get_fused_bonds_between_rings(mol, current_node, next_node, bond_rings)
            for bond_idx in fused_bonds:
                if bond_idx not in assigned_bonds:
                    DPO_total += dpo_value
                    assigned_bonds.add(bond_idx)

            # Mark the edge as processed
            processed_edges.add(use_edge)
            processed_edges.add(unused_edge)
            
            print(f"Processed edge {use_edge}, category: {category}, path: {new_path}, angle: {adjusted_angle}, DPO_value: {dpo_value}, DPO_total: {DPO_total}")

    # Check for any unprocessed edges
    unprocessed_edges = set(edge_info.keys()) - set(edge_categories.keys())
    if unprocessed_edges:
        print("Warning: Some edges were not processed in the main loop.")
        for edge in unprocessed_edges:
            print(f"Unprocessed edge: {edge}")

    return DPO_total, assigned_bonds, edge_categories
def get_fused_bonds_between_rings(mol, ring_idx1, ring_idx2, bond_rings):
    """
    Get the bond indices shared between two rings.
    """
    bonds_ring1 = set(bond_rings[ring_idx1])
    bonds_ring2 = set(bond_rings[ring_idx2])
    fused_bonds = bonds_ring1.intersection(bonds_ring2)
    return fused_bonds
def get_fused_bonds_in_ring(mol, ring_idx, bond_rings):
    """
    Get the fused bonds in a given ring.

    Parameters:
    - mol: RDKit molecule object.
    - ring_idx: Index of the ring in the bond_rings list.
    - bond_rings: List of lists, where each sublist contains bond indices for a ring.

    Returns:
    - fused_bonds: List of bond indices that are fused bonds in the given ring.
    """
    # Access ring information from the molecule
    ring_info = mol.GetRingInfo()
    fused_bonds = []
    # Get the bond indices that make up the ring with index ring_idx
    ring_bonds = bond_rings[ring_idx]
    # print(ring_info)
    # print(ring_bonds)
    # Iterate over each bond in the ring
    for bond_idx in ring_bonds:
        # Get the number of rings this bond is part of
        num_bond_rings = ring_info.NumBondRings(bond_idx)
        if num_bond_rings > 1:
            # If the bond is part of more than one ring, it's a fused bond
            fused_bonds.append(bond_idx)

    # Return the list of fused bond indices
    return fused_bonds
def calculate_ring_centers_OLD(mol, atom_rings):
   """
   Calculate the center (geometric centroid) of each ring.

   Returns:
   - ring_centers: List of numpy arrays representing the center coordinates of each ring.
   """
   ring_centers = []
   conformer = mol.GetConformer()

   for ring in atom_rings:
       coords = np.array([conformer.GetAtomPosition(idx) for idx in ring])
       center = coords.mean(axis=0)
       ring_centers.append(center)

   return ring_centers
def angle_at_ring_j_OLD(ring_centers, i, j, k):
   """
   Calculate the angle at ring j formed by centers of rings i, j, and k.

   Parameters:
   - ring_centers: List of ring center coordinates.
   - i, j, k: Indices of the rings.

   Returns:
   - angle: Angle in degrees.
   """
   center_i = ring_centers[i]
   center_j = ring_centers[j]
   center_k = ring_centers[k]

   vector1 = center_j - center_i
   vector2 = center_k - center_j

   angle = angle_between_vectors(vector1, vector2)
   return angle
def angle_at_ring_j(ring_centers, i, j, k):
    """
    Calculate the angle at ring j formed by rings i, j, and k.
    """
    vector_ij = ring_centers[i] - ring_centers[j]  # Note the reversed order
    vector_jk = ring_centers[k] - ring_centers[j]
    angle = angle_between_vectors(vector_ij, vector_jk)
    return angle
def angle_between_vectors(v1, v2):
    dot_product = np.dot(v1, v2)
    magnitudes = np.linalg.norm(v1) * np.linalg.norm(v2)
    cos_angle = dot_product / magnitudes
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    return np.degrees(angle)
def angle_between_vectors_OLD(v1, v2):
   """
   Calculate the angle between two vectors in degrees.
   """
   cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
   cos_theta = np.clip(cos_theta, -1.0, 1.0)  # Ensure numerical stability
   angle = np.degrees(np.arccos(cos_theta))
   return angle
def align_vector_with_path(vector, path_direction):
    if np.dot(vector, path_direction) < 0:
        return -vector
    return vector
def find_longest_linear_path_OLD(G, ring_centers, angle_threshold=AngleThreshold):
   """
   Find the longest linear path in the ring graph based on angular criteria.
   """
   longest_path = []
   
   for node in G.nodes():
       for neighbor in G.neighbors(node):
           path = [node, neighbor]
           current_node = neighbor
           previous_node = node
           while True:
               neighbors = list(G.neighbors(current_node))
               neighbors = [n for n in neighbors if n != previous_node]
               if not neighbors:
                   break
               next_node = neighbors[0]
               angle = angle_at_ring_j(ring_centers, previous_node, current_node, next_node)
               if abs(angle - 180) <= angle_threshold:
                   path.append(next_node)
                   previous_node, current_node = current_node, next_node
               else:
                   break
           if len(path) > len(longest_path):
               longest_path = path
   
   return longest_path
def find_longest_linear_path2(G, ring_centers, angle_threshold=AngleThreshold):
    longest_path = []
    for start_node in G.nodes():
        for end_node in G.nodes():
            if start_node != end_node:
                paths = list(nx.all_simple_paths(G, start_node, end_node))
                for path in paths:
                    if is_path_linear(path, ring_centers, angle_threshold):
                        if len(path) > len(longest_path):
                            longest_path = path
    return longest_path
def find_longest_linear_path3(G, ring_centers, angle_threshold=AngleThreshold):
    def path_length_score(path):
        if len(path) < 3:
            return len(path)
        
        total_angle_deviation = 0
        for i in range(len(path) - 2):
            v1 = ring_centers[path[i+1]] - ring_centers[path[i]]
            v2 = ring_centers[path[i+2]] - ring_centers[path[i+1]]
            angle = angle_between_vectors(v1, v2)
            total_angle_deviation += abs(180 - angle)
            # total_angle_deviation += abs(angle)
            # print("total_angle_deviation",total_angle_deviation)
        
        return len(path) - (total_angle_deviation / (len(path) - 2)) / 180

    longest_path = []
    best_score = 0
    
    for start_node in G.nodes():
        for end_node in G.nodes():
            if start_node != end_node:
                paths = list(nx.all_simple_paths(G, start_node, end_node))
                # print("paths",paths)
                for path in paths:
                    if is_path_linear(path, ring_centers, angle_threshold):
                        # print("path {} is linear".format(path))
                        score = path_length_score(path)
                        if score > best_score:
                            longest_path = path
                            best_score = score
    
    return longest_path
def find_longest_linear_path4(G, ring_centers, angle_threshold=AngleThreshold):
    def path_length_score(path):
        if len(path) < 3:
            return len(path), 0
        
        total_angle_deviation = 0
        parallel_segments = 0
        for i in range(len(path) - 2):
            v1 = ring_centers[path[i+1]] - ring_centers[path[i]]
            v2 = ring_centers[path[i+2]] - ring_centers[path[i+1]]
            angle = angle_between_vectors(v1, v2)
            total_angle_deviation += abs(angle)
            if abs(angle) <= angle_threshold:
                parallel_segments += 1
        
        length_score = len(path) - (total_angle_deviation / (len(path) - 2)) / 180
        return length_score, parallel_segments

    longest_path = []
    best_score = 0
    best_parallel_segments = 0
    
    for start_node in G.nodes():
        for end_node in G.nodes():
            if start_node != end_node:
                paths = list(nx.all_simple_paths(G, start_node, end_node))
                for path in paths:
                    if is_path_linear(path, ring_centers, angle_threshold):
                        score, parallel_segments = path_length_score(path)
                        if score > best_score or (score == best_score and parallel_segments > best_parallel_segments):
                            longest_path = path
                            best_score = score
                            best_parallel_segments = parallel_segments
    
    return longest_path
def find_longest_linear_path5(G, ring_centers, angle_threshold=AngleThreshold):
    def path_length_score(path):
        if len(path) < 3:
            return len(path), 0, count_overlayers(G, path)
        
        total_angle_deviation = 0
        parallel_segments = 0
        for i in range(len(path) - 2):
            v1 = ring_centers[path[i+1]] - ring_centers[path[i]]
            v2 = ring_centers[path[i+2]] - ring_centers[path[i+1]]
            angle = angle_between_vectors(v1, v2)
            total_angle_deviation += abs(angle)
            if abs(angle) <= angle_threshold:
                parallel_segments += 1
        
        length_score = len(path) - (total_angle_deviation / (len(path) - 2)) / 180
        overlayers = count_overlayers(G, path)
        return length_score, parallel_segments, overlayers

    def count_overlayers(G, path):
        overlayers = 0
        path_set = set(path)
        for node in path:
            neighbors = set(G.neighbors(node))
            overlayers += len(neighbors - path_set)
        return overlayers

    longest_path = []
    best_score = 0
    best_parallel_segments = 0
    least_overlayers = float('inf')
    
    for start_node in G.nodes():
        for end_node in G.nodes():
            if start_node != end_node:
                paths = list(nx.all_simple_paths(G, start_node, end_node))
                for path in paths:
                    if is_path_linear(path, ring_centers, angle_threshold):
                        score, parallel_segments, overlayers = path_length_score(path)
                        if (score > best_score or 
                            (score == best_score and parallel_segments > best_parallel_segments) or
                            (score == best_score and parallel_segments == best_parallel_segments and overlayers < least_overlayers)):
                            longest_path = path
                            best_score = score
                            best_parallel_segments = parallel_segments
                            least_overlayers = overlayers
    
    return longest_path
def assign_layers(ring_centers, layer_threshold=0.5):  # Increased threshold
    ring_layers = {}
    rings_with_indices = list(enumerate(ring_centers))
    sorted_rings = sorted(rings_with_indices, key=lambda x: x[1][2])  # Sort by Z-coordinate
    current_layer = 0
    last_z = None
    for node, center in sorted_rings:
        z = center[2]
        if last_z is not None and abs(z - last_z) > layer_threshold:
            current_layer += 1
        ring_layers[node] = current_layer
        last_z = z
    return ring_layers

def count_fused_bonds_to_others(G, path):
    path_set = set(path)
    fused_bonds = 0
    for node in path:
        neighbors = set(G.neighbors(node))
        external_neighbors = neighbors - path_set
        # For each external neighbor, check if it's connected via a fused bond
        # Assuming all edges in G represent fused bonds
        fused_bonds += len(external_neighbors)
    return fused_bonds

def find_reference_segment2(G, ring_centers, angle_threshold=AngleThreshold):
    def segment_score(path):
        if len(path) < 3:
            return len(path), 0, 0, count_overlayers(G, path)
        
        linear_length = 0
        current_linear_length = 1
        parallel_bonds = 0
        linear_segments = 0
        
        for i in range(len(path) - 2):
            v1 = ring_centers[path[i+1]] - ring_centers[path[i]]
            v2 = ring_centers[path[i+2]] - ring_centers[path[i+1]]
            angle = angle_between_vectors(v1, v2)
            
            if abs(angle) <= angle_threshold:
                current_linear_length += 1
                parallel_bonds += 1
            else:
                if current_linear_length > linear_length:
                    linear_length = current_linear_length
                if current_linear_length > 1:
                    linear_segments += 1
                current_linear_length = 1
        
        if current_linear_length > linear_length:
            linear_length = current_linear_length
        if current_linear_length > 1:
            linear_segments += 1
        
        overlayers = count_overlayers(G, path)
        return linear_length, parallel_bonds, linear_segments, overlayers

    def count_overlayers(G, path):
        overlayers = 0
        path_set = set(path)
        for node in path:
            neighbors = set(G.neighbors(node))
            overlayers += len(neighbors - path_set)
        return overlayers

    best_segment = []
    best_linear_length = 0
    best_parallel_bonds = 0
    best_linear_segments = 0
    least_overlayers = float('inf')
    
    for start_node in G.nodes():
        for end_node in G.nodes():
            if start_node != end_node:
                paths = list(nx.all_simple_paths(G, start_node, end_node))
                for path in paths:
                    linear_length, parallel_bonds, linear_segments, overlayers = segment_score(path)
                    if (linear_length > best_linear_length or
                        (linear_length == best_linear_length and parallel_bonds > best_parallel_bonds) or
                        (linear_length == best_linear_length and parallel_bonds == best_parallel_bonds and overlayers < least_overlayers)):
                        best_segment = path
                        best_linear_length = linear_length
                        best_parallel_bonds = parallel_bonds
                        best_linear_segments = linear_segments
                        least_overlayers = overlayers
    
    # Extract the longest linear subsegment from the best segment
    longest_linear_subsegment = []
    current_subsegment = [best_segment[0]]
    for i in range(1, len(best_segment)):
        if len(current_subsegment) == 1 or angle_between_vectors(
            ring_centers[current_subsegment[-1]] - ring_centers[current_subsegment[-2]],
            ring_centers[best_segment[i]] - ring_centers[current_subsegment[-1]]
        ) <= angle_threshold:
            current_subsegment.append(best_segment[i])
        else:
            if len(current_subsegment) > len(longest_linear_subsegment):
                longest_linear_subsegment = current_subsegment
            current_subsegment = [best_segment[i-1], best_segment[i]]
    
    if len(current_subsegment) > len(longest_linear_subsegment):
        longest_linear_subsegment = current_subsegment

    return longest_linear_subsegment
def find_reference_segmentstuffed(G, ring_centers, angle_threshold=AngleThreshold):
    ring_layers = assign_layers(ring_centers)
    def segment_score(path):
        if len(path) < 3:
            return len(path), 0, 0, count_overlayers(G, path, ring_layers), 0
        
        linear_length = 0
        current_linear_length = 1
        parallel_bonds = 0
        linear_segments = 0
        
        for i in range(len(path) - 2):
            v1 = ring_centers[path[i+1]] - ring_centers[path[i]]
            v2 = ring_centers[path[i+2]] - ring_centers[path[i+1]]
            angle = angle_between_vectors(v1, v2)
            
            if abs(angle) <= angle_threshold:
                current_linear_length += 1
                parallel_bonds += 1
            else:
                if current_linear_length > linear_length:
                    linear_length = current_linear_length
                if current_linear_length > 1:
                    linear_segments += 1
                current_linear_length = 1
        
        if current_linear_length > linear_length:
            linear_length = current_linear_length
        if current_linear_length > 1:
            linear_segments += 1
        
        overlayers = count_overlayers(G, path, ring_layers)
        
        # Count fused bonds to other segments
        fused_bonds_to_others = count_fused_bonds_to_others(G, path)
        
        return linear_length, parallel_bonds, linear_segments, overlayers, fused_bonds_to_others


    def count_overlayers1(G, path):
        overlayers = 0
        path_set = set(path)
        for node in path:
            neighbors = set(G.neighbors(node))
            overlayers += len(neighbors - path_set)
        return overlayers
    def count_overlayers(G, path, ring_layers):
        overlayers = 0
        path_set = set(path)
        path_layers = set(ring_layers[node] for node in path)
        for node in path:
            neighbors = set(G.neighbors(node))
            for neighbor in neighbors - path_set:
                # Only count neighbors in different layers as overlayers
                if ring_layers[neighbor] not in path_layers:
                    overlayers += 1
        return overlayers

    

    # best_segment = []
    # best_linear_length = 0
    # best_parallel_bonds = 0
    # best_linear_segments = 0
    # least_overlayers = float('inf')
    
    # # Iterate over all pairs of nodes to find all paths
    # for start_node in G.nodes():
    #     for end_node in G.nodes():
    #         if start_node != end_node:
    #             paths = list(nx.all_simple_paths(G, start_node, end_node))
    #             for path in paths:
    #                 linear_length, parallel_bonds, linear_segments, overlayers = segment_score(path)
    #                 # Update the best segment based on the scoring criteria
    #                 if (linear_length > best_linear_length or
    #                     (linear_length == best_linear_length and parallel_bonds > best_parallel_bonds) or
    #                     (linear_length == best_linear_length and parallel_bonds == best_parallel_bonds and overlayers < least_overlayers)):
    #                     best_segment = path
    #                     best_linear_length = linear_length
    #                     best_parallel_bonds = parallel_bonds
    #                     best_linear_segments = linear_segments
    #                     least_overlayers = overlayers
    # Initialize variables
    best_segment = []
    best_linear_length = 0
    best_parallel_bonds = 0
    best_linear_segments = 0
    least_overlayers = float('inf')
    most_fused_bonds = -1  # Initialize to -1 to ensure any non-negative number will be larger

    # In the loop where you evaluate each path
    for path in paths:
        linear_length, parallel_bonds, linear_segments, overlayers, fused_bonds_to_others = segment_score(path)
        # Update the best segment based on the scoring criteria
        if (linear_length > best_linear_length or
            (linear_length == best_linear_length and parallel_bonds > best_parallel_bonds) or
            (linear_length == best_linear_length and parallel_bonds == best_parallel_bonds and overlayers < least_overlayers) or
            (linear_length == best_linear_length and parallel_bonds == best_parallel_bonds and overlayers == least_overlayers and fused_bonds_to_others > most_fused_bonds)):
            
            best_segment = path
            best_linear_length = linear_length
            best_parallel_bonds = parallel_bonds
            best_linear_segments = linear_segments
            least_overlayers = overlayers
            most_fused_bonds = fused_bonds_to_others

    
    # Extract the longest linear subsegment from the best segment
    longest_linear_subsegment = []
    current_subsegment = [best_segment[0]]
    for i in range(1, len(best_segment)):
        if len(current_subsegment) == 1 or angle_between_vectors(
            ring_centers[current_subsegment[-1]] - ring_centers[current_subsegment[-2]],
            ring_centers[best_segment[i]] - ring_centers[current_subsegment[-1]]
        ) <= angle_threshold:
            current_subsegment.append(best_segment[i])
        else:
            if len(current_subsegment) > len(longest_linear_subsegment):
                longest_linear_subsegment = current_subsegment
            current_subsegment = [best_segment[i-1], best_segment[i]]
    
    if len(current_subsegment) > len(longest_linear_subsegment):
        longest_linear_subsegment = current_subsegment

    # Ensure the reference segment starts from a terminal node if possible
    terminal_nodes = [node for node in longest_linear_subsegment if G.degree(node) == 1]
    if terminal_nodes:
        if longest_linear_subsegment[0] not in terminal_nodes:
            longest_linear_subsegment.reverse()

    return longest_linear_subsegment
def find_reference_segment4(G, ring_centers, angle_threshold=AngleThreshold):
    ring_layers = assign_layers(ring_centers)
    def segment_score(path):
        if len(path) < 3:
            overlayers = count_overlayers(G, path, ring_layers)
            fused_bonds_to_others = count_fused_bonds_to_others(G, path)
            return len(path), 0, 0, overlayers, fused_bonds_to_others
        
        linear_length = 0
        current_linear_length = 1
        parallel_bonds = 0
        linear_segments = 0
        
        for i in range(len(path) - 2):
            v1 = ring_centers[path[i+1]] - ring_centers[path[i]]
            v2 = ring_centers[path[i+2]] - ring_centers[path[i+1]]
            angle = angle_between_vectors(v1, v2)
            
            if abs(angle) <= angle_threshold:
                current_linear_length += 1
                parallel_bonds += 1
            else:
                if current_linear_length > linear_length:
                    linear_length = current_linear_length
                if current_linear_length > 1:
                    linear_segments += 1
                current_linear_length = 1
        
        if current_linear_length > linear_length:
            linear_length = current_linear_length
        if current_linear_length > 1:
            linear_segments += 1
        
        overlayers = count_overlayers(G, path, ring_layers)
        fused_bonds_to_others = count_fused_bonds_to_others(G, path)
        
        return linear_length, parallel_bonds, linear_segments, overlayers, fused_bonds_to_others
    
    def count_overlayers(G, path, ring_layers):
        overlayers = 0
        path_set = set(path)
        path_layers = set(ring_layers[node] for node in path)
        for node in path:
            neighbors = set(G.neighbors(node))
            for neighbor in neighbors - path_set:
                if ring_layers[neighbor] not in path_layers:
                    overlayers += 1
                    print("node", node, "overlayer", overlayers)
        return overlayers

    def count_fused_bonds_to_others(G, path):
        path_set = set(path)
        fused_bonds = 0
        for node in path:
            neighbors = set(G.neighbors(node))
            external_neighbors = neighbors - path_set
            fused_bonds += len(external_neighbors)
        return fused_bonds

    best_segment = []
    best_linear_length = 0
    best_parallel_bonds = 0
    best_linear_segments = 0
    least_overlayers = float('inf')
    most_fused_bonds = -1
    
    for start_node in G.nodes():
        for end_node in G.nodes():
            if start_node != end_node:
                paths = list(nx.all_simple_paths(G, start_node, end_node))
                for path in paths:
                    linear_length, parallel_bonds, linear_segments, overlayers, fused_bonds_to_others = segment_score(path)
                    if (linear_length > best_linear_length or
                        (linear_length == best_linear_length and parallel_bonds > best_parallel_bonds) or
                        (linear_length == best_linear_length and parallel_bonds == best_parallel_bonds and overlayers < least_overlayers) or
                        (linear_length == best_linear_length and parallel_bonds == best_parallel_bonds and overlayers == least_overlayers and fused_bonds_to_others > most_fused_bonds)):
                        best_segment = path
                        best_linear_length = linear_length
                        best_parallel_bonds = parallel_bonds
                        best_linear_segments = linear_segments
                        least_overlayers = overlayers
                        most_fused_bonds = fused_bonds_to_others
    
    # Extract the longest linear subsegment from the best segment
    longest_linear_subsegment = []
    current_subsegment = [best_segment[0]]
    for i in range(1, len(best_segment)):
        if len(current_subsegment) == 1 or angle_between_vectors(
            ring_centers[current_subsegment[-1]] - ring_centers[current_subsegment[-2]],
            ring_centers[best_segment[i]] - ring_centers[current_subsegment[-1]]
        ) <= angle_threshold:
            current_subsegment.append(best_segment[i])
        else:
            if len(current_subsegment) > len(longest_linear_subsegment):
                longest_linear_subsegment = current_subsegment
            current_subsegment = [best_segment[i-1], best_segment[i]]
    
    if len(current_subsegment) > len(longest_linear_subsegment):
        longest_linear_subsegment = current_subsegment

    # Ensure the reference segment starts from a terminal node if possible
    terminal_nodes = [node for node in longest_linear_subsegment if G.degree(node) == 1]
    if terminal_nodes:
        if longest_linear_subsegment[0] not in terminal_nodes:
            longest_linear_subsegment.reverse()

    return longest_linear_subsegment
def find_reference_segmentshit(G, ring_centers, angle_threshold=AngleThreshold):
    def segment_score(path):
        # Compute the main axis of the segment
        if len(path) >= 2:
            start_point = ring_centers[path[0]]
            end_point = ring_centers[path[-1]]
            main_axis = end_point - start_point
            main_axis /= np.linalg.norm(main_axis)  # Normalize the vector

            # For each ring, compute its perpendicular distance to the main axis
            ring_layers = {}
            layer_threshold = 0.1  # Adjust based on your coordinate scale
            for node in G.nodes():
                ring_center = ring_centers[node]
                # Vector from start point to ring center
                v = ring_center - start_point
                # Project v onto main_axis
                projection_length = np.dot(v, main_axis)
                projection_point = start_point + projection_length * main_axis
                # Compute perpendicular distance
                perpendicular_distance = np.linalg.norm(ring_center - projection_point)
                # Assign layers based on perpendicular distance
                if perpendicular_distance <= layer_threshold:
                    layer = 0  # On the main axis (same layer)
                elif ring_center[1] > projection_point[1]:
                    layer = 1  # Above the main axis
                else:
                    layer = -1  # Below the main axis
                ring_layers[node] = layer
        else:
            # For single-node paths, assign all rings to layer 0
            ring_layers = {node: 0 for node in G.nodes()}

        # Now compute overlayers using the ring_layers
        overlayers = count_overlayers(G, path, ring_layers)

        # Compute linear_length, parallel_bonds, linear_segments, fused_bonds_to_others
        linear_length = len(path)
        parallel_bonds = max(0, len(path) - 1)  # Since in 2D, adjacent rings are considered parallel
        linear_segments = 1  # Assuming the path is one segment

        fused_bonds_to_others = count_fused_bonds_to_others(G, path)

        return linear_length, parallel_bonds, linear_segments, overlayers, fused_bonds_to_others

    def count_overlayers(G, path, ring_layers):
        overlayers = 0
        path_set = set(path)
        path_layer = ring_layers[path[0]]  # All nodes in path should have the same layer
        for node in path:
            neighbors = set(G.neighbors(node))
            for neighbor in neighbors - path_set:
                if ring_layers[neighbor] != path_layer:
                    overlayers += 1
        return overlayers

    def count_fused_bonds_to_others(G, path):
        path_set = set(path)
        fused_bonds = 0
        for node in path:
            neighbors = set(G.neighbors(node))
            external_neighbors = neighbors - path_set
            fused_bonds += len(external_neighbors)
        return fused_bonds

    # Initialize variables
    best_segment = []
    best_linear_length = 0
    best_parallel_bonds = 0
    least_overlayers = float('inf')
    most_fused_bonds = -1

    # Iterate over all paths
    for start_node in G.nodes():
        for end_node in G.nodes():
            if start_node != end_node:
                paths = list(nx.all_simple_paths(G, start_node, end_node))
                for path in paths:
                    linear_length, parallel_bonds, linear_segments, overlayers, fused_bonds_to_others = segment_score(path)
                    if (linear_length > best_linear_length or
                        (linear_length == best_linear_length and parallel_bonds > best_parallel_bonds) or
                        (linear_length == best_linear_length and parallel_bonds == best_parallel_bonds and overlayers < least_overlayers) or
                        (linear_length == best_linear_length and parallel_bonds == best_parallel_bonds and overlayers == least_overlayers and fused_bonds_to_others > most_fused_bonds)):
                        best_segment = path
                        best_linear_length = linear_length
                        best_parallel_bonds = parallel_bonds
                        least_overlayers = overlayers
                        most_fused_bonds = fused_bonds_to_others

    # Ensure the reference segment starts from a terminal node if possible
    if not best_segment:
        return []
    terminal_nodes = [node for node in best_segment if G.degree(node) == 1]
    if terminal_nodes:
        if best_segment[0] not in terminal_nodes:
            best_segment.reverse()

    return best_segment
def find_reference_segment(G, ring_centers, angle_threshold=AngleThreshold):
    ring_layers = assign_layers(ring_centers)
    
    def segment_score(path, G, ring_centers, angle_threshold):
        # if len(path) < 3:
        #     overlayers = count_overlayers(G, path)
        #     fused_bonds_to_others = count_fused_bonds_to_others(G, path)
        #     return len(path), 0, 0, overlayers, fused_bonds_to_others

        linear_length = 0
        current_linear_length = 1
        parallel_bonds = 0
        linear_segments = 0

        # Calculate the segment vector
        segment_vector = ring_centers[path[-1]] - ring_centers[path[0]]

        # Iterate over all edges in the graph to count parallel bonds
        for edge in G.edges():
            edge_vector = ring_centers[edge[1]] - ring_centers[edge[0]]
            angle = angle_between_vectors(segment_vector, edge_vector)

            # print(f"Path: {path}, Graph edge: {edge[0]} -> {edge[1]}, Angle: {angle}")

            # Check for parallel or anti-parallel vectors
            if abs(angle) <= angle_threshold or abs(angle - 180) <= angle_threshold:
                parallel_bonds += 1

        # Iterate over the path to calculate linear segments and lengths
        for i in range(len(path) - 1):
            edge_vector = ring_centers[path[i+1]] - ring_centers[path[i]]
            angle = angle_between_vectors(segment_vector, edge_vector)

            # print(f"Path segment: {path[i]} -> {path[i+1]}, Angle: {angle}")

            if abs(angle) <= angle_threshold:
                current_linear_length += 1
            else:
                if current_linear_length > linear_length:
                    linear_length = current_linear_length
                if current_linear_length > 1:
                    linear_segments += 1
                current_linear_length = 1

        if current_linear_length > linear_length:
            linear_length = current_linear_length
        if current_linear_length > 1:
            linear_segments += 1

        overlayers = count_overlayers(G, path)
        fused_bonds_to_others = count_fused_bonds_to_others(G, path)

        # print(f"Path: {path}, Linear Length: {linear_length}, Parallel Bonds: {parallel_bonds}, Linear Segments: {linear_segments}, Overlayers: {overlayers}, Fused Bonds to Others: {fused_bonds_to_others}")

        return linear_length, parallel_bonds, linear_segments, overlayers, fused_bonds_to_others
    
    def count_overlayers(G, path):
        overlayers = 0
        path_set = set(path)
        for node in path:
            neighbors = set(G.neighbors(node))
            overlayers += len(neighbors - path_set)
        return overlayers
    
    def count_fused_bonds_to_others(G, path):
        path_set = set(path)
        fused_bonds = 0
        for node in path:
            neighbors = set(G.neighbors(node))
            external_neighbors = neighbors - path_set
            fused_bonds += len(external_neighbors)
        return fused_bonds
    
    best_segment = []
    best_linear_length = 0
    best_parallel_bonds = 0
    best_linear_segments = 0
    least_overlayers = float('inf')
    most_fused_bonds = -1
    
    for start_node in G.nodes():
        for end_node in G.nodes():
            if start_node != end_node:
                paths = list(nx.all_simple_paths(G, start_node, end_node))
                for path in paths:
                    linear_length, parallel_bonds, linear_segments, overlayers, fused_bonds_to_others = segment_score(path, G, ring_centers, angle_threshold)
                    # print("PATH",path,"linear_length",linear_length,"parallel_bonds",parallel_bonds,"linear_segments",linear_segments,"overlayers",overlayers,"fused_bonds",fused_bonds_to_others) 
                    if (linear_length > best_linear_length or
                        (linear_length == best_linear_length and parallel_bonds > best_parallel_bonds) or
                        # (linear_length == best_linear_length and parallel_bonds == best_parallel_bonds and overlayers <= least_overlayers) or
                        (linear_length == best_linear_length and parallel_bonds == best_parallel_bonds and fused_bonds_to_others > most_fused_bonds) or
                        (linear_length == best_linear_length and parallel_bonds == best_parallel_bonds and overlayers == least_overlayers and fused_bonds_to_others > most_fused_bonds)):
                        best_segment = path
                        best_linear_length = linear_length
                        best_parallel_bonds = parallel_bonds
                        best_linear_segments = linear_segments
                        least_overlayers = overlayers
                        most_fused_bonds = fused_bonds_to_others
                        # print("best segment", best_segment,"best_linear_length",best_linear_length,"best_parallel_bonds",best_parallel_bonds,"best_linear_segments",best_linear_segments,"least_overlayers",least_overlayers,"most_fused_bonds",most_fused_bonds) 
    
    # Extract the longest linear subsegment from the best segment
    if not best_segment:
        return []
    
    longest_linear_subsegment = []
    current_subsegment = [best_segment[0]]
    for i in range(1, len(best_segment)):
        if len(current_subsegment) == 1 or angle_between_vectors(
            ring_centers[current_subsegment[-1]] - ring_centers[current_subsegment[-2]],
            ring_centers[best_segment[i]] - ring_centers[current_subsegment[-1]]
        ) <= angle_threshold:
            current_subsegment.append(best_segment[i])
        else:
            if len(current_subsegment) > len(longest_linear_subsegment):
                longest_linear_subsegment = current_subsegment
            current_subsegment = [best_segment[i-1], best_segment[i]]
    
    if len(current_subsegment) > len(longest_linear_subsegment):
        longest_linear_subsegment = current_subsegment
    
    # Ensure the reference segment starts from a terminal node if possible
    terminal_nodes = [node for node in longest_linear_subsegment if G.degree(node) == 1]
    if terminal_nodes:
        if longest_linear_subsegment[0] not in terminal_nodes:
            longest_linear_subsegment.reverse()
    
    return longest_linear_subsegment

def calculate_angles_to_linear_segment_OLD(G, ring_centers, linear_segment):
    angles = {}
    if len(linear_segment) < 2:
        return angles
    
    segment_vector = ring_centers[linear_segment[-1]] - ring_centers[linear_segment[0]]
    
    for node in G.nodes():
        if node not in linear_segment:
            nearest_segment_node = min(linear_segment, key=lambda x: np.linalg.norm(ring_centers[node] - ring_centers[x]))
            node_vector = ring_centers[node] - ring_centers[nearest_segment_node]
            angle = angle_between_vectors(segment_vector, node_vector)
            angles[node] = angle
    
    return angles
def calculate_angles_to_linear_segment_OLD3(G, ring_centers, linear_segment):
    angles = {}
    if len(linear_segment) < 2:
        return angles
    
    # Get all edges in the graph
    all_edges = list(G.edges())
    
    # Calculate the angles for each edge in the linear segment
    for i in range(len(linear_segment) - 1):
        linear_edge = (linear_segment[i], linear_segment[i+1])
        linear_edge_vector = ring_centers[linear_edge[1]] - ring_centers[linear_edge[0]]

        # print("Linear edge", linear_edge, "vector", linear_edge_vector)
        
        for edge in all_edges:
            # Skip edges that are part of the linear segment
            if edge[0] in linear_segment and edge[1] in linear_segment:
                continue
            
            edge_vector = ring_centers[edge[1]] - ring_centers[edge[0]]
            
            # Find the closest point on the linear edge to the midpoint of the current edge
            edge_midpoint = (ring_centers[edge[0]] + ring_centers[edge[1]]) / 2
            linear_edge_start = ring_centers[linear_edge[0]]
            linear_edge_end = ring_centers[linear_edge[1]]
            
            t = np.dot(edge_midpoint - linear_edge_start, linear_edge_vector) / np.dot(linear_edge_vector, linear_edge_vector)
            t = max(0, min(1, t))  # Clamp t to [0, 1]
            closest_point = linear_edge_start + t * linear_edge_vector
            
            connection_vector = edge_midpoint - closest_point
            
            angle = angle_between_vectors(linear_edge_vector, edge_vector)
            
            # print("angle",angle)

            # Store the angle for each edge
            edge_key = tuple(sorted(edge))  # Ensure consistent key regardless of edge direction
            if edge_key not in angles or angle < angles[edge_key]:
                angles[edge_key] = angle
            
            # print(f"Edge {edge_key}, Angle: {angle}")
    
    return angles
def calculate_angles_to_linear_segment2(G, ring_centers, linear_segment):
    angles = {}
    if len(linear_segment) < 2:
        return angles
    
    # # Calculate the angles for each edge in the linear segment
    # for i in range(len(linear_segment) - 1):
    #     edge = (linear_segment[i], linear_segment[i+1])
    #     edge_vector = ring_centers[edge[1]] - ring_centers[edge[0]]

    #     print("edge",edge,"vector",edge_vector)
        
    #     for node in G.nodes():
    #         if node not in linear_segment:
    #             # Find the closest edge in the linear segment
    #             closest_edge = min(zip(linear_segment, linear_segment[1:]), 
    #                                key=lambda x: np.linalg.norm(ring_centers[node] - (ring_centers[x[0]] + ring_centers[x[1]])/2))
                
    #             node_vector = ring_centers[node] - ring_centers[closest_edge[0]]
    #             angle = angle_between_vectors(edge_vector, node_vector)
                
    #             # Store the smallest angle for each node
    #             if node not in angles or angle < angles[node]:
    #                 angles[node] = angle
    #             print(angles)
    
    # return angles
     # Get all edges in the graph
    all_edges = list(G.edges())
    
    # Calculate the angles for each edge in the linear segment
    for i in range(len(linear_segment) - 1):
        linear_edge = (linear_segment[i], linear_segment[i+1])
        linear_edge_vector = ring_centers[linear_edge[1]] - ring_centers[linear_edge[0]]

        print("Linear edge", linear_edge, "vector", linear_edge_vector)
        
        for edge in all_edges:
            # Skip edges that are part of the linear segment
            if edge[0] in linear_segment and edge[1] in linear_segment:
                continue
            
            edge_vector = ring_centers[edge[1]] - ring_centers[edge[0]]
            
            # Find the closest point on the linear edge to the midpoint of the current edge
            edge_midpoint = (ring_centers[edge[0]] + ring_centers[edge[1]]) / 2
            linear_edge_start = ring_centers[linear_edge[0]]
            linear_edge_end = ring_centers[linear_edge[1]]
            
            t = np.dot(edge_midpoint - linear_edge_start, linear_edge_vector) / np.dot(linear_edge_vector, linear_edge_vector)
            t = max(0, min(1, t))  # Clamp t to [0, 1]
            closest_point = linear_edge_start + t * linear_edge_vector
            
            connection_vector = edge_midpoint - closest_point
            
            angle = angle_between_vectors(linear_edge_vector, edge_vector)
            
            # Store the angle for each edge
            edge_key = tuple(sorted(edge))  # Ensure consistent key regardless of edge direction
            if edge_key not in angles or angle < angles[edge_key]:
                angles[edge_key] = angle
            
            print(f"Edge {edge_key}, Angle: {angle}")
    
    return angles
def calculate_angles_to_linear_segment3(G, ring_centers, linear_segment):
    angles = {}
    if len(linear_segment) < 2:
        return angles
    
    # Get all edges in the graph
    all_edges = list(G.edges())
    
    # Calculate the vector of the linear segment
    linear_vector = ring_centers[linear_segment[-1]] - ring_centers[linear_segment[0]]
    
    for edge in all_edges:
        # Skip edges that are part of the linear segment
        if edge[0] in linear_segment and edge[1] in linear_segment:
            continue
        
        edge_vector = ring_centers[edge[1]] - ring_centers[edge[0]]
        
        # Find the closest point in the linear segment
        closest_point_index = find_closest_point(edge, linear_segment, ring_centers)
        
        # Calculate angle between edge and linear segment
        if closest_point_index == 0:
            segment_vector = ring_centers[linear_segment[1]] - ring_centers[linear_segment[0]]
        elif closest_point_index == len(linear_segment) - 1:
            segment_vector = ring_centers[linear_segment[-1]] - ring_centers[linear_segment[-2]]
        else:
            segment_vector = ring_centers[linear_segment[closest_point_index+1]] - ring_centers[linear_segment[closest_point_index-1]]
        
        angle = angle_between_vectors(segment_vector, edge_vector)
        
        # Store the angle for each edge
        edge_key = tuple(sorted(edge))  # Ensure consistent key regardless of edge direction
        angles[edge_key] = angle
        
        print(f"Edge {edge_key}, Closest linear point {linear_segment[closest_point_index]}, Angle: {angle:.2f}")
    
    return angles
def calculate_angles_to_linear_segment4(G, ring_centers, linear_segment):
    angles = {}
    if len(linear_segment) < 2:
        return angles
    
    # Get all edges in the graph
    all_edges = list(G.edges())
    
    # Calculate the vector of the linear segment
    linear_vector = ring_centers[linear_segment[-1]] - ring_centers[linear_segment[0]]
    
    for edge in all_edges:
        # Skip edges that are part of the linear segment
        if edge[0] in linear_segment and edge[1] in linear_segment:
            continue
        
        # Find the closest points in the linear segment for both ends of the edge
        closest_point_0 = find_closest_point(edge[0], linear_segment, ring_centers)
        closest_point_1 = find_closest_point(edge[1], linear_segment, ring_centers)
        
        # Determine which end of the edge is closer to the linear segment
        if closest_point_0 < closest_point_1 or (closest_point_0 == closest_point_1 and 
           np.linalg.norm(ring_centers[edge[0]] - ring_centers[linear_segment[closest_point_0]]) < 
           np.linalg.norm(ring_centers[edge[1]] - ring_centers[linear_segment[closest_point_1]])):
            edge_vector = ring_centers[edge[1]] - ring_centers[edge[0]]
            closest_point_index = closest_point_0
        else:
            edge_vector = ring_centers[edge[0]] - ring_centers[edge[1]]
            closest_point_index = closest_point_1
        
        # Calculate segment vector
        if closest_point_index == 0:
            segment_vector = ring_centers[linear_segment[1]] - ring_centers[linear_segment[0]]
        elif closest_point_index == len(linear_segment) - 1:
            segment_vector = ring_centers[linear_segment[-1]] - ring_centers[linear_segment[-2]]
        else:
            segment_vector = ring_centers[linear_segment[closest_point_index+1]] - ring_centers[linear_segment[closest_point_index-1]]
        
        angle = angle_between_vectors(segment_vector, edge_vector)
        
        
        # Store the angle for each edge
        edge_key = tuple(sorted(edge))  # Ensure consistent key regardless of edge direction
        angles[edge_key] = angle
        
        print(f"Edge {edge_key}, Closest linear point {linear_segment[closest_point_index]}, Angle: {angle:.2f}")
    
    return angles
def calculate_angles_to_linear_segment5(G, ring_centers, linear_segment):
    edge_info = {}
    if len(linear_segment) < 2:
        return edge_info
    
    all_edges = list(G.edges())
    linear_vector = ring_centers[linear_segment[-1]] - ring_centers[linear_segment[0]]
    
    for edge in all_edges:
        if edge[0] in linear_segment and edge[1] in linear_segment:
            continue
        
        closest_point_0 = find_closest_point(edge[0], linear_segment, ring_centers)
        closest_point_1 = find_closest_point(edge[1], linear_segment, ring_centers)
        
        if closest_point_0 < closest_point_1 or (closest_point_0 == closest_point_1 and 
           np.linalg.norm(ring_centers[edge[0]] - ring_centers[linear_segment[closest_point_0]]) < 
           np.linalg.norm(ring_centers[edge[1]] - ring_centers[linear_segment[closest_point_1]])):
            edge_vector = ring_centers[edge[1]] - ring_centers[edge[0]]
            closest_point_index = closest_point_0
            direction = (edge[0], edge[1])
        else:
            edge_vector = ring_centers[edge[0]] - ring_centers[edge[1]]
            closest_point_index = closest_point_1
            direction = (edge[1], edge[0])
        
        if closest_point_index == 0:
            segment_vector = ring_centers[linear_segment[1]] - ring_centers[linear_segment[0]]
        elif closest_point_index == len(linear_segment) - 1:
            segment_vector = ring_centers[linear_segment[-1]] - ring_centers[linear_segment[-2]]
        else:
            segment_vector = ring_centers[linear_segment[closest_point_index+1]] - ring_centers[linear_segment[closest_point_index-1]]
        
        angle = angle_between_vectors(segment_vector, edge_vector)
        
        edge_key = tuple(sorted(edge))
        edge_info[edge_key] = {
            'angle': angle,
            'closest_point': linear_segment[closest_point_index],
            'direction': direction
        }
        
    return edge_info
def calculate_angles_to_linear_segment(G, ring_centers, reference_segment):
    edge_info = {}
    
    # Calculate the direction of the reference segment
    ref_direction = ring_centers[reference_segment[-1]] - ring_centers[reference_segment[0]]
    ref_direction /= np.linalg.norm(ref_direction)
    
    for edge in G.edges():
        node1, node2 = edge
        edge_vector = ring_centers[node2] - ring_centers[node1]
        edge_vector /= np.linalg.norm(edge_vector)
        
        # Calculate the angle between the edge and the reference segment
        angle = np.degrees(np.arccos(np.clip(np.dot(ref_direction, edge_vector), -1.0, 1.0)))
        
        edge_info[edge] = {'angle': angle}
    
    return edge_info
# def find_closest_point(edge, linear_segment, ring_centers):
#     edge_center = (ring_centers[edge[0]] + ring_centers[edge[1]]) / 2
#     distances = [np.linalg.norm(edge_center - ring_centers[i]) for i in linear_segment]
#     return np.argmin(distances)
def find_closest_point(node, linear_segment, ring_centers):
    distances = [np.linalg.norm(ring_centers[node] - ring_centers[i]) for i in linear_segment]
    return np.argmin(distances)
def is_path_linear_OLD(path, ring_centers, AngleThreshold):
    if len(path) < 3:
        return True
    for i in range(len(path) - 2):
        v1 = ring_centers[path[i+1]] - ring_centers[path[i]]
        v2 = ring_centers[path[i+2]] - ring_centers[path[i+1]]
        angle = angle_between_vectors(v1, v2)
        # print("path and angle",path,angle,"vector1",v1,"vector2",v2,i,"Length",len(path))
        # if abs(180 - angle) > angle_threshold:
        if abs(angle) > angle_threshold:
            return False
    return True
def is_path_linear(path, ring_centers, AngleThreshold):
    if len(path) < 3:
        return True
    for i in range(len(path) - 2):
        v1 = ring_centers[path[i+1]] - ring_centers[path[i]]
        v2 = ring_centers[path[i+2]] - ring_centers[path[i+1]]
        angle = angle_between_vectors(v1, v2)
        if abs(angle) > angle_threshold:
            return False
    return True
def calculate_bla_and_baa(mol, aromatic_rings):
    blas = []
    baas = []

    for ring in aromatic_rings:
        bond_lengths = []
        bond_angles = []
        for i in range(len(ring)):
            a1 = ring[i]
            a2 = ring[(i + 1) % len(ring)]
            a3 = ring[(i + 2) % len(ring)]
            p1 = mol.GetConformer().GetAtomPosition(a1)
            p2 = mol.GetConformer().GetAtomPosition(a2)
            p3 = mol.GetConformer().GetAtomPosition(a3)

            dist = np.linalg.norm(np.array(p1) - np.array(p2))
            bond_lengths.append(dist)
            angle = calculate_bond_angle(p1, p2, p3)
            bond_angles.append(angle)

        bla = sum(abs(bond_lengths[i] - bond_lengths[(i + 1) % len(bond_lengths)]) for i in range(len(bond_lengths))) / len(bond_lengths)
        blas.append(bla)

        baa = sum(abs(bond_angles[i] - bond_angles[(i + 1) % len(bond_angles)]) for i in range(len(bond_angles))) / len(bond_angles)
        baas.append(baa)

    return blas, baas
def calculate_max_z_displacement(mol):
    z_coords = [mol.GetConformer().GetAtomPosition(atom.GetIdx()).z for atom in mol.GetAtoms()]
    max_z_displacement = max(z_coords) - min(z_coords)

        # Calculate the RMSD of z-coordinates
    mean_z = np.mean(z_coords)
    rmsd_z = np.sqrt(np.mean((np.array(z_coords) - mean_z) ** 2))

     # Calculate the Mean Absolute Deviation (MAD) of z-coordinates
    mad_z = np.mean(np.abs(np.array(z_coords) - mean_z))

    return max_z_displacement, mean_z, rmsd_z, mad_z
def calculate_projected_area1(mol):
    conformer = mol.GetConformer()
    coords = np.array([conformer.GetAtomPosition(i) for i in range(mol.GetNumAtoms())])
    xy_coords = coords[:, :2]  # Project onto XY plane
    hull = ConvexHull(xy_coords)
    area = hull.volume  # For 2D, 'volume' returns the area
    return area
def calculate_projected_area(mol):
    conformer = mol.GetConformer()
    coords = np.array([conformer.GetAtomPosition(i) for i in range(mol.GetNumAtoms())])
    
    # Calculate the range of coordinates in each axis
    x_range = np.ptp(coords[:, 0])  # Range of X values
    y_range = np.ptp(coords[:, 1])  # Range of Y values
    z_range = np.ptp(coords[:, 2])  # Range of Z values
    
    # Check which coordinate has minimal variance and exclude it
    if x_range < y_range and x_range < z_range:
        # Project onto YZ plane
        projected_coords = coords[:, 1:]
    elif y_range < x_range and y_range < z_range:
        # Project onto XZ plane
        projected_coords = coords[:, [0, 2]]
    else:
        # Project onto XY plane
        projected_coords = coords[:, :2]
    
    hull = ConvexHull(projected_coords)
    area = hull.volume  # For 2D, 'volume' returns the area
    return area

def calculate_max_cc_distance(mol):
    conformer = mol.GetConformer()
    carbon_indices = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() == 'C']
    max_distance_cc_distance = 0.0
    for i in carbon_indices:
        for j in carbon_indices:
            if i < j:
                pos_i = np.array(conformer.GetAtomPosition(i))[:2]
                pos_j = np.array(conformer.GetAtomPosition(j))[:2]
                distance = np.linalg.norm(pos_i - pos_j)
                max_distance_cc_distance = max(max_distance_cc_distance, distance)
    return max_distance_cc_distance
def calculate_asymmetry(mol):
    conformer = mol.GetConformer()
    coords = np.array([conformer.GetAtomPosition(i) for i in range(mol.GetNumAtoms())])
    centroid = np.mean(coords, axis=0)
    deviations = coords - centroid
    asymmetry = np.sum(np.linalg.norm(deviations, axis=1))
    return asymmetry
def extract_coordinates(log_file):
    with open(log_file, 'r') as file:
        lines = file.readlines()
    
    coords_start = False
    coordinates = []
    element_symbols = {1: 'H', 6: 'C', 7: 'N', 8: 'O'}  # Add more elements as needed

    for line in lines:
        if 'Standard orientation:' in line:
            coords_start = True
            coordinates = []  # Clear previous coordinates
        elif 'Rotational constants' in line:
            coords_start = False
        
        if coords_start:
            try:
                data = line.split()
                if len(data) == 6 and data[0].isdigit():
                    atom_num = int(data[1])
                    if atom_num in element_symbols:
                        coordinates.append([element_symbols[atom_num], float(data[3]), float(data[4]), float(data[5])])
            except:
                continue
    
    return coordinates
def write_xyz(coordinates, xyz_file):
    with open(xyz_file, 'w') as file:
        file.write(f"{len(coordinates)}\n")
        file.write("Converted from Gaussian log file\n")
        for atom in coordinates:
            file.write(f"{atom[0]} {atom[1]:.6f} {atom[2]:.6f} {atom[3]:.6f}\n")
def convert_log_to_xyz(log_directory):
    for filename in os.listdir(log_directory):
        if filename.endswith(".log"):
            log_file = os.path.join(log_directory, filename)
            coordinates = extract_coordinates(log_file)
            if coordinates:
                xyz_file = os.path.join(log_directory, filename.replace('.log', '.xyz'))
                write_xyz(coordinates, xyz_file)
                print(f"Converted {log_file} to {xyz_file}")
def calculate_pyramidalization_angle(central_atom, neighbors):
    """
    Calculate the pyramidalization angle for a given central atom.

    Parameters:
    central_atom (numpy array): 1x3 array of the central atom's coordinates.
    neighbors (list of numpy arrays): List of three 1x3 arrays of the neighboring atoms' coordinates.

    Returns:
    float: Pyramidalization angle in degrees.
    """
    # Get the vectors in the plane defined by the neighbors
    v1 = neighbors[1] - neighbors[0]
    v2 = neighbors[2] - neighbors[0]
    
    # Calculate the normal vector to the plane
    normal = np.cross(v1, v2)
    normal_norm = normal / np.linalg.norm(normal)  # Normalize
    
    # Project the central atom onto the plane to find the perpendicular distance
    d = np.dot(central_atom - neighbors[0], normal_norm)
    
    # Calculate the average bond length between the central atom and its neighbors
    bond_lengths = [np.linalg.norm(central_atom - neighbor) for neighbor in neighbors]
    R = np.mean(bond_lengths)
    
    # Calculate the pyramidalization angle in radians and then convert to degrees
    theta = np.arctan(d / R)
    theta_degrees = np.degrees(theta)
    
    return theta_degrees
def process_file(file):
    global min_energy_kcal_mol, min_xTB_energy_kcal_mol

    log_file = os.path.join(directory, file)
    xyz_file = log_file.replace('.log', '.xyz')
    print(f"Processing file: {file}")
    if not os.path.exists(xyz_file):
        print(f"XYZ file not found for {log_file}")
        convert_log_to_xyz(directory)
        xyz_file = log_file.replace('.log', '.xyz')
        return None

    mol = Chem.MolFromXYZFile(xyz_file)
#    mol = Chem.Mol(mol)
    Chem.rdmolops.AssignStereochemistry(mol, cleanIt=True, force=True)
    rdDetermineBonds.DetermineBonds(mol)

    if mol is None:
        print(f"Failed to read molecule from {xyz_file}")
        return None

    # Initialize pyramidalization angles list
    pyramidalization_angles = []

    for atom in mol.GetAtoms():
        atom_idx = atom.GetIdx()

        # Find neighbors of the atom
        neighbors = [mol.GetConformer().GetAtomPosition(nbr.GetIdx()) for nbr in atom.GetNeighbors()]
        
        if len(neighbors) == 3:  # Pyramidalization requires 3 neighbors
            central_atom = mol.GetConformer().GetAtomPosition(atom_idx)
            neighbors_coords = [np.array(nbr) for nbr in neighbors]
            central_atom_coords = np.array(central_atom)

            # Calculate the pyramidalization angle
            pyramidalization_angle = calculate_pyramidalization_angle(central_atom_coords, neighbors_coords)
            pyramidalization_angles.append(pyramidalization_angle)

    # At this point, you have a list of pyramidalization angles
    # You can now use them as required, e.g., calculate their mean or RMSD

    mean_pyramidalization = np.mean(pyramidalization_angles) if pyramidalization_angles else float('nan')
    rmsd_pyramidalization = np.sqrt(np.mean(np.square(pyramidalization_angles))) if pyramidalization_angles else float('nan')

    #return mean_pyramidalization, rmsd_pyramidalization

    # energy, energy_kcal_mol, homo_energy, lumo_energy, band_gap_eV = extract_energy(log_file)
    energy, energy_kcal_mol, homo_energy, lumo_energy, band_gap_eV, total_dipole, electronic_spatial_extent, max_mulliken_charge, min_mulliken_charge, degrees_of_freedom, quadrupole_moment, traceless_quadrupole_moment, octapole_moment, hexadecapole_moment = extract_energy(log_file)
    xtb_raw_energy = xtb_raw_values.get(file, float('nan'))
    xtb_raw_energy_kcal_mol = xtb_raw_energy * 627.5095

    D4_rel_energy = D4_rel_values.get(file, float('nan'))

    bonded_carbons = find_bonded_carbons(mol)
    dihedrals, bond_angles, sum_less_90, sum_greater_90, count_less_90, count_greater_90, sum_abs_120_minus_angle, count_angles = find_dihedrals_and_bond_angles(mol, bonded_carbons)
    total_hydrogen_distance, total_H_distance2, total_countH_under5 = find_hydrogen_distances(mol)
    mean_CC_distance, rmsd_bond_lengths = calculate_rmsd_bond_lengths(mol, bonded_carbons)
    aromatic_rings = detect_aromatic_rings(mol)
    num_aromatic_rings = len(aromatic_rings)
    blas, baas = calculate_bla_and_baa(mol, aromatic_rings)
    total_dpo, longest_linear_path = calculate_dpo(mol)
    print(xyz_file,total_dpo)
    mean_bla = np.mean(blas) if blas else float('nan')
    mean_baa = np.mean(baas) if baas else float('nan')
    max_z_displacement, mean_z, rmsd_z, mad_z = calculate_max_z_displacement(mol)
    area = calculate_projected_area(mol)
    max_cc_distance = calculate_max_cc_distance(mol)
    asymmetry = calculate_asymmetry(mol)
    
    print(f"Finished processing file: {file}")

    return (file, energy_kcal_mol, D4_rel_energy, homo_energy, lumo_energy, band_gap_eV, xtb_raw_energy, sum_less_90, count_less_90, sum_greater_90, count_greater_90,
            sum_abs_120_minus_angle, total_hydrogen_distance, mean_CC_distance, rmsd_bond_lengths,
            mean_bla, mean_baa,max_z_displacement, mean_z, rmsd_z, mad_z, mean_pyramidalization, rmsd_pyramidalization,total_dpo, total_dipole,
            electronic_spatial_extent, max_mulliken_charge, min_mulliken_charge, degrees_of_freedom, area, max_cc_distance, asymmetry, longest_linear_path)
def write_results(results,directory,min_energy_kcal_mol,min_xTB_energy_kcal_mol):
    output_file = 'analysis_results.{}.csv'.format(directory)
    with open(output_file, 'a') as f:
        # write the header if the file is empty
        if os.path.getsize(output_file) == 0:
            f.write('file,energy_kcal_mol,rel_energy_kcal_mol,D4_rel_energy,homo_energy,lumo_energy,band_gap_eV,xtb_raw_energy,'
                    'xtb_rel_energy_kcal_mol,sum_less_90,count_less_90,sum_greater_90,count_greater_90,sum_abs_120_minus_angle,total_hydrogen_distance,'
                    'mean_CC_distance,rmsd_bond_lengths,mean_bla,mean_baa,max_z_displacement,mean_z,rmsd_z,mad_z,mean_pyramidalization,rmsd_pyramidalization,'
                    'total_dpo,total_dipole,electronic_spatial_extent,max_mulliken_charge,min_mulliken_charge,degrees_of_freedom,area,'
                    'max_cc_distance,asymmetry,longest_linear_path\n')


        for result in results:
            if result is not None:
                file, energy_kcal_mol, D4_rel_energy, homo_energy, lumo_energy, band_gap_eV, xtb_raw_energy, sum_less_90, count_less_90, sum_greater_90, count_greater_90, \
                sum_abs_120_minus_angle, total_hydrogen_distance, mean_CC_distance, rmsd_bond_lengths, mean_bla, mean_baa, max_z_displacement, mean_z, \
                rmsd_z, mad_z, mean_pyramidalization, rmsd_pyramidalization, total_dpo, total_dipole, electronic_spatial_extent, max_mulliken_charge, \
                 min_mulliken_charge, degrees_of_freedom, area, max_cc_distance, asymmetry, longest_linear_path = result

                rel_energy_kcal_mol = energy_kcal_mol - min_energy_kcal_mol
                xtb_rel_energy_kcal_mol = (xtb_raw_energy * 627.5095) - min_xTB_energy_kcal_mol

#                xtb_rel_energy_kcal_mol = xtb_raw_energy_kcal_mol - min_xTB_energy_kcal_mol

                f.write(f"{file},{energy_kcal_mol},{rel_energy_kcal_mol},{D4_rel_energy},{homo_energy},{lumo_energy},{band_gap_eV},"
                        f"{xtb_raw_energy},{xtb_rel_energy_kcal_mol},{sum_less_90:.2f},{count_less_90},{sum_greater_90:.2f},"
                        f"{count_greater_90},{sum_abs_120_minus_angle:.2f},{total_hydrogen_distance:.2f},{mean_CC_distance:.5f},"
                        f"{rmsd_bond_lengths:.5f},{mean_bla:.5f},{mean_baa},{max_z_displacement:.5f},{mean_z:.5f},{rmsd_z:.5f},"
                        f"{mad_z:.5f},{mean_pyramidalization:.5f},{rmsd_pyramidalization:.5f},{total_dpo},{total_dipole},"
                        f"{electronic_spatial_extent},{max_mulliken_charge},{min_mulliken_charge},{degrees_of_freedom},"
                        f"{area},{max_cc_distance:.5f},{asymmetry:.5f},{longest_linear_path}\n")
def load_existing_results(directory):
    output_file = f'analysis_results.{directory}.csv'
    existing_files = set()
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            for line in f.readlines()[1:]:
                existing_files.add(line.split(',')[0])
    return existing_files   

if __name__ == "__main__":
    start_time = time.time()
    print("starting")
    #directory = r'C44H24'
   # directory = r'C36H20'
    for directory in [r'C36H20',r'C38H20',r'C40H20',r'C40H22',r'C42H22',r'C44H24']:
    #for directory in [r'C36H20']:
    # for directory in [r'DPO_test']:
    # for directory in [r'ML_DPO_Data_test']:
    # for directory in [r'C44H24_subset']:
    # for directory in [r'COMPAS1']:
    #for directory in [r'PAFs']:
                #extract D4 energies
        lowercase_directory=directory.lower()
        D4_e_list='Ener_D4_list_log_{}.txt'.format(lowercase_directory)
        D4_rel_values = {}
        with open(D4_e_list, 'r') as D4_file:
       # with open('list_xls_c44h24_xtb_abs.txt', 'r') as xtb_file:
            for line in D4_file:
                filename, D4_rel_energy = line.split()
                D4_rel_values[filename] = float(D4_rel_energy)
        
        # Load xTB raw values from the text file
        if directory == r'C44H24':
            print("C44H24 is the directory for this analysis")
            xtb_e_list='list_xls_c44h24_xtb_abs.txt'
        else:
            xtb_e_list='list_xtb_abs_ener.txt'
        xtb_raw_values = {}
        with open(xtb_e_list, 'r') as xtb_file:
       # with open('list_xls_c44h24_xtb_abs.txt', 'r') as xtb_file:
            for line in xtb_file:
                filename, xtb_raw_energy = line.split()
                xtb_raw_values[filename] = float(xtb_raw_energy)
        

        # Find the minimum energy values across all files in parallel
        log_files = [f for f in os.listdir(directory) if f.endswith('.log')]  # Adjust number of files as needed
        existing_files = load_existing_results(directory)
        log_files = [file for file in log_files if file not in existing_files]

        # print("starting parallel processing")
        # num_processes = min(4, cpu_count())  # Use at most 4 processes
        # # with Pool(processes=cpu_count()) as pool:
        # with Pool(processes=num_processes) as pool:
        #     all_results = pool.map(process_file, log_files)
        # result = process_file(process_file)
        # if result is not None:
        #     all_results.append(result)
        # print("finished parallel processing")
#        print("Starting sequential processing")
#        all_results = []
#        for i, file in enumerate(log_files):
#            print(f"Processing file: {file}")
#            result = process_file(file)
#            if result is not None:
#                all_results.append(result)
#            if (i + 1) % 100 == 0:
#                min_energy_kcal_mol = min(res[1] for res in all_results if res[1] is not None) #May suffer from the iterative process occuring here - for the current draft it doesn't matter as that data is placeholder, but will need FIXING
#                min_xTB_energy_kcal_mol = min(res[2] * 627.5095 for res in all_results if res[2] is not None)
#                write_results(all_results,directory,min_energy_kcal_mol,min_xTB_energy_kcal_mol)
#                all_results = []
#            print(f"Finished processing file: {file}")
#        print("Finished sequential processing")

        # Parallel processing setup
        num_processes = min(4, cpu_count())  # Use at most 4 processes
        all_results = []

        def process_in_chunks(files):
            with Pool(processes=num_processes) as pool:
                chunk_results = pool.map(process_file, files)
                return [result for result in chunk_results if result is not None]

        # Process files in chunks of 100 in parallel
        for i in range(0, len(log_files), 100):
            chunk_files = log_files[i:i+100]
            print(f"Processing files {i+1} to {i+len(chunk_files)} in parallel...")
            chunk_results = process_in_chunks(chunk_files)
            all_results.extend(chunk_results)
    
            if chunk_results:
                min_energy_kcal_mol = min(res[1] for res in all_results if res[1] is not None)
                min_xTB_energy_kcal_mol = min(res[6] * 627.5095 for res in all_results if res[2] is not None)
                write_results(all_results, directory, min_energy_kcal_mol, min_xTB_energy_kcal_mol)
                all_results = []  # Reset the results after writing

        print("Finished processing all files.")

        if all_results:
            min_energy_kcal_mol = min(res[1] for res in all_results if res[1] is not None) #May suffer from the iterative process occuring here - for the current draft it doesn't matter as that data is placeholder, but will need FIXING
            min_xTB_energy_kcal_mol = min(res[6] * 627.5095 for res in all_results if res[2] is not None)
            write_results(all_results,directory,min_energy_kcal_mol,min_xTB_energy_kcal_mol)

        # # Filter out None results and calculate the minimum energies
        # all_results = [res for res in all_results if res is not None]
        # print(directory,all_results, (res[1] for res in all_results if res[1] is not None))
        # min_energy_kcal_mol = min(res[1] for res in all_results if res[1] is not None)
        # min_xTB_energy_kcal_mol = min(res[2] * 627.5095 for res in all_results if res[2] is not None)
        # # Write the results to the output file
        # write_results(all_results,directory)

        # Visualization part (if needed)
#        for result in all_results:
#            if result is not None:
#                file, *_ = result
#                mol = Chem.MolFromXYZFile(os.path.join(directory, file.replace('.log', '.xyz')))
#                if mol is not None:
#                    aromatic_rings = detect_aromatic_rings(mol)
#                    ring_centers = calculate_ring_centers(mol, aromatic_rings)
#                    G = build_ring_graph(mol, aromatic_rings, ring_centers)
#                    print(file, mol,G)
#                    visualize_graph(G, ring_centers)

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"\nTotal time taken: {elapsed_time:.2f} seconds")
