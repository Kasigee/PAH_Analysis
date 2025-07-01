import os
import time
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdDetermineBonds
from multiprocessing import Pool, cpu_count
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull


##############################################################################
# SETTINGS / CONSTANTS
##############################################################################

CHUNK_SIZE = 50  # how many files to process in each chunk
NUM_PROCESSES = min(4, cpu_count())  # or however many you want
AngleThreshold = 25  # Used in angle-based checks for 'linear' paths in ring etworks


##############################################################################
# CSV COLUMNS
##############################################################################
# The final CSV will have these columns, in this order.
ALL_COLUMNS = [
    'file',
    'energy_kcal_mol',
    'rel_energy_kcal_mol',
    'D4_rel_energy',
    'homo_energy',
    'lumo_energy',
    'band_gap_eV',
    'xtb_raw_energy',
    'xtb_rel_energy_kcal_mol',
    'sum_less_90',
    'count_less_90',
    'sum_greater_90',
    'count_greater_90',
    'sum_abs_120_minus_angle',
    'rmsd_bond_angle',
    'total_hydrogen_distance',
    'mean_CC_distance',
    'rmsd_bond_lengths',
    'mean_bla',
    'mean_baa',
    'max_z_displacement',
    'mean_z',
    'rmsd_z',
    'mad_z',
    'mean_pyramidalization',
    'rmsd_pyramidalization',
    'total_dpo',
    'total_dipole',
    'electronic_spatial_extent',
    'max_mulliken_charge',
    'min_mulliken_charge',
    'degrees_of_freedom',
    'area',
    'max_cc_distance',
    'asymmetry',
    'longest_linear_path',
    'wiener_val',
    'randic_val',
    'harary_val',
    'hyper_wiener_val',
    'abc_val',
    'ecc_val',
    'balaban_val',
    'zagreb1_val',
    'zagreb2_val',
    'cluj_val',
    'avg_homa_val',
    'ring_homas_val',
    'avg_homa_fused_val',
    'avg_homa_edge_val',
    'avg_homa2_val',
    'avg_homa3_val',
    'avg_homa4_val',
    'avg_homa5_val',
    'avg_homa6_val',
    'avg_homa7_val',
    'avg_homa8_val',
    'avg_homa9_val',
    'avg_homa10_val',
    'avg_homa11_val',
    'avg_homa12_val',
    'avg_homa13_val',
    'avg_homa14_val',
    'avg_homa15_val',
    'avg_homa16_val',
    'avg_homa17_val',
    'avg_homa18_val',
    'avg_homa19_val',
    'avg_homa20_val',
    'avg_homa21_val',
    'avg_homa22_val',
    'avg_homa23_val',
    'avg_homa24_val',
    'avg_homa25_val',
    'avg_homa26_val',
    'radius_gyr_val',
    'surf_val',
    'I_princ_1_val',
    'I_princ_2_val',
    'I_princ_3_val',
    'w3d_val',
    'bay_val'
]

##############################################################################
# ENERGY EXTRACTION
##############################################################################

def extract_energy(log_file):
    """
    Extract Gaussian log energies, HOMO/LUMO, band gap, Mulliken charges, dipoles, etc.
    from a Gaussian .log file.
    """
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
        # Final SCF energy
        if 'SCF Done' in line:
            last_scf_line = line

        # Occupied / Virtual MO energies
        if 'Alpha  occ. eigenvalues' in line or 'Beta  occ. eigenvalues' in line:
            occ_energies.extend([float(x) for x in line.split()[4:]])
        if 'Alpha virt. eigenvalues' in line or 'Beta virt. eigenvalues' in line:
            virt_energies.extend([float(x) for x in line.split()[4:]])

        # Mulliken charges
        if 'Mulliken charges:' in line:
            collecting_mulliken = True
            mulliken_charges = []
            continue

        if collecting_mulliken:
            if 'Sum of Mulliken charges' in line:
                collecting_mulliken = False
                continue
            else:
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        charge = float(parts[2])
                        mulliken_charges.append(charge)
                    except ValueError:
                        pass

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
            import re
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
            qm_line1 = lines[i+1]
            qm_line2 = lines[i+2]
            import re
            matches1 = re.findall(r'([XYZ]{2})=\s*([-+]?\d*\.\d+|\d+)', qm_line1)
            matches2 = re.findall(r'([XYZ]{2})=\s*([-+]?\d*\.\d+|\d+)', qm_line2)
            for key, value in (matches1 + matches2):
                quadrupole_moment[key] = float(value)

        # Traceless Quadrupole moment
        if 'Traceless Quadrupole moment (field-independent basis, Debye-Ang):' in line:
            tqm_line1 = lines[i+1]
            tqm_line2 = lines[i+2]
            import re
            matches1 = re.findall(r'([XYZ]{2})=\s*([-+]?\d*\.\d+|\d+)', tqm_line1)
            matches2 = re.findall(r'([XYZ]{2})=\s*([-+]?\d*\.\d+|\d+)', tqm_line2)
            for key, value in (matches1 + matches2):
                traceless_quadrupole_moment[key] = float(value)

        # Octapole moment
        if 'Octapole moment (field-independent basis, Debye-Ang**2):' in line:
            oct_line1 = lines[i+1]
            oct_line2 = lines[i+2]
            matches1 = re.findall(r'([XYZ]{3})=\s*([-+]?\d*\.\d+|\d+)', oct_line1)
            matches2 = re.findall(r'([XYZ]{3})=\s*([-+]?\d*\.\d+|\d+)', oct_line2)
            for key, value in (matches1 + matches2):
                octapole_moment[key] = float(value)

        # Hexadecapole moment
        if 'Hexadecapole moment (field-independent basis, Debye-Ang**3):' in line:
            hdp_line1 = lines[i+1]
            hdp_line2 = lines[i+2]
            hdp_line3 = lines[i+3]
            matches1 = re.findall(r'([XYZ]{4})=\s*([-+]?\d*\.\d+|\d+)', hdp_line1)
            matches2 = re.findall(r'([XYZ]{4})=\s*([-+]?\d*\.\d+|\d+)', hdp_line2)
            matches3 = re.findall(r'([XYZ]{4})=\s*([-+]?\d*\.\d+|\d+)', hdp_line3)
            for key, value in (matches1 + matches2 + matches3):
                hexadecapole_moment[key] = float(value)

    # Extract HOMO / LUMO
    if occ_energies:
        homo_energy = max(occ_energies)
    if virt_energies:
        lumo_energy = min(virt_energies)

    # Calculate band gap in eV
    band_gap_eV = None
    if (homo_energy is not None) and (lumo_energy is not None):
        band_gap_hartree = lumo_energy - homo_energy
        band_gap_eV = band_gap_hartree * 27.2114

    # Final SCF energy (kcal/mol)
    energy, energy_kcal_mol = None, None
    if last_scf_line:
        energy = float(last_scf_line.split()[4])
        energy_kcal_mol = energy * 627.5095

    # Mulliken charges
    max_mulliken_charge = max(mulliken_charges) if mulliken_charges else None
    min_mulliken_charge = min(mulliken_charges) if mulliken_charges else None

    return (energy,
            energy_kcal_mol,
            homo_energy,
            lumo_energy,
            band_gap_eV,
            total_dipole,
            electronic_spatial_extent,
            max_mulliken_charge,
            min_mulliken_charge,
            degrees_of_freedom,
            quadrupole_moment,
            traceless_quadrupole_moment,
            octapole_moment,
            hexadecapole_moment)


##############################################################################
# GEOMETRIC HELPERS
##############################################################################

def calculate_dihedral(p1, p2, p3, p4):
    b1 = p2 - p1
    b2 = p3 - p2
    b3 = p4 - p3

    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)

    n1 /= np.linalg.norm(n1)
    n2 /= np.linalg.norm(n2)

    cos_theta = np.dot(n1, n2)
    theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))

    dihedral_angle = np.degrees(theta)
    return dihedral_angle

def calculate_bond_angle(p1, p2, p3):
    v1 = p1 - p2
    v2 = p3 - p2

    v1 /= np.linalg.norm(v1)
    v2 /= np.linalg.norm(v2)

    cos_theta = np.dot(v1, v2)
    theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))

    bond_angle = np.degrees(theta)
    return bond_angle


##############################################################################
# BOND / GEOMETRIC ANALYSES
##############################################################################

def find_bonded_carbons(mol):
    """
    Return list of (atom1, atom2) indices for all C–C bonds in the molecule.
    """
    bonded_atoms = []
    bonds = mol.GetBonds()
    for bond in bonds:
        atom1 = bond.GetBeginAtomIdx()
        atom2 = bond.GetEndAtomIdx()
        if (mol.GetAtomWithIdx(atom1).GetAtomicNum() == 6 and
            mol.GetAtomWithIdx(atom2).GetAtomicNum() == 6):
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
    sum_less_90 = 0
    sum_greater_90 = 0
    sum_abs_120_minus_angle = 0
    sum_squared_deviation = 0
    rmsd_bond_angle = 0
    count_less_90 = 0
    count_greater_90 = 0
    count_angles = 0

    conf = mol.GetConformer()

    for a1 in bonded_carbons_dict:
        for a2 in bonded_carbons_dict[a1]:
            for a3 in bonded_carbons_dict[a2]:
                if a3 != a1:
                    # Bond angle
                    p1 = np.array(conf.GetAtomPosition(a1))
                    p2 = np.array(conf.GetAtomPosition(a2))
                    p3 = np.array(conf.GetAtomPosition(a3))
                    bond_angle = calculate_bond_angle(p1, p2, p3)
                    bond_angles.append((a1, a2, a3, bond_angle))
                    sum_abs_120_minus_angle += abs(120 - bond_angle)
                    sum_squared_deviation += (120 - bond_angle) ** 2  # Add squared deviation
                    count_angles += 1

                    # Dihedral
                    for a4 in bonded_carbons_dict[a3]:
                        if a4 not in (a2, a1):
                            dihedral_tuple = tuple(sorted((a1, a2, a3, a4)))
                            if dihedral_tuple not in unique_dihedrals:
                                unique_dihedrals.add(dihedral_tuple)
                                p4 = np.array(conf.GetAtomPosition(a4))
                                dihedral_angle = calculate_dihedral(p1, p2, p3, p4)
                                dihedrals.append((a1, a2, a3, a4, dihedral_angle))
                                if dihedral_angle < 90:
                                    sum_less_90 += dihedral_angle
                                    count_less_90 += 1
                                else:
                                    sum_greater_90 += 180 - dihedral_angle
                                    count_greater_90 += 1

    # Calculate RMSD of bond angles
    rmsd_bond_angle = (sum_squared_deviation / count_angles) ** 0.5 if count_angles > 0 else 0

    return (dihedrals,
            bond_angles,
            sum_less_90,
            sum_greater_90,
            count_less_90,
            count_greater_90,
            sum_abs_120_minus_angle,
            count_angles,
            rmsd_bond_angle)

def find_hydrogen_distances(mol, cutoff=2.5):
    """
    Sum of distances between all H–H pairs,
    plus the sum for only those under the cutoff,
    plus count of those under the cutoff.
    """
    conf = mol.GetConformer()
    hydrogen_atoms = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() == 1]
    total_distance = 0.0
    total_distance2 = 0.0
    countH_under2_5 = 0

    for i in range(len(hydrogen_atoms)):
        for j in range(i + 1, len(hydrogen_atoms)):
            h1 = np.array(conf.GetAtomPosition(hydrogen_atoms[i]))
            h2 = np.array(conf.GetAtomPosition(hydrogen_atoms[j]))
            dist = np.linalg.norm(h1 - h2)
            total_distance += dist
            if dist <= cutoff:
                total_distance2 += dist
                countH_under2_5 += 1

    return total_distance, total_distance2, countH_under2_5

def calculate_rmsd_bond_lengths(mol, bonded_carbons):
    """
    Mean and RMSD for all C–C bond lengths.
    """
    conf = mol.GetConformer()
    distances = []
    for a1, a2 in bonded_carbons:
        p1 = np.array(conf.GetAtomPosition(a1))
        p2 = np.array(conf.GetAtomPosition(a2))
        distances.append(np.linalg.norm(p1 - p2))

    if not distances:
        return float('nan'), float('nan')

    mean_distance = np.mean(distances)
    rmsd = np.sqrt(np.mean((np.array(distances) - mean_distance)**2))
    return mean_distance, rmsd

##############################################################################
# AROMATIC RING DETECTION / DPO CALC
##############################################################################

def detect_aromatic_rings(mol):
    """
    Return list of rings (each ring is a list of atom indices) 
    where the ring is declared aromatic by RDKit.
    """
    ri = mol.GetRingInfo()
    aromatic_rings = []
    for ring_atoms in ri.AtomRings():
        if all(mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in ring_atoms):
            aromatic_rings.append(ring_atoms)
    return aromatic_rings

def rings_are_fused(ring1, ring2):
    """
    Return True if two rings share >= 2 atoms (i.e. fused).
    """
    shared_atoms = set(ring1).intersection(set(ring2))
    return len(shared_atoms) >= 2

def calculate_ring_center(mol, ring):
    """
    Calculate the geometric center of the ring 
    (using only carbon atoms if present, else all atoms).
    """
    conf = mol.GetConformer()
    coords = [conf.GetAtomPosition(idx) for idx in ring if mol.GetAtomWithIdx(idx).GetAtomicNum() == 6]
    if not coords:
        coords = [conf.GetAtomPosition(idx) for idx in ring]
    return np.mean(coords, axis=0)

def build_ring_graph(mol, atom_rings):
    """
    Build an undirected graph of ring indices, adding edges if two rings are fused.
    """
    G = nx.Graph()
    ring_centers = [calculate_ring_center(mol, ring) for ring in atom_rings]
    
    for idx in range(len(atom_rings)):
        G.add_node(idx)
    
    for i in range(len(atom_rings)):
        for j in range(i+1, len(atom_rings)):
            if rings_are_fused(atom_rings[i], atom_rings[j]):
                G.add_edge(i, j)
    
    return G, ring_centers

def angle_between_vectors(v1, v2):
    """
    Return angle in degrees between two vectors v1 and v2.
    """
    dot_product = np.dot(v1, v2)
    mag = (np.linalg.norm(v1) * np.linalg.norm(v2))
    if mag < 1e-12:
        return 0.0
    cos_theta = np.clip(dot_product / mag, -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))

def calculate_angles_relative_to_reference(G, ring_centers, reference_segment):
    """
    For each edge in G, compute the angle between that edge vector
    and the vector spanning the reference_segment (start to end).
    """
    edge_info = {}
    if len(reference_segment) < 2:
        return edge_info
    
    # Vector from first ring in reference_segment to last ring
    ref_vector = ring_centers[reference_segment[-1]] - ring_centers[reference_segment[0]]
    ref_len = np.linalg.norm(ref_vector)
    if ref_len < 1e-12:
        return edge_info
    ref_vector /= ref_len  # normalize
    
    for edge in G.edges():
        node1, node2 = edge
        edge_vec = ring_centers[node2] - ring_centers[node1]
        edge_len = np.linalg.norm(edge_vec)
        if edge_len < 1e-12:
            angle = 0.0
        else:
            edge_vec /= edge_len
            angle = np.degrees(np.arccos(np.clip(np.dot(ref_vector, edge_vec), -1.0, 1.0)))
        
        # Store info
        edge_info[edge] = {
            'angle': angle,
            'direction': (node1, node2)
        }
        edge_info[(node2, node1)] = {
            'angle': angle,
            'direction': (node2, node1)
        }
    return edge_info

def get_fused_bonds_between_rings(mol, ring_idx1, ring_idx2, bond_rings):
    """
    Return set of bond indices that two ring indices share (fused).
    """
    bonds_ring1 = set(bond_rings[ring_idx1])
    bonds_ring2 = set(bond_rings[ring_idx2])
    return bonds_ring1.intersection(bonds_ring2)

def assign_dpo_to_angulated_segments(mol, G, reference_segment, edge_info, ring_centers, a, b, c, d):
    """
    The final function that assigns DPO to edges branching from the reference segment.
    """
    DPO_total = 0.0
    assigned_bonds = set()
    edge_categories = {}
    
    ring_info = mol.GetRingInfo()
    bond_rings = ring_info.BondRings()

    def get_base_category(angle):
        # Basic angle-based category
        if abs(angle - 60) <= AngleThreshold:
            return 'b'
        elif abs(angle - 120) <= AngleThreshold:
            return 'c'
        elif abs(angle - 0) <= AngleThreshold or abs(angle - 180) <= AngleThreshold:
            return 'a'
        else:
            return 'd'
    
    def calculate_dpo_value(category, path):
        # Weighted formula depending on the category
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
            return d  # fallback or "d" category
    
    # BFS/Queue approach: start from reference edges
    from collections import deque
    queue = deque()
    visited_edges = set()

    # Add edges in reference segment first with special category
    for i in range(len(reference_segment) - 1):
        edge = (reference_segment[i], reference_segment[i+1])
        if edge in edge_info:
            # Mark them as 'ref' or something akin to 'a0'
            edge_categories[edge] = {
                'category': f'a{i}',  # Just a label
                'dpo_value': 1 - i * a,
                'path': ['a'] * (i + 1),
                'angle': 0.0
            }
            # Add fused bonds to assigned
            fused_bonds = get_fused_bonds_between_rings(mol, reference_segment[i], reference_segment[i+1], bond_rings)
            for bond_idx in fused_bonds:
                assigned_bonds.add(bond_idx)
            visited_edges.add(edge)

    # Now push all edges from reference segment to queue
    for i in range(len(reference_segment) - 1):
        node1 = reference_segment[i]
        node2 = reference_segment[i+1]
        for neighbor in G.neighbors(node1):
            e = tuple(sorted((node1, neighbor)))
            if e not in visited_edges and e in edge_info:
                queue.append((e, edge_categories.get((node1,node2),{}).get('path',[])))
                visited_edges.add(e)
        for neighbor in G.neighbors(node2):
            e = tuple(sorted((node2, neighbor)))
            if e not in visited_edges and e in edge_info:
                queue.append((e, edge_categories.get((node1,node2),{}).get('path',[])))
                visited_edges.add(e)

    # Process the queue
    while queue:
        (n1, n2), path = queue.popleft()
        edge_data = edge_info[(n1, n2)]
        angle = edge_data['angle']
        category = get_base_category(angle)
        new_path = path + [category]
        dpo_val = calculate_dpo_value(category, new_path)

        # Save
        edge_categories[(n1, n2)] = {
            'category': category,
            'dpo_value': dpo_val,
            'path': new_path,
            'angle': angle
        }

        # Assign fused bonds
        fused_bonds = get_fused_bonds_between_rings(mol, n1, n2, bond_rings)
        for bond_idx in fused_bonds:
            if bond_idx not in assigned_bonds:
                DPO_total += dpo_val
                assigned_bonds.add(bond_idx)

        # BFS to neighbors
        for neighbor in G.neighbors(n2):
            e = tuple(sorted((n2, neighbor)))
            if e not in visited_edges and e in edge_info:
                queue.append((e, new_path))
                visited_edges.add(e)

    return DPO_total, assigned_bonds, edge_categories


def find_reference_segment(G, ring_centers, angle_threshold=AngleThreshold):
    """
    Heuristic to find a 'reference' path in the ring-graph, 
    maximizing some criteria (length, parallel bonds, minimal overlap, etc.).
    Then extract a linear subsegment from it.
    """
    def count_overlayers(G, path):
        """
        How many edges/nodes branch off from the path?
        """
        path_set = set(path)
        overlayers = 0
        for node in path:
            neighbors = set(G.neighbors(node))
            overlayers += len(neighbors - path_set)
        return overlayers
    
    def count_fused_bonds_to_others(G, path):
        """
        Count how many edges go from path-nodes to outside the path.
        """
        path_set = set(path)
        fused_bonds = 0
        for node in path:
            neighbors = set(G.neighbors(node))
            external_neighbors = neighbors - path_set
            fused_bonds += len(external_neighbors)
        return fused_bonds

    def segment_score(path):
        # "Linear length" is just the path length if angles are near 180
        # This is a simplistic approach
        if len(path) < 2:
            return 0, 0, 0, float('inf'), 0
        
        # a. 'linear_length' ~ length of path
        linear_length = len(path)

        # b. 'parallel_bonds': count edges whose angle is near 180 w.r.t. path direction
        #   We'll skip that for brevity here, or treat them as "just length"
        parallel_bonds = 0  # (not used in detail now)

        # c. overlayer count
        overlayers = count_overlayers(G, path)
        fused_bonds_to_others_ = count_fused_bonds_to_others(G, path)

        # Return a tuple to compare
        return (linear_length,
                parallel_bonds,
                1,  # linear_segments is not used in this minimal version
                overlayers,
                fused_bonds_to_others_)

    best_segment = []
    best_stats = (-1, -1, -1, float('inf'), -1)  # see segment_score signature

    all_nodes = list(G.nodes())
    for start_node in all_nodes:
        for end_node in all_nodes:
            if start_node == end_node:
                continue
            # get all simple paths
            paths = list(nx.all_simple_paths(G, start_node, end_node))
            for path in paths:
                stats = segment_score(path)
                # We want: bigger linear_length better,
                # then fewer overlayers better,
                # then more fused_bonds_to_others better
                # so we do a custom comparison
                (lin_len, p_bonds, lin_segs, ovl, fused_) = stats
                (best_lin_len, best_p_bonds, best_lin_segs, best_ovl, best_fused) = best_stats

                # Compare
                # 1) bigger lin_len
                if lin_len > best_lin_len:
                    best_segment = path
                    best_stats = stats
                elif lin_len == best_lin_len:
                    # 2) fewer overlayers
                    if ovl < best_ovl:
                        best_segment = path
                        best_stats = stats
                    elif ovl == best_ovl:
                        # 3) more fused
                        if fused_ > best_fused:
                            best_segment = path
                            best_stats = stats

    # Now we do an optional "extract the longest linear subsegment":
    # For simplicity, just return best_segment
    return best_segment


def is_path_linear(path, ring_centers, angle_threshold):
    """
    Check if consecutive segments in 'path' are 'straight enough' w.r.t. angle_threshold.
    """
    if len(path) < 3:
        return True
    for i in range(len(path) - 2):
        v1 = ring_centers[path[i+1]] - ring_centers[path[i]]
        v2 = ring_centers[path[i+2]] - ring_centers[path[i+1]]
        angle = angle_between_vectors(v1, v2)
        if abs(angle) > angle_threshold:
            return False
    return True


##############################################################################
# DPO CALC
##############################################################################

def calculate_dpo(mol, a=0.05, b=-1/4, c=1/3, d=1/3):
    """
    Master function to calculate 'DPO' for the PAH. 
    Uses:
      - detect_aromatic_rings
      - build_ring_graph
      - find_reference_segment
      - calculate_angles_relative_to_reference
      - assign_dpo_to_angulated_segments
    Returns total DPO, and length of "reference path".
    """
    aromatic_rings = detect_aromatic_rings(mol)
    if not aromatic_rings:
        return 0.0, 0  # No rings => no DPO

    G, ring_centers = build_ring_graph(mol, aromatic_rings)
    reference_segment = find_reference_segment(G, ring_centers, AngleThreshold)
    if len(reference_segment) < 2:
        return 0.0, len(reference_segment)

    edge_info = calculate_angles_relative_to_reference(G, ring_centers, reference_segment)
    DPO_total, assigned_bonds, edge_categories = assign_dpo_to_angulated_segments(
        mol, G, reference_segment, edge_info, ring_centers, a, b, c, d
    )
    return DPO_total, len(reference_segment)


##############################################################################
# BLA/BAA, Z-DISPLACEMENT, PROJECTED AREA, ETC.
##############################################################################

def calculate_bla_and_baa(mol, aromatic_rings):
    """
    Calculate Bond Length Alternation (BLA) and Bond Angle Alternation (BAA)
    for each ring, return lists.
    """
    conf = mol.GetConformer()
    blas = []
    baas = []

    for ring in aromatic_rings:
        bond_lengths = []
        bond_angles = []
        ring_list = list(ring)

        # consecutive triplets in ring (i, i+1, i+2) mod ring-len
        for i in range(len(ring_list)):
            a1 = ring_list[i]
            a2 = ring_list[(i + 1) % len(ring_list)]
            a3 = ring_list[(i + 2) % len(ring_list)]
            p1 = np.array(conf.GetAtomPosition(a1))
            p2 = np.array(conf.GetAtomPosition(a2))
            p3 = np.array(conf.GetAtomPosition(a3))

            dist = np.linalg.norm(p1 - p2)
            bond_lengths.append(dist)

            angle = calculate_bond_angle(p1, p2, p3)
            bond_angles.append(angle)

        if len(bond_lengths) > 1:
            # BLA = average of |L_i - L_{i+1}|
            bla = np.mean([abs(bond_lengths[i] - bond_lengths[(i + 1) % len(bond_lengths)])
                           for i in range(len(bond_lengths))])
        else:
            bla = float('nan')
        if len(bond_angles) > 1:
            baa = np.mean([abs(bond_angles[i] - bond_angles[(i + 1) % len(bond_angles)])
                           for i in range(len(bond_angles))])
        else:
            baa = float('nan')

        blas.append(bla)
        baas.append(baa)

    return blas, baas

def calculate_max_z_displacement(mol):
    """
    Max Z displacement, plus mean Z, RMSD(Z), and mean absolute deviation (MAD).
    """
    conf = mol.GetConformer()
    z_coords = [conf.GetAtomPosition(i).z for i in range(mol.GetNumAtoms())]
    max_z_disp = max(z_coords) - min(z_coords)
    mean_z = np.mean(z_coords)
    rmsd_z = np.sqrt(np.mean((z_coords - mean_z)**2))
    mad_z = np.mean(np.abs(z_coords - mean_z))
    return max_z_disp, mean_z, rmsd_z, mad_z

def calculate_projected_area(mol):
    """
    Compute the 2D convex hull area of the best-projected axis 
    (the plane where the molecule presumably looks largest).
    """
    conf = mol.GetConformer()
    coords = np.array([conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())])

    # Compare ranges in x, y, z
    x_range = np.ptp(coords[:, 0])
    y_range = np.ptp(coords[:, 1])
    z_range = np.ptp(coords[:, 2])

    # Use the "smallest" axis as the 'ignore' axis
    if x_range < y_range and x_range < z_range:
        # project onto YZ
        projected = coords[:, 1:]
    elif y_range < x_range and y_range < z_range:
        # project onto XZ
        projected = coords[:, [0, 2]]
    else:
        # default: project onto XY
        projected = coords[:, :2]

    if len(projected) < 3:
        return 0.0
    hull = ConvexHull(projected)
    return hull.volume  # In 2D, hull.volume = area

def calculate_max_cc_distance(mol):
    """
    Max distance (in XY plane) between any two carbon atoms.
    """
    conf = mol.GetConformer()
    carbon_indices = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() == 'C']
    max_dist = 0.0
    for i in range(len(carbon_indices)):
        for j in range(i+1, len(carbon_indices)):
            p1 = np.array(conf.GetAtomPosition(carbon_indices[i]))[:2]
            p2 = np.array(conf.GetAtomPosition(carbon_indices[j]))[:2]
            dist = np.linalg.norm(p1 - p2)
            if dist > max_dist:
                max_dist = dist
    return max_dist

def calculate_asymmetry(mol):
    """
    A quick measure of how spread out the coords are from the centroid.
    """
    conf = mol.GetConformer()
    coords = np.array([conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())])
    centroid = np.mean(coords, axis=0)
    devs = coords - centroid
    return np.sum(np.linalg.norm(devs, axis=1))


##############################################################################
# LOG -> XYZ CONVERSION
##############################################################################

def extract_coordinates(log_file):
    """
    Read 'Standard orientation' from Gaussian .log and parse atomic coords.
    """
    with open(log_file, 'r') as file:
        lines = file.readlines()

    coords_start = False
    coordinates = []
    # Map atomic numbers to symbols
    element_symbols = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 16: 'S', 9: 'F', 17: 'Cl'}

    for line in lines:
        if 'Standard orientation:' in line:
            coords_start = True
            coordinates = []  # reset
            continue
        if coords_start and 'Rotational constants' in line:
            coords_start = False

        if coords_start:
            parts = line.split()
            if len(parts) == 6 and parts[0].isdigit():
                try:
                    atom_num = int(parts[1])  # atomic number
                    x = float(parts[3])
                    y = float(parts[4])
                    z = float(parts[5])
                    symbol = element_symbols.get(atom_num, 'X')
                    coordinates.append([symbol, x, y, z])
                except ValueError:
                    pass
    return coordinates

def write_xyz(coordinates, xyz_file):
    """
    Write an XYZ file from a list of [symbol, x, y, z].
    """
    with open(xyz_file, 'w') as f:
        f.write(f"{len(coordinates)}\n")
        f.write("Converted from Gaussian log file\n")
        for atom in coordinates:
            f.write(f"{atom[0]} {atom[1]:.6f} {atom[2]:.6f} {atom[3]:.6f}\n")

def convert_log_to_xyz(log_directory):
    """
    Convert all .log files in a directory to .xyz using Standard orientation.
    """
    for filename in os.listdir(log_directory):
        if filename.endswith(".log"):
            log_file = os.path.join(log_directory, filename)
            coords = extract_coordinates(log_file)
            if coords:
                xyz_file = log_file.replace('.log', '.xyz')
                write_xyz(coords, xyz_file)
                print(f"Converted {log_file} to {xyz_file}")

def convert_log_to_xyz_single(log_file, xyz_file):
    """
    Convert single .log -> .xyz if possible.
    """
    coords = extract_coordinates(log_file)
    if coords:
        write_xyz(coords, xyz_file)


##############################################################################
# PYRAMIDALIZATION ANGLE (OPTIONAL EXAMPLE)
##############################################################################

def calculate_pyramidalization_angle(central_atom, neighbors):
    """
    For 3 neighbors, how far 'out of plane' the central atom is.
    """
    v1 = neighbors[1] - neighbors[0]
    v2 = neighbors[2] - neighbors[0]
    normal = np.cross(v1, v2)
    normal_norm = normal / np.linalg.norm(normal)
    d = np.dot(central_atom - neighbors[0], normal_norm)
    R = np.mean([np.linalg.norm(central_atom - nbr) for nbr in neighbors])
    theta = np.arctan2(d, R)
    return np.degrees(theta)


##############################################################################
# TOPOLOGICAL INDICES: WIENER, RANDIC, ETC.
##############################################################################

import networkx as nx
from rdkit import Chem

def mol_to_nx_graph(mol):
    """
    Convert an RDKit Mol into an undirected NetworkX graph.
    Nodes = atom indices.
    Edge if there's a bond between i, j.
    """
    G = nx.Graph()
    for atom in mol.GetAtoms():
        G.add_node(atom.GetIdx())
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        G.add_edge(i, j)
    return G

#def calculate_wiener_index(mol):
#    """
#    Wiener index = sum of shortest-path distances over all distinct pairs (i,j).
#    """
#    G = mol_to_nx_graph(mol)
#    # all-pairs BFS approach in networkx
#    distances = 0
#    # We iterate BFS from each node i, gather distances to j>i
#    for i in G.nodes():
#        sp_lengths = nx.shortest_path_length(G, source=i)
#        for j, dist in sp_lengths:
#            if j > i:
#                distances += dist
#    return distances


def calculate_wiener_index(mol):
    """
    Wiener index = sum of distances over all pairs of vertices in the bond graph.
    """
    G = nx.Graph()
    for atom in mol.GetAtoms():
        G.add_node(atom.GetIdx())
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        G.add_edge(i, j)
    # all-pairs shortest path
    sp = dict(nx.all_pairs_shortest_path_length(G))
    wiener = 0
    nodes = sorted(G.nodes())
    for i in range(len(nodes)):
        for j in range(i+1, len(nodes)):
            wiener += sp[nodes[i]][nodes[j]]
    return wiener

def calculate_randic_index(mol):
    """
    Randić index (connectivity index) = sum over edges (1 / sqrt(d(u)*d(v)) ).
    """
    G = mol_to_nx_graph(mol)
    # degrees
    deg = dict(G.degree())
    randic = 0.0
    for (u, v) in G.edges():
        randic += 1.0 / np.sqrt(deg[u]*deg[v])
    return randic

def harary_index(mol):
    """
    Harary index = sum( 1 / distance(i,j) ) for all i<j, i != j
    Distance = shortest path distance.
    If distance = 0 (same node), skip.
    """
    G = mol_to_nx_graph(mol)
    harary_sum = 0.0
    for i in G.nodes():
        lengths = nx.shortest_path_length(G, source=i)
        for j, dist in lengths.items():
            if j > i and dist > 0:
                harary_sum += 1.0 / dist
    return harary_sum

def hyper_wiener_index(mol):
    """
    Hyper-Wiener = 1/2 sum over i<j of [dist(i,j) + dist(i,j)^2].
    """
    G = mol_to_nx_graph(mol)
    hw_sum = 0
    for i in G.nodes():
        lengths = nx.shortest_path_length(G, source=i)
        for j, dist in lengths.items():
            if j > i:
                hw_sum += dist + dist*dist
    return 0.5 * hw_sum

def szeged_index(mol):
    """
    Szeged index = sum_{edges} [ n1(e)*n2(e) ],
    where removing edge e splits G into two components
    of sizes n1, n2 (with u in one, v in the other).
    """
    G = mol_to_nx_graph(mol)
    sz = 0
    for (u,v) in G.edges():
        G.remove_edge(u,v)
        # BFS from u to see how many vertices are in that component
        comp_u = set(nx.bfs_tree(G,u))
        n1 = len(comp_u)
        n2 = G.number_of_nodes() - n1
        sz += (n1 * n2)
        G.add_edge(u,v)  # restore
    return sz

def abc_index(mol):
    """
    ABC index = sum over edges of sqrt( (deg(u)+deg(v)-2) / (deg(u)*deg(v)) ).
    """
    G = mol_to_nx_graph(mol)
    deg = dict(G.degree())
    abc_sum = 0.0
    for (u,v) in G.edges():
        numerator = deg[u] + deg[v] - 2
        denominator = deg[u]*deg[v]
        if denominator > 0:
            val = (numerator / denominator) if numerator>0 else 0
            abc_sum += np.sqrt(val)
    return abc_sum

def eccentric_connectivity_index(mol):
    """
    ECI = sum over all vertices v of [deg(v)*ecc(v)],
    where ecc(v) = max( distance(v,x) ) for x in V.
    """
    G = mol_to_nx_graph(mol)
    deg = dict(G.degree())
    eci = 0
    for v in G.nodes():
        lengths = nx.shortest_path_length(G, source=v)
        ecc = max(lengths.values())  # maximum distance from v
        eci += deg[v]*ecc
    return eci

def balaban_index(mol):
    """
    Balaban's J = (e / (e - n + 1)) * sum(1/sqrt(d(u)*d(v))) over edges.
    """
    G = mol_to_nx_graph(mol)
    n = G.number_of_nodes()
    e = G.number_of_edges()
    deg = dict(G.degree())
    # Avoid division by zero if e == n-1
    denom = (e - n + 1)
    if denom == 0:
        return 0.0  # or fallback
    s = 0.0
    for (u,v) in G.edges():
        s += 1.0 / np.sqrt(deg[u]*deg[v])
    J = (e / denom) * s
    return J

def first_zagreb_index(mol):
    G = mol_to_nx_graph(mol)
    deg = dict(G.degree())
    return sum(d*d for d in deg.values())

def second_zagreb_index(mol):
    G = mol_to_nx_graph(mol)
    deg = dict(G.degree())
    # Common variant: sum_{(u,v)} d(u)*d(v)
    s = 0
    for (u,v) in G.edges():
        s += deg[u]*deg[v]
    return s

def cluj_index(mol):
    """
    One version of the Cluj index:
      sum_{(u,v) in E} [ 1 / sqrt(deg(u) + deg(v)) ]
    Reference (approx): E. V. Vasile, "Cluj Indices ..."

    For ring-laden systems, this is still O(n + e) once we build the graph.

    :param mol: RDKit Mol, with bonds perceived
    :return: float (Cluj index)
    """
    G = nx.Graph()
    for atom in mol.GetAtoms():
        G.add_node(atom.GetIdx())
    for bond in mol.GetBonds():
        G.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())

    deg = dict(G.degree())
    ci_sum = 0.0
    for (u, v) in G.edges():
        val = deg[u] + deg[v]
        if val > 0:
            ci_sum += 1.0 / np.sqrt(val)
    return ci_sum

def homa_aromatic_ringsOLD(mol, alpha=257.7, R_opt=1.388):
    """
    Compute average HOMA for all detected aromatic rings in 'mol'.
    :param alpha: typically 257.7 for C–C
    :param R_opt: ~1.388 A for ideal aromatic C–C
    :return: (average_homa, list_of_individual_homas)

    We skip rings not purely C–C or not recognized as aromatic by RDKit.
    """
    from rdkit import Chem
    import numpy as np

    ri = mol.GetRingInfo()
    ring_homas = []

    conf = mol.GetConformer()  # 3D coordinates assumed
    for ring_atoms in ri.AtomRings():
        # Check if all atoms are aromatic carbons
        all_aromatic_c = True
        for a_idx in ring_atoms:
            at = mol.GetAtomWithIdx(a_idx)
            if at.GetAtomicNum() != 6 or not at.GetIsAromatic():
                all_aromatic_c = False
                break
        if not all_aromatic_c:
            continue

        # Gather ring bond lengths
        bond_lengths = []
        ring_size = len(ring_atoms)
        for i in range(ring_size):
            a1 = ring_atoms[i]
            a2 = ring_atoms[(i+1) % ring_size]
            p1 = np.array(conf.GetAtomPosition(a1))
            p2 = np.array(conf.GetAtomPosition(a2))
            dist = np.linalg.norm(p1 - p2)
            bond_lengths.append(dist)

        # If ring_size > 2 ...
        if bond_lengths:
            # HOMA formula
            n = len(bond_lengths)
            sum_sq = 0.0
            for R in bond_lengths:
                diff = (R - R_opt)
                sum_sq += diff * diff
            homa_val = 1.0 - (alpha / n) * sum_sq
            ring_homas.append(homa_val)

    if not ring_homas:
        return (float('nan'), [])
    avg_homa = sum(ring_homas)/len(ring_homas)
    return (avg_homa, ring_homas)


def homa_aromatic_rings(mol, alpha=257.7, R_opt=1.388):
    """
    Compute:
      - average HOMA over all purely C–C aromatic rings,
      - list of HOMA per ring,
      - HOMA over all 'fused' bonds,
      - HOMA over all 'edge' bonds.

    Returns:
        avg_homa: float
        ring_homas: list of floats
        homa_fused: float
        homa_edge: float
    """
    from rdkit import Chem
    import numpy as np

    ri = mol.GetRingInfo()
    conf = mol.GetConformer()

    # 1) select only the purely C–C aromatic rings
    aromatic_rings = []
    for ring in ri.AtomRings():
        if all(mol.GetAtomWithIdx(i).GetAtomicNum() == 6 and
               mol.GetAtomWithIdx(i).GetIsAromatic()
               for i in ring):
            aromatic_rings.append(ring)

    # 2) compute per-ring HOMA exactly as before
    ring_homas = []
    for ring in aromatic_rings:
        # get successive bond lengths around the ring
        lengths = []
        n = len(ring)
        for i in range(n):
            i1, i2 = ring[i], ring[(i+1) % n]
            p1 = np.array(conf.GetAtomPosition(i1))
            p2 = np.array(conf.GetAtomPosition(i2))
            lengths.append(np.linalg.norm(p1-p2))
        # HOMA formula
        sq = sum((L - R_opt)**2 for L in lengths)
        ring_homas.append(1.0 - (alpha/n)*sq)

    # 3) classify bonds into 'fused' vs 'edge'
    #    we use the matching BondRings() entries
    bond_rings = ri.BondRings()
    # pick only those BondRings corresponding to our aromatic_rings
    # (AtomRings and BondRings are in parallel order)
    aromatic_bond_rings = [
        bond_rings[i]
        for i, ring in enumerate(ri.AtomRings())
        if ring in aromatic_rings
    ]

    # count occurrences of each bond index
    counts = {}
    for br in aromatic_bond_rings:
        for bidx in br:
            counts[bidx] = counts.get(bidx, 0) + 1

    fused = [b for b,c in counts.items() if c > 1]
    edge  = [b for b,c in counts.items() if c == 1]

    # 4) helper to compute HOMA on an arbitrary bond-index list
    def homa_on_bonds(bond_list):
        m = len(bond_list)
        if m == 0:
            return float('nan')
        s = 0.0
        for bidx in bond_list:
            bond = mol.GetBondWithIdx(bidx)
            a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            R = np.linalg.norm(
                np.array(conf.GetAtomPosition(a1)) -
                np.array(conf.GetAtomPosition(a2))
            )
            s += (R - R_opt)**2
        return 1.0 - (alpha/m)*s

    homa_fused = homa_on_bonds(fused)
    homa_edge  = homa_on_bonds(edge)

    # 5) average per-ring HOMA
    avg_homa = float(np.nan) if not ring_homas else sum(ring_homas)/len(ring_homas)

    return avg_homa, ring_homas, homa_fused, homa_edge


def radius_of_gyration(mol):
    """
    Compute (unweighted) radius of gyration from the 3D coordinates.
    If you want mass-weighted, adapt with atomic masses.

    :return: float
    """
    import numpy as np
    conf = mol.GetConformer()
    coords = [conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())]
    if not coords:
        return float('nan')

    coords_arr = np.array([[p.x, p.y, p.z] for p in coords])
    centroid = coords_arr.mean(axis=0)
    diffs = coords_arr - centroid
    sq = np.sum(diffs*diffs, axis=1)
    rg = np.sqrt(np.mean(sq))
    return rg

def radius_of_gyration_mass_weighted(mol):
    import numpy as np
    from rdkit.Chem.rdchem import GetPeriodicTable

    conf = mol.GetConformer()
    coords = [conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())]
    if not coords:
        return float('nan')

    coords_arr = np.array([[p.x, p.y, p.z] for p in coords])
    ptable = GetPeriodicTable()

    masses = []
    for atom in mol.GetAtoms():
        mass = ptable.GetAtomicWeight(atom.GetAtomicNum())
        masses.append(mass)
    masses = np.array(masses)
    total_m = np.sum(masses)

    # mass-weighted centroid
    centroid_m = np.sum(coords_arr.T * masses, axis=1)/total_m  # shape (3,)

    # sum( m_i * |r_i - r_cm|^2 ) / total_m
    diffs = coords_arr - centroid_m
    sq_weight = np.sum(masses * np.sum(diffs*diffs, axis=1))
    rg = np.sqrt(sq_weight / total_m)
    return rg

from rdkit.Chem import rdFreeSASA
#from rdkit.Chem.rdFreeSASA import classifyAtoms, FreeSASAAtomType
# if your build supports it, you could do:
#atypes = classifyAtoms(mol, FreeSASAAtomType.WHATEVER)
# then build radii from the classification


def molecular_surface_area(mol):
    """
    Compute approximate solvent-accessible surface area (SASA)
    using RDKit's rdFreeSASA.
    Return in A^2.

    If you want a 'van der Waals surface area', you can
    define a radii set differently.
    """
    # You need atomic radii. RDKit provides a default from FreeSASA class
    # Alternatively define your own, e.g. a dictionary {atomicNum: radius}
    # to mimic a "vdW" or "probe" radius.

    # Must have 3D coords assigned
    if not mol.GetConformer().Is3D():
        return float('nan')
    # create a list of radii
    #radii = rdFreeSASA.ClassicRadii()  # default set
    atomic_radii = {
        1: 1.20,  # H
        6: 1.70,  # C
        7: 1.55,  # N
        8: 1.52,  # O
        9: 1.47,  # F
        16: 1.80, # S
        17: 1.75, # Cl
        # etc. or fallback for unknown
    }

    radii = []
    for atom in mol.GetAtoms():
        z = atom.GetAtomicNum()
        if z in atomic_radii:
            radii.append(atomic_radii[z])
        else:
            # fallback
            radii.append(1.50)
    # compute
    sasa = rdFreeSASA.CalcSASA(mol, radii)
    return sasa

def inertia_tensor(mol):
    """
    Returns the 3x3 mass-weighted inertia tensor in principal axes (or just the raw inertia matrix).
    We do mass-weighted. Return principal moments + principal axes, for example.

    :return: (moments, axes)
       moments = np.array([I1, I2, I3]) sorted ascending
       axes    = 3x3 matrix (the principal axes as row vectors, e.g.)
    """
    import numpy as np
    from rdkit.Chem.rdchem import GetPeriodicTable

    conf = mol.GetConformer()
    if not conf.Is3D():
        return None, None

    coords = []
    ptable = GetPeriodicTable()
    masses = []
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        pos = conf.GetAtomPosition(idx)
        coords.append([pos.x, pos.y, pos.z])
        masses.append(ptable.GetAtomicWeight(atom.GetAtomicNum()))
    coords = np.array(coords)
    masses = np.array(masses)
    M = np.sum(masses)

    # mass-weighted centroid
    centroid = np.sum(coords.T * masses, axis=1)/M

    # shift coords to centroid
    coords -= centroid

    # build inertia tensor
    # Ixx = sum(m_i(y^2 + z^2)), Ixy= - sum(m_i x_i y_i), ...
    # We can do it with a direct summation:
    I = np.zeros((3,3), dtype=float)
    for i,(x,y,z) in enumerate(coords):
        m = masses[i]
        I[0,0] += m*(y*y + z*z)
        I[1,1] += m*(x*x + z*z)
        I[2,2] += m*(x*x + y*y)
        I[0,1] -= m*x*y
        I[0,2] -= m*x*z
        I[1,2] -= m*y*z
    # fill symmetric
    I[1,0] = I[0,1]
    I[2,0] = I[0,2]
    I[2,1] = I[1,2]

    # diagonalize
    vals, vecs = np.linalg.eigh(I)
    # vals sorted ascending
    # vecs columns are eigenvectors
    # if we prefer row vectors as principal axes:
    vecs = vecs.T

    return vals, vecs

def wiener_3d(mol):
    """
    3D analog of Wiener index:
      sum of Euclidean distances between all pairs of atoms.

    :return: float
    """
    import numpy as np
    conf = mol.GetConformer()
    coords = [conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())]
    n = len(coords)
    coords_arr = np.array([[p.x, p.y, p.z] for p in coords])

    w3d = 0.0
    for i in range(n):
        for j in range(i+1, n):
            diff = coords_arr[j] - coords_arr[i]
            dist = np.linalg.norm(diff)
            w3d += dist
    return w3d

def count_bay_regions(mol, angle_threshold=120.0):
    """
    A naive 'bay region' count for ring-fused PAHs:
      - Find ring pairs that share exactly two adjacent atoms (fused).
      - Check the angle between the ring planes or the angle at the shared edge.
      - If it's sufficiently 'concave', count as a 'bay region'.

    This is a simplistic approach; real bay region detection can be more involved.
    :param angle_threshold: e.g. 120 deg for “concavity” test
    :return: integer count of bay-like ring fusions
    """
    import numpy as np
    from rdkit import Chem

    ri = mol.GetRingInfo()
    ring_atoms_list = ri.AtomRings()
    n_rings = len(ring_atoms_list)
    conf = mol.GetConformer()
    if n_rings < 2:
        return 0

    def ring_normal(atoms):
        # naive plane normal from 1st 3 coords:
        # or do an SVD approach for all ring coords
        pts = np.array([conf.GetAtomPosition(a) for a in atoms])
        center = pts.mean(axis=0)
        pts_centered = pts - center
        # SVD
        u,s,vh = np.linalg.svd(pts_centered, full_matrices=False)
        normal = vh[-1,:]  # last row
        return normal / np.linalg.norm(normal)

    bay_count = 0

    for i in range(n_rings):
        ring_i = ring_atoms_list[i]
        set_i = set(ring_i)
        normal_i = ring_normal(ring_i)
        for j in range(i+1, n_rings):
            ring_j = ring_atoms_list[j]
            set_j = set(ring_j)
            shared = set_i.intersection(set_j)
            if len(shared)==2:
                # see if they are adjacent in the ring sense
                # check the ring_i’s adjacency to see if those 2 atoms are next to each other
                # or we do a simpler check: measure angle between ring normals
                normal_j = ring_normal(ring_j)
                dot_ = np.dot(normal_i, normal_j)
                # clamp
                dot_ = max(min(dot_,1.0),-1.0)
                angle = np.degrees(np.arccos(dot_))
                # if angle is bigger than ~ (180 - angle_threshold?), we might have a bay?
                # or do geometry at the shared edge. We'll do a naive approach:
                # say if ring planes are fairly "co-planar," then there's a potential bay
                if abs(angle - 180) < angle_threshold:
                    bay_count += 1

    return bay_count


##############################################################################
# MAIN PROCESSING:  process_file -> write_results
##############################################################################

def process_file(file):
    global directory, xtb_raw_values, D4_rel_values

    log_file = os.path.join(directory, file)
    xyz_file = log_file.replace('.log', '.xyz')

    # Convert if .xyz not found
    if not os.path.exists(xyz_file):
        convert_log_to_xyz(directory)
        if not os.path.exists(xyz_file):
            print(f"Failed to generate XYZ for {log_file}, skipping.")
            return None

    mol = Chem.MolFromXYZFile(xyz_file)
    if mol is None:
        print(f"Failed RDKit read of {xyz_file}, skipping.")
        return None

    Chem.rdmolops.AssignStereochemistry(mol, cleanIt=True, force=True)
    rdDetermineBonds.DetermineBonds(mol)

    # Energies
    (energy,
     energy_kcal_mol,
     homo_energy,
     lumo_energy,
     band_gap_eV,
     total_dipole,
     electronic_spatial_extent,
     max_mulliken_charge,
     min_mulliken_charge,
     degrees_of_freedom,
     _quadrupole_moment,
     _traceless_quadrupole_moment,
     _octapole_moment,
     _hexadecapole_moment) = extract_energy(log_file)

    # xTB raw
    xtb_raw_energy = xtb_raw_values.get(file, float('nan'))

    # D4 relative energy
    D4_rel_energy = D4_rel_values.get(file, float('nan'))

    # Angles & distances
    bonded_carbons = find_bonded_carbons(mol)
    (dihedrals,
     bond_angles,
     sum_less_90,
     sum_greater_90,
     count_less_90,
     count_greater_90,
     sum_abs_120_minus_angle,
     count_angles,
     rmsd_bond_angle) = find_dihedrals_and_bond_angles(mol, bonded_carbons)

    (total_hydrogen_distance,
     total_H_distance2,
     total_countH_under5) = find_hydrogen_distances(mol)

    mean_CC_distance, rmsd_bond_lengths = calculate_rmsd_bond_lengths(mol, bonded_carbons)

    # BLA / BAA
    aromatic_rings = detect_aromatic_rings(mol)
    blas, baas = calculate_bla_and_baa(mol, aromatic_rings)
    mean_bla = np.mean(blas) if blas else float('nan')
    mean_baa = np.mean(baas) if baas else float('nan')

    # DPO
    total_dpo, longest_linear_path = calculate_dpo(mol)

    # Z displacement
    (max_z_disp, mean_z, rmsd_z, mad_z) = calculate_max_z_displacement(mol)

    # 2D area
    area = calculate_projected_area(mol)

    # Max C–C distance in XY
    max_cc_distance = calculate_max_cc_distance(mol)

    # Asymmetry
    asymmetry = calculate_asymmetry(mol)

    # Possibly compute pyramidalization angles (optional)
    # (example: mean of all possible 3-neighbor atoms)
    pyramids = []
    conf = mol.GetConformer()
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        neighbors = [conf.GetAtomPosition(nb.GetIdx()) for nb in atom.GetNeighbors()]
        if len(neighbors) == 3:
            center = conf.GetAtomPosition(idx)
            pyramids.append(calculate_pyramidalization_angle(np.array(center), [np.array(x) for x in neighbors]))
    mean_pyramidalization = np.mean(pyramids) if pyramids else float('nan')
    rmsd_pyramidalization = np.sqrt(np.mean(np.square(pyramids))) if pyramids else float('nan')

    # New: topological descriptors (Wiener, Randic)
    wiener_val    = calculate_wiener_index(mol)
    randic_val    = calculate_randic_index(mol)
    harary_val    = harary_index(mol)
    hyper_wiener_val = hyper_wiener_index(mol)
    abc_val       = abc_index(mol)
    ecc_val       = eccentric_connectivity_index(mol)
    balaban_val   = balaban_index(mol)
    zagreb1_val   = first_zagreb_index(mol)
    zagreb2_val   = second_zagreb_index(mol)
    cluj_val      = cluj_index(mol)
    (avg_homa_val, ring_homas_val, avg_homa_fused_val, avg_homa_edge_val) = homa_aromatic_rings(mol)
    (avg_homa2_val, ring_homas2, homa_fused2, homa_edge2) = homa_aromatic_rings(mol, alpha=257.7, R_opt=1.397)
    (avg_homa3_val, ring_homas3, homa_fused3, homa_edge3) = homa_aromatic_rings(mol, alpha=257.7, R_opt=1.3914)
    (avg_homa4_val, ring_homas4, homa_fused4, homa_edge4) = homa_aromatic_rings(mol, alpha=257.7, R_opt=1.38)
    (avg_homa5_val, ring_homas5, homa_fused5, homa_edge5) = homa_aromatic_rings(mol, alpha=257.7, R_opt=1.37)
    (avg_homa6_val, ring_homas6, homa_fused6, homa_edge6) = homa_aromatic_rings(mol, alpha=257.7, R_opt=1.36)
    (avg_homa7_val, ring_homas7, homa_fused7, homa_edge7) = homa_aromatic_rings(mol, alpha=257.7, R_opt=1.35)
    (avg_homa8_val, ring_homas8, homa_fused8, homa_edge8) = homa_aromatic_rings(mol, alpha=257.7, R_opt=1.34)
    (avg_homa9_val, ring_homas9, homa_fused9, homa_edge9) = homa_aromatic_rings(mol, alpha=257.7, R_opt=1.33)
    (avg_homa10_val, ring_homas10, homa_fused10, homa_edge10) = homa_aromatic_rings(mol, alpha=257.7, R_opt=1.32)
    (avg_homa11_val, ring_homas11, homa_fused11, homa_edge11) = homa_aromatic_rings(mol, alpha=257.7, R_opt=1.31)
    (avg_homa12_val, ring_homas12, homa_fused12, homa_edge12) = homa_aromatic_rings(mol, alpha=257.7, R_opt=1.30)
    (avg_homa13_val, ring_homas13, homa_fused13, homa_edge13) = homa_aromatic_rings(mol, alpha=257.7, R_opt=1.00)
    (avg_homa14_val, ring_homas14, homa_fused14, homa_edge14) = homa_aromatic_rings(mol, alpha=257.7, R_opt=0)
    (avg_homa15_val, ring_homas15, homa_fused15, homa_edge15) = homa_aromatic_rings(mol, alpha=257.7, R_opt=1.41)
    (avg_homa16_val, ring_homas16, homa_fused16, homa_edge16) = homa_aromatic_rings(mol, alpha=257.7, R_opt=1.42)
    (avg_homa17_val, ring_homas17, homa_fused17, homa_edge17) = homa_aromatic_rings(mol, alpha=257.7, R_opt=1.43)
    (avg_homa18_val, ring_homas18, homa_fused18, homa_edge18) = homa_aromatic_rings(mol, alpha=257.7, R_opt=1.44)
    (avg_homa19_val, ring_homas19, homa_fused19, homa_edge19) = homa_aromatic_rings(mol, alpha=257.7, R_opt=1.45)
    (avg_homa20_val, ring_homas20, homa_fused20, homa_edge20) = homa_aromatic_rings(mol, alpha=257.7, R_opt=1.5)
    (avg_homa21_val, ring_homas21, homa_fused21, homa_edge21) = homa_aromatic_rings(mol, alpha=257.7, R_opt=2)
    (avg_homa22_val, ring_homas22, homa_fused22, homa_edge22) = homa_aromatic_rings(mol, alpha=257.7, R_opt=3)
    (avg_homa23_val, ring_homas23, homa_fused23, homa_edge23) = homa_aromatic_rings(mol, alpha=257.7, R_opt=1.46)
    (avg_homa24_val, ring_homas24, homa_fused24, homa_edge24) = homa_aromatic_rings(mol, alpha=257.7, R_opt=1.47)
    (avg_homa25_val, ring_homas25, homa_fused25, homa_edge25) = homa_aromatic_rings(mol, alpha=257.7, R_opt=1.48)
    (avg_homa26_val, ring_homas26, homa_fused26, homa_edge26) = homa_aromatic_rings(mol, alpha=257.7, R_opt=1.49)

    radius_gyr_val = radius_of_gyration(mol)
    surf_val       = molecular_surface_area(mol)
    Ivals, Iaxes   = inertia_tensor(mol)
    if Ivals is not None:
        I_princ_1_val, I_princ_2_val, I_princ_3_val = Ivals
    else:
        I_princ_1_val = I_princ_2_val = I_princ_3_val = float('nan')
    w3d_val        = wiener_3d(mol)
    bay_val        = count_bay_regions(mol)

    return (file,
            energy_kcal_mol,
            D4_rel_energy,
            homo_energy,
            lumo_energy,
            band_gap_eV,
            xtb_raw_energy,
            sum_less_90,
            count_less_90,
            sum_greater_90,
            count_greater_90,
            sum_abs_120_minus_angle,
            rmsd_bond_angle,
            H_total,               # total_hydrogen_distance
            mean_CC_distance,
            rmsd_bond_lengths,
            mean_bla,
            mean_baa,
            max_z_disp,
            mean_z,
            rmsd_z,
            mad_z,
            mean_pyramidalization,
            rmsd_pyramidalization,
            total_dpo,
            total_dipole,
            electronic_spatial_extent,
            max_mulliken_charge,
            min_mulliken_charge,
            degrees_of_freedom,
            area,
            max_cc_distance,
            asymmetry,
            longest_linear_path,
            wiener_val,
            randic_val,
            harary_val,
            hyper_wiener_val,
            abc_val,
            ecc_val,
            balaban_val,
            zagreb1_val,
            zagreb2_val,
            cluj_val,
            avg_homa_val,
            ring_homas_val,
            avg_homa_fused_val,
            avg_homa_edge_val,
            avg_homa2_val,
            avg_homa3_val,
            avg_homa4_val,
            avg_homa5_val,
            avg_homa6_val,
            avg_homa7_val,
            avg_homa8_val,
            avg_homa9_val,
            avg_homa10_val,
            avg_homa11_val,
            avg_homa12_val,
            avg_homa13_val,
            avg_homa14_val,
            avg_homa15_val,
            avg_homa16_val,
            avg_homa17_val,
            avg_homa18_val,
            avg_homa19_val,
            avg_homa20_val,
            avg_homa21_val,
            avg_homa22_val,
            avg_homa23_val,
            avg_homa24_val,
            avg_homa25_val,
            avg_homa26_val,
            radius_gyr_val,
            surf_val,
            I_princ_1_val,
            I_princ_2_val,
            I_princ_3_val,
            w3d_val,
            bay_val
           )

def write_results(results, directory, min_energy_kcal_mol, min_xTB_energy_kcal_mol):
    """
    Append results to CSV file, skipping if already present (or you can overwrite).
    """
    output_file = f'analysis_results.{directory}.csv'
    new_file = not os.path.exists(output_file) or (os.path.getsize(output_file) == 0)

    with open(output_file, 'a') as f:
        if new_file:
            # Add wiener_index, randic_index in the CSV header
            f.write('file,energy_kcal_mol,rel_energy_kcal_mol,D4_rel_energy,homo_energy,lumo_energy,band_gap_eV,'
                    'xtb_raw_energy,xtb_rel_energy_kcal_mol,sum_less_90,count_less_90,sum_greater_90,count_greater_90,'
                    'sum_abs_120_minus_angle,rmsd_bond_angle,total_hydrogen_distance,mean_CC_distance,rmsd_bond_lengths,mean_bla,'
                    'mean_baa,max_z_displacement,mean_z,rmsd_z,mad_z,mean_pyramidalization,rmsd_pyramidalization,'
                    'total_dpo,total_dipole,electronic_spatial_extent,max_mulliken_charge,min_mulliken_charge,'
                    'degrees_of_freedom,area,max_cc_distance,asymmetry,longest_linear_path,wiener_val,randic_val,'
                    'harary_val,hyper_wiener_val,abc_val,eccentric_connectivity_val,balaban_val,zagreb1_val,zagreb2_val,'
                    'cluj_val,avg_homa_val,ring_homas_val,avg_homa_fused_val,avg_homa_edge_val,avg_homa2_val,avg_homa3_val,avg_homa4_val,avg_homa5_val,avg_homa6_val,'
                    'avg_homa7_val,avg_homa8_val,avg_homa9_val,avg_homa10_val,avg_homa11_val,'
                    'avg_homa12_val,avg_homa13_val,avg_homa14_val,avg_homa15_val,avg_homa16_val,avg_homa17_val,'
                    'avg_homa18_val,avg_homa19_val,avg_homa20_val,avg_homa21_val,avg_homa22_val,'
                    'avg_homa23_val,avg_homa24_val,avg_homa25_val,avg_homa26_val,radius_gyr_val,'
                    'surf_area_val,I_princ_1_val,I_princ_2_val,I_princ_3_val,w3d_val,bay_val\n')

        for r in results:
            if r is None:
                continue
            (file,
             energy_kcal_mol,
             D4_rel_energy,
             homo_energy,
             lumo_energy,
             band_gap_eV,
             xtb_raw_energy,
             sum_less_90,
             count_less_90,
             sum_greater_90,
             count_greater_90,
             sum_abs_120_minus_angle,
             rmsd_bond_angle,
             total_hydrogen_distance,
             mean_CC_distance,
             rmsd_bond_lengths,
             mean_bla,
             mean_baa,
             max_z_disp,
             mean_z,
             rmsd_z,
             mad_z,
             mean_pyramidalization,
             rmsd_pyramidalization,
             total_dpo,
             total_dipole,
             electronic_spatial_extent,
             max_mulliken_charge,
             min_mulliken_charge,
             degrees_of_freedom,
             area,
             max_cc_distance,
             asymmetry,
             longest_linear_path,
             wiener_val,
             randic_val,
             harary_val,
             hyper_wiener_val,
             abc_val,
             ecc_val,
             balaban_val,
             zagreb1_val,
             zagreb2_val,
             cluj_val,
             avg_homa_val,
             ring_homas_val,
             avg_homa_fused_val,
             avg_homa_edge_val,
             avg_homa2_val,
             avg_homa3_val,
             avg_homa4_val,
             avg_homa5_val,
             avg_homa6_val,
             avg_homa7_val,
             avg_homa8_val,
             avg_homa9_val,
             avg_homa10_val,
             avg_homa11_val,
             avg_homa12_val,
             avg_homa13_val,
             avg_homa14_val,
             avg_homa15_val,
             avg_homa16_val,
             avg_homa17_val,
             avg_homa18_val,
             avg_homa19_val,
             avg_homa20_val,
             avg_homa21_val,
             avg_homa22_val,
             avg_homa23_val,
             avg_homa24_val,
             avg_homa25_val,
             avg_homa26_val,
             radius_gyr_val,
             surf_val,
             I_princ_1_val,
             I_princ_2_val,
             I_princ_3_val,
             w3d_val,
             bay_val) = r

            # If energies are valid, define relative energies:
            if energy_kcal_mol is not None:
                rel_energy_kcal_mol = energy_kcal_mol - min_energy_kcal_mol
            else:
                rel_energy_kcal_mol = float('nan')

            if (not np.isnan(xtb_raw_energy)):
                xtb_rel_energy_kcal_mol = (xtb_raw_energy * 627.5095) - min_xTB_energy_kcal_mol
            else:
                xtb_rel_energy_kcal_mol = float('nan')

            # Write line
            f.write(f"{file},{energy_kcal_mol},{rel_energy_kcal_mol},{D4_rel_energy},{homo_energy},{lumo_energy},"
                    f"{band_gap_eV},{xtb_raw_energy},{xtb_rel_energy_kcal_mol},{sum_less_90:.2f},{count_less_90},"
                    f"{sum_greater_90:.2f},{count_greater_90},{sum_abs_120_minus_angle:.2f},{rmsd_bond_angle}"
                    f"{total_hydrogen_distance:.2f},{mean_CC_distance:.5f},{rmsd_bond_lengths:.5f},"
                    f"{mean_bla:.5f},{mean_baa:.5f},{max_z_disp:.5f},{mean_z:.5f},{rmsd_z:.5f},{mad_z:.5f},"
                    f"{mean_pyramidalization:.5f},{rmsd_pyramidalization:.5f},{total_dpo},{total_dipole},"
                    f"{electronic_spatial_extent},{max_mulliken_charge},{min_mulliken_charge},"
                    f"{degrees_of_freedom},{area},{max_cc_distance:.5f},{asymmetry:.5f},"
                    f"{longest_linear_path},{wiener_val},{randic_val},{harary_val},{hyper_wiener_val},"
                    f"{abc_val},{ecc_val},{balaban_val},{zagreb1_val},{zagreb2_val},{cluj_val},"
                    f"{avg_homa_val},{ring_homas_val},{avg_homa_fused_val},{avg_homa_edge_val}"
                    f"{avg_homa2_val},{avg_homa3_val},{avg_homa4_val},{avg_homa5_val},{avg_homa6_val},"
                    f"{avg_homa7_val},{avg_homa8_val},{avg_homa9_val},{avg_homa10_val},"
                    f"{avg_homa11_val},{avg_homa12_val},{avg_homa13_val},{avg_homa14_val},"
                    f"{avg_homa15_val},{avg_homa16_val},{avg_homa17_val},{avg_homa18_val},"
                    f"{avg_homa19_val},{avg_homa20_val},{avg_homa21_val},{avg_homa22_val},"
                    f"{avg_homa23_val},{avg_homa24_val},{avg_homa25_val},{avg_homa26_val},"
                    f"{radius_gyr_val},{surf_val},{I_princ_1_val},{I_princ_2_val},{I_princ_3_val},{w3d_val},{bay_val}\n")

def load_existing_results(directory):
    """
    Return the set of file-names already found in the existing CSV 
    (so we skip them).
    """
    output_file = f'analysis_results.{directory}.csv'
    existing_files = set()
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            lines = f.readlines()
            # skip header
            for line in lines[1:]:
                if not line.strip():
                    continue
                items = line.split(',')
                # first item = file
                existing_files.add(items[0].strip())
    return existing_files


##############################################################################
# SINGLE-FILE "FULL" COMPUTE
##############################################################################
def compute_all_columns_for_file(log_file, directory, D4_rel_values, xtb_raw_values):
    """
    If a file is brand new or we want to forcibly recalc everything, 
    do all computations and return a dict of column->value.
    """
    row_data = dict.fromkeys(ALL_COLUMNS, np.nan)
    row_data['file'] = log_file

    # 1) ensure xyz
    log_fullpath = os.path.join(directory, log_file)
    xyz_file = log_fullpath.replace('.log','.xyz')
    if not os.path.exists(xyz_file):
        convert_log_to_xyz_single(log_fullpath, xyz_file)
    if not os.path.exists(xyz_file):
        # fail
        return row_data

    # load RDKit
    mol = Chem.MolFromXYZFile(xyz_file)
    if mol is None:
        return row_data
    rdDetermineBonds.DetermineBonds(mol)

    # 2) energies
    (energy,
     energy_kcal_mol,
     homo_energy,
     lumo_energy,
     band_gap_eV,
     total_dipole,
     electronic_spatial_extent,
     max_mulliken_charge,
     min_mulliken_charge,
     degrees_of_freedom,
     _qm, _tqm, _oct, _hex) = extract_energy(log_fullpath)

    row_data['energy_kcal_mol'] = energy_kcal_mol
    row_data['homo_energy']     = homo_energy
    row_data['lumo_energy']     = lumo_energy
    row_data['band_gap_eV']     = band_gap_eV
    row_data['total_dipole']    = total_dipole
    row_data['electronic_spatial_extent'] = electronic_spatial_extent
    row_data['max_mulliken_charge'] = max_mulliken_charge
    row_data['min_mulliken_charge'] = min_mulliken_charge
    row_data['degrees_of_freedom'] = degrees_of_freedom

    # 3) D4 / xTB
    row_data['D4_rel_energy'] = D4_rel_values.get(log_file, np.nan)
    raw_xtb = xtb_raw_values.get(log_file, np.nan)
    row_data['xtb_raw_energy'] = raw_xtb

    # 4) geometry measures
    bc = find_bonded_carbons(mol)
    (dihedrals, bond_angles, sum_less_90, sum_greater_90, count_less_90,
     count_greater_90, sum_abs_120_minus_angle, count_angles, rmsd_bond_angle) = find_dihedrals_and_bond_angles(mol, bc)
    row_data['sum_less_90'] = sum_less_90
    row_data['sum_greater_90'] = sum_greater_90
    row_data['count_less_90'] = count_less_90
    row_data['count_greater_90'] = count_greater_90
    row_data['sum_abs_120_minus_angle'] = sum_abs_120_minus_angle
    row_data['rmsd_bond_angle'] = rmsd_bond_angle

    (totalH, totalH2, countHunder5) = find_hydrogen_distances(mol)
    row_data['total_hydrogen_distance'] = totalH

    meanCC, rmsdCC = calculate_rmsd_bond_lengths(mol, bc)
    row_data['mean_CC_distance'] = meanCC
    row_data['rmsd_bond_lengths'] = rmsdCC

    # BLA/BAA
    arings = detect_aromatic_rings(mol)
    blas, baas = calculate_bla_and_baa(mol, arings)
    row_data['mean_bla'] = np.mean(blas) if blas else np.nan
    row_data['mean_baa'] = np.mean(baas) if baas else np.nan

    # DPO
    dpo_val, longpath = calculate_dpo(mol)
    row_data['total_dpo'] = dpo_val
    row_data['longest_linear_path'] = longpath

    # Z disp
    (mzdisp, meanz, rmsdz, madz) = calculate_max_z_displacement(mol)
    row_data['max_z_displacement'] = mzdisp
    row_data['mean_z'] = meanz
    row_data['rmsd_z'] = rmsdz
    row_data['mad_z'] = madz

    # area
    area_ = calculate_projected_area(mol)
    row_data['area'] = area_

    # max CC distance
    row_data['max_cc_distance'] = calculate_max_cc_distance(mol)

    # asymmetry
    row_data['asymmetry'] = calculate_asymmetry(mol)

    # pyramidalization angles
    pyramids = []
    conf = mol.GetConformer()
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        neighbors = [conf.GetAtomPosition(nb.GetIdx()) for nb in atom.GetNeighbors()]
        if len(neighbors)==3:
            ctr = conf.GetAtomPosition(idx)
            val_ = calculate_pyramidalization_angle(np.array(ctr), [np.array(x) for x in neighbors])
            pyramids.append(val_)
    if pyramids:
        row_data['mean_pyramidalization'] = np.mean(pyramids)
        row_data['rmsd_pyramidalization'] = np.sqrt(np.mean(np.square(pyramids)))
    else:
        row_data['mean_pyramidalization'] = np.nan
        row_data['rmsd_pyramidalization'] = np.nan

    row_data['wiener_val'] = calculate_wiener_index(mol)
    row_data['randic_val'] = calculate_randic_index(mol)
    row_data['harary_val'] = harary_index(mol)
    row_data['hyper_wiener_val'] = hyper_wiener_index(mol)
    row_data['abc_val'] = abc_index(mol)
    row_data['ecc_val'] = eccentric_connectivity_index(mol)
    row_data['balaban_val'] = balaban_index(mol)
    row_data['zagreb1_val'] = first_zagreb_index(mol)
    row_data['zagreb2_val'] = second_zagreb_index(mol)
    row_data['cluj_val'] = cluj_index(mol)
    avg_homa_val, ring_homas_val,  avg_homa_fused_val, avg_homa_edge_val = homa_aromatic_rings(mol)
    row_data['avg_homa_val'] = avg_homa_val
    row_data['ring_homas_val'] = ring_homas_val
    row_data['avg_homa_fused_val'] = avg_homa_fused_val
    row_data['avg_homa_edge_val'] = avg_homa_edge_val
    avg_homa2_val, ring_homas2,  homa_fused2, homa_edge2 = homa_aromatic_rings(mol, alpha=257.7, R_opt=1.397)
    row_data['avg_homa2_val'] = avg_homa2_val
    avg_homa3_val, ring_homas3,  homa_fused3, homa_edge3 = homa_aromatic_rings(mol, alpha=257.7, R_opt=1.3914)
    row_data['avg_homa3_val'] = avg_homa3_val
    avg_homa4_val, ring_homas4,  homa_fused4, homa_edge4 = homa_aromatic_rings(mol, alpha=257.7, R_opt=1.38)
    row_data['avg_homa4_val'] = avg_homa4_val
    avg_homa5_val, ring_homas5,  homa_fused5, homa_edge5 = homa_aromatic_rings(mol, alpha=257.7, R_opt=1.37)
    row_data['avg_homa5_val'] = avg_homa5_val
    avg_homa6_val, ring_homas6,  homa_fused6, homa_edge6 = homa_aromatic_rings(mol, alpha=257.7, R_opt=1.36)
    row_data['avg_homa6_val'] = avg_homa6_val
    avg_homa7_val, ring_homas7,  homa_fused7, homa_edge7 = homa_aromatic_rings(mol, alpha=257.7, R_opt=1.35)
    row_data['avg_homa7_val'] = avg_homa7_val
    avg_homa8_val, ring_homas8,  homa_fused8, homa_edge8 = homa_aromatic_rings(mol, alpha=257.7, R_opt=1.34)
    row_data['avg_homa8_val'] = avg_homa8_val
    avg_homa9_val, ring_homas9,  homa_fused9, homa_edge9 = homa_aromatic_rings(mol, alpha=257.7, R_opt=1.33)
    row_data['avg_homa9_val'] = avg_homa9_val
    avg_homa10_val, ring_homas10,  homa_fused10, homa_edge10 = homa_aromatic_rings(mol, alpha=257.7, R_opt=1.32)
    row_data['avg_homa10_val'] = avg_homa10_val
    avg_homa11_val, ring_homas11,  homa_fused11, homa_edge11 = homa_aromatic_rings(mol, alpha=257.7, R_opt=1.31)
    row_data['avg_homa11_val'] = avg_homa11_val
    avg_homa12_val, ring_homas12,  homa_fused12, homa_edge12 = homa_aromatic_rings(mol, alpha=257.7, R_opt=1.30)
    row_data['avg_homa12_val'] = avg_homa12_val
    avg_homa13_val, ring_homas13,  homa_fused13, homa_edge13 = homa_aromatic_rings(mol, alpha=257.7, R_opt=1.00)
    row_data['avg_homa13_val'] = avg_homa13_val
    avg_homa14_val, ring_homas14,  homa_fused14, homa_edge14 = homa_aromatic_rings(mol, alpha=257.7, R_opt=0)
    row_data['avg_homa14_val'] = avg_homa14_val
    avg_homa15_val, ring_homas15,  homa_fused15, homa_edge15 = homa_aromatic_rings(mol, alpha=257.7, R_opt=1.41)
    row_data['avg_homa15_val'] = avg_homa15_val
    avg_homa16_val, ring_homas16,  homa_fused16, homa_edge16 = homa_aromatic_rings(mol, alpha=257.7, R_opt=1.42)
    row_data['avg_homa16_val'] = avg_homa16_val
    avg_homa17_val, ring_homas17,  homa_fused17, homa_edge17 = homa_aromatic_rings(mol, alpha=257.7, R_opt=1.43)
    row_data['avg_homa17_val'] = avg_homa17_val
    avg_homa18_val, ring_homas18,  homa_fused18, homa_edge18 = homa_aromatic_rings(mol, alpha=257.7, R_opt=1.44)
    row_data['avg_homa18_val'] = avg_homa18_val
    avg_homa19_val, ring_homas19,  homa_fused19, homa_edge19 = homa_aromatic_rings(mol, alpha=257.7, R_opt=1.45)
    row_data['avg_homa19_val'] = avg_homa19_val
    avg_homa20_val, ring_homas20,  homa_fused20, homa_edge20 = homa_aromatic_rings(mol, alpha=257.7, R_opt=1.5)
    row_data['avg_homa20_val'] = avg_homa20_val
    avg_homa21_val, ring_homas21,  homa_fused21, homa_edge21 = homa_aromatic_rings(mol, alpha=257.7, R_opt=2)
    row_data['avg_homa21_val'] = avg_homa21_val
    avg_homa22_val, ring_homas22,  homa_fused22, homa_edge22 = homa_aromatic_rings(mol, alpha=257.7, R_opt=3)
    row_data['avg_homa22_val'] = avg_homa22_val
    avg_homa23_val, ring_homas23,  homa_fused23, homa_edge23 = homa_aromatic_rings(mol, alpha=257.7, R_opt=1.46)
    row_data['avg_homa23_val'] = avg_homa23_val
    avg_homa24_val, ring_homas24,  homa_fused24, homa_edge24 = homa_aromatic_rings(mol, alpha=257.7, R_opt=1.47)
    row_data['avg_homa24_val'] = avg_homa24_val
    avg_homa25_val, ring_homas25,  homa_fused25, homa_edge25 = homa_aromatic_rings(mol, alpha=257.7, R_opt=1.48)
    row_data['avg_homa25_val'] = avg_homa25_val
    avg_homa26_val, ring_homas26,  homa_fused26, homa_edge26 = homa_aromatic_rings(mol, alpha=257.7, R_opt=1.49)
    row_data['avg_homa26_val'] = avg_homa26_val
    row_data['radius_gyr_val'] = radius_of_gyration(mol)
    row_data['surf_val'] = molecular_surface_area(mol)
    Ivals, Iaxes = inertia_tensor(mol)
    row_data['I_princ_1_val'] = Ivals[0]  # if Ivals is not None
    row_data['I_princ_2_val'] = Ivals[1]
    row_data['I_princ_3_val'] = Ivals[2]
    row_data['w3d_val'] = wiener_3d(mol)
    row_data['bay_val'] = count_bay_regions(mol)


#    row_data['wiener_index'] = calculate_wiener_index(mol)
#    row_data['randic_index'] = calculate_randic_index(mol)
#    row_data['harary_index'] = harary_index(mol)
#    row_data['hyper_wiener'] = hyper_wiener_index(mol)
#    row_data['abc'] = abc_index(mol)
#    row_data['eccentric_connectivity'] = eccentric_connectivity_index(mol)
#    row_data['balaban'] = balaban_index(mol)
#    row_data['zagreb1'] = first_zagreb_index(mol)
#    row_data['zagreb2'] = second_zagreb_index(mol)
#    row_data['cluj_index'] = cluj_index(mol)
#    
#    avg_homa, ring_homas = homa_aromatic_rings(mol)
#    row_data['avg_homa'] = avg_homa
#    
#    rg = radius_of_gyration(mol)
#    row_data['radius_gyr'] = rg
#    
#    surf = molecular_surface_area(mol)
#    row_data['surf_area'] = surf
#    
#    Ivals, Iaxes = inertia_tensor(mol)
#    if Ivals is not None:
#        row_data['I_princ_1'] = Ivals[0]
#        row_data['I_princ_2'] = Ivals[1]
#        row_data['I_princ_3'] = Ivals[2]
#    else:
#        row_data['I_princ_1'] = np.nan
#        row_data['I_princ_2'] = np.nan
#        row_data['I_princ_3'] = np.nan
#        
#    w3d_val = wiener_3d(mol)
#    row_data['wiener_3d'] = w3d_val
#       
#    bay_val = count_bay_regions(mol)
#    row_data['bay_regions'] = bay_val
      
    return row_data


##############################################################################
# PARTIAL PROCESS
##############################################################################

def process_file_partially(log_file, df, directory, D4_rel_values, xtb_raw_values):
    """
    If row does not exist => do full compute.
    Else => find which columns are missing/NaN => compute only those columns.
    Return a dict of updated columns for that row.
    """
    # Is this file in df?
    idx = df.index[df['file']==log_file]
    if len(idx)==0:
        # brand new => do full
        full_data = compute_all_columns_for_file(log_file, directory, D4_rel_values, xtb_raw_values)
        return full_data
    else:
        row_i = idx[0]
        # check if any columns are missing or NaN
        missing_cols = []
        for col in ALL_COLUMNS:
            if col not in df.columns:
                missing_cols.append(col)
            else:
                val = df.loc[row_i, col]
                # if it's NaN or None
                if pd.isna(val):
                    missing_cols.append(col)

        if not missing_cols:
            # nothing to do
            return {}

        # We'll do a "one-shot" compute: re-compute everything 
        # then only fill missing columns. 
        # (Because partial re-check might be complicated if each column 
        #  came from a different function.)
        full_data = compute_all_columns_for_file(log_file, directory, D4_rel_values, xtb_raw_values)
        # now just extract the missing ones
        partial_data = {}
        for mc in missing_cols:
            if mc in full_data:
                partial_data[mc] = full_data[mc]
            else:
                partial_data[mc] = np.nan
        return partial_data


def chunker(seq, size):
    """
    Yields consecutive sub-lists (chunks) from 'seq' of length 'size'
    """
    for pos in range(0, len(seq), size):
        yield seq[pos:pos+size]

##############################################################################
# PARALLEL WORKER FUNCTION
##############################################################################

def partial_update_worker(args):
    """
    Worker function that:
     1) extracts the relevant info from 'args'
     2) calls process_file_partially (or a full approach),
     3) returns (filename, updated_data_dict).
    """
    (lf, df_snapshot, directory, D4_rel_values, xtb_raw_values) = args

    # Run partial update or full compute.
    # Note: 'df_snapshot' is the entire DataFrame or just a copy. Large DataFrame
    # might slow things down. You could pass only the row for 'lf' if you prefer.
    updated_data = process_file_partially(lf, df_snapshot, directory, D4_rel_values, xtb_raw_values)
    # Return the file name plus the updated dictionary
    return (lf, updated_data)

##############################################################################
# CHUNK-BASED PARALLEL PARTIAL UPDATE
##############################################################################

def partial_update_csv_parallel(directory, csv_file, D4_rel_values, xtb_raw_values, chunk_size=CHUNK_SIZE):
    """
    Parallel version of 'partial_update_csv':
      - We split the .log files into chunks of 'chunk_size'.
      - For each chunk, we spin up a Pool, distribute tasks.
      - Each worker returns (filename, updated_data).
      - We merge those into 'df' and save to CSV after each chunk.
    """
    # 1) load or create CSV
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
    else:
        df = pd.DataFrame(columns=ALL_COLUMNS)

    # 2) ensure 'file' column
    if 'file' not in df.columns:
        df['file'] = None

    # 3) gather .log
    logs = [f for f in os.listdir(directory) if f.endswith('.log')]

    # unify with existing
    existing_files = df['file'].dropna().unique().tolist()
    all_files = sorted(set(logs).union(set(existing_files)))
    total_files = len(all_files)
    if total_files == 0:
        print("No files to process.")
        return

    print(f"Parallel partial update for {directory}: {total_files} total files. chunk_size={chunk_size}")
    # We'll process in chunks
    from multiprocessing import Pool

    for chunk_index, chunk in enumerate(chunker(all_files, chunk_size), start=1):
        # Prepare arguments for each file in chunk
        worker_args = []
        for lf in chunk:
            # We pass a "snapshot" of df so each worker can do partial update logic
            # (Large DataFrames might be an issue; see notes above.)
            worker_args.append((lf, df.copy(), directory, D4_rel_values, xtb_raw_values))

        print(f"  -> Chunk #{chunk_index}: processing {len(chunk)} files in parallel.")
        with Pool(processes=NUM_PROCESSES) as pool:
            results = pool.map(partial_update_worker, worker_args)

        # 'results' is a list of (lf, updated_data)
        # Merge into df
        for (lf, updated_data) in results:
            if not updated_data:
                continue
            # either new row or partial update
            idx = df.index[df['file'] == lf]
            if len(idx) == 0:
                # new row
                df = df.append(updated_data, ignore_index=True)
            else:
                # partial update
                row_i = idx[0]
                for k, v in updated_data.items():
                    df.loc[row_i, k] = v

        # Save after each chunk (fail-safe in case of crash)
        df.to_csv(csv_file, index=False)
        print(f"  -> Completed chunk #{chunk_index}, CSV updated.\n")

    print(f"Finished parallel partial updates for {directory}.")

##############################################################################
# MAIN
##############################################################################

if __name__ == "__main__":
    start_time = time.time()
    print("Starting time:", start_time)
    
    # Example: multiple directories
    for directory in [r'C36H20', r'C38H20', r'C40H20', r'C40H22', r'C42H22', r'C44H24',r'PAH335']:
    #for directory in [r'COMPAS1_xyz_DFTseries',r'PAH335']:
        print(f"\nProcessing directory in parallel: {directory}")
        csv_file = f'analysis_results.{directory}.csv'

        # load D4, xTB
        lowercase_directory = directory.lower()
        D4_e_list = f'Ener_D4_list_log_{lowercase_directory}.txt'
        D4_rel_values = {}
        if os.path.exists(D4_e_list):
            with open(D4_e_list,'r') as f:
                for line in f:
                    filename, val = line.split()
                    D4_rel_values[filename] = float(val)
        # xTB
        if directory == r'C44H24':
            xtb_e_list = 'list_xls_c44h24_xtb_abs.txt'
        else:
            xtb_e_list = 'list_xtb_abs_ener.txt'
        xtb_raw_values = {}
        if os.path.exists(xtb_e_list):
            with open(xtb_e_list,'r') as f:
                for line in f:
                    filename, val = line.split()
                    xtb_raw_values[filename] = float(val)

        #  Now do parallel partial update (or chunk-based approach)
        partial_update_csv_parallel(directory, csv_file, D4_rel_values, xtb_raw_values, chunk_size=CHUNK_SIZE)

    print(f"All done in {time.time() - start_time:.2f} s.")
