#!/usr/bin/env python3
"""
batch_homa.py

For each .xyz in the given folder, compute:
  • avg per‐ring HOMA
  • global HOMA (unique bonds)
  • weighted avg HOMA
  • HOMA over fused bonds
  • HOMA over edge bonds

Writes the results to:
    <folder_name>_homas.csv
"""

import os, sys, csv
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds

# import your functions from test_homa.py
from test_diff_homa_calcs3 import (
    per_ring_homas,
    global_homa_unique,
    weighted_avg_homa,
    classify_fused_edge_bonds,
    compute_homa
)

def process_xyz(path):
    mol = Chem.MolFromXYZFile(path)
    if mol is None:
        raise ValueError("RDKit failed to read " + path)
    rdDetermineBonds.DetermineBonds(mol)

    # per‐ring
    ring_homas, avg_ring = per_ring_homas(mol)

    # global unique‐bond
    homa_global = global_homa_unique(mol)

    # weighted avg
    homa_weighted = weighted_avg_homa(mol)

    # fused vs edge
    fused, edge = classify_fused_edge_bonds(mol)
    homa_fused = compute_homa(mol, fused)
    homa_edge  = compute_homa(mol, edge)

    return {
        'file': os.path.basename(path),
        'avg_ring_homa': avg_ring,
        'global_homa':  homa_global,
        'weighted_homa':homa_weighted,
        'fused_homa':   homa_fused,
        'edge_homa':    homa_edge
    }

def main(folder):
    if not os.path.isdir(folder):
        print(f"Error: '{folder}' is not a directory")
        sys.exit(1)

    xyzs = sorted(f for f in os.listdir(folder) if f.lower().endswith('.xyz'))
    results = []
    for xyz in xyzs:
        path = os.path.join(folder, xyz)
        try:
            row = process_xyz(path)
        except Exception as e:
            print(f"Skipping {xyz}: {e}", file=sys.stderr)
            continue
        results.append(row)

    if not results:
        print("No valid .xyz files found.")
        return

    outname = os.path.basename(os.path.normpath(folder)) + '_homas.csv'
    with open(outname, 'w', newline='') as csvf:
        writer = csv.DictWriter(csvf, fieldnames=[
            'file',
            'avg_ring_homa',
            'global_homa',
            'weighted_homa',
            'fused_homa',
            'edge_homa'
        ])
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(f"Wrote {len(results)} entries to {outname}")

if __name__=='__main__':
    if len(sys.argv)!=2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1])

