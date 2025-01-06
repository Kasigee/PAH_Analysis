import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import os

import numpy as np  # Needed for dihedral/bond-angle math

def load_molecule_from_xyz(xyz_file):
    """
    Loads an .xyz file into an RDKit Mol object, then determines bonds
    so 'mol.GetBonds()' won't be empty.
    """
    from rdkit import Chem
    from rdkit.Chem import rdDetermineBonds
    with open(xyz_file, 'r') as f:
        xyz_data = f.read()
    mol = Chem.MolFromXYZBlock(xyz_data)
    if mol is None:
        raise ValueError("Could not parse XYZ into RDKit Mol.")
    # Let RDKit guess connectivity from the 3D coordinates
    rdDetermineBonds.DetermineBonds(mol)
    return mol

def calculate_bond_angle(p1, p2, p3):
    v1 = p1 - p2
    v2 = p3 - p2

    v1 /= np.linalg.norm(v1)
    v2 /= np.linalg.norm(v2)

    cos_theta = np.dot(v1, v2)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    bond_angle = np.degrees(theta)
    return bond_angle

def calculate_dihedral(p1, p2, p3, p4):
    b1 = p2 - p1
    b2 = p3 - p2
    b3 = p4 - p3

    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)

    n1 /= np.linalg.norm(n1)
    n2 /= np.linalg.norm(n2)

    cos_theta = np.dot(n1, n2)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    return np.degrees(theta)

def find_bonded_carbons(mol):
    """
    Return a list of (atom1, atom2) for every C–C bond.
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
    sum_abs_120_minus_angle = 0
    count_angles = 0

    # We'll collect dihedrals first, then do <90 or >90 sums
    for a1 in bonded_carbons_dict:
        for a2 in bonded_carbons_dict[a1]:
            for a3 in bonded_carbons_dict[a2]:
                if a3 != a1:
                    # Bond angle
                    p1 = mol.GetConformer().GetAtomPosition(a1)
                    p2 = mol.GetConformer().GetAtomPosition(a2)
                    p3 = mol.GetConformer().GetAtomPosition(a3)
                    angle = calculate_bond_angle(p1, p2, p3)
                    bond_angles.append((a1, a2, a3, angle))
                    sum_abs_120_minus_angle += abs(120 - angle)
                    count_angles += 1

                    # Dihedral
                    for a4 in bonded_carbons_dict[a3]:
                        if a4 != a2 and a4 != a1:
                            dihedral_tuple = (a1, a2, a3, a4)
                            reversed_tuple = (a4, a3, a2, a1)
                            if (dihedral_tuple not in unique_dihedrals and
                                reversed_tuple not in unique_dihedrals):
                                unique_dihedrals.add(dihedral_tuple)
                                p4 = mol.GetConformer().GetAtomPosition(a4)
                                dih_angle = calculate_dihedral(p1, p2, p3, p4)
                                dihedrals.append((a1, a2, a3, a4, dih_angle))

    # Now sum up angles <90 or >90 to get "sum_less_90" and "sum_greater_90"
    sum_less_90, sum_greater_90 = 0.0, 0.0
    for (_, _, _, _, angle_deg) in dihedrals:
        if angle_deg < 90:
            sum_less_90 += angle_deg
        else:
            sum_greater_90 += (180 - angle_deg)

    return (
        dihedrals,
        bond_angles,
        sum_less_90,
        sum_greater_90,
        # not returning the counts < or > 90 for now, we only need sums
        None, None,
        sum_abs_120_minus_angle,
        count_angles,
    )

def analyze_all_dihedrals(mol):
    """
    Analyze the entire molecule's dihedrals, returning:
      - num_dihedrals
      - sum_all_deviations ( = sum_less_90 + sum_greater_90 )
    """
    bonded_carbons = find_bonded_carbons(mol)
    (dihedrals,
     bond_angles,
     sum_less_90,
     sum_greater_90,
     _count_less_90,
     _count_greater_90,
     sum_abs_120_minus_angle,
     count_angles) = find_dihedrals_and_bond_angles(mol, bonded_carbons)

    num_dihedrals = len(dihedrals)
    sum_all_deviations = sum_less_90 + sum_greater_90

    return {
        "num_dihedrals": num_dihedrals,
        "sum_all_deviations": sum_all_deviations
    }

def approximate_isomerization_energy_dihedral(sum_dev):
    """
    Old dihedral-only formula:
      E_iso = 0.01498026*sum_dev + 5.01448849
      MAD = 3.618
    """
    A = 0.01498026
    B = 5.01448849
    MAD = 3.618
    energy = A * sum_dev + B
    return energy, energy - MAD, energy + MAD

def approximate_isomerization_energy_xtb(sum_dev, xtb_value):
    """
    New formula with sum_dev + XTB:
      E_iso = 0.00678795*sum_dev + 1.07126936*XTB + 3.49502511
      MAD = 2.07925630
    """
    A = 0.00678795
    B = 1.07126936
    C = 3.49502511
    MAD = 2.07925630
    energy = A * sum_dev + B * xtb_value + C
    return energy, energy - MAD, energy + MAD

class MyIsomerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Isomerization Energy Tool (Tkinter)")

        # Main frame
        main_frame = ttk.Frame(root, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")

        #
        # ROW 0: XYZ FILE
        #
        ttk.Label(main_frame, text="XYZ File:").grid(row=0, column=0, sticky="w")
        self.xyz_path_var = tk.StringVar()
        self.xyz_entry = ttk.Entry(main_frame, textvariable=self.xyz_path_var, width=40)
        self.xyz_entry.grid(row=0, column=1, padx=5, sticky="ew")
        browse_btn = ttk.Button(main_frame, text="Browse", command=self.browse_xyz)
        browse_btn.grid(row=0, column=2, sticky="w")

        #
        # ROW 1: COMBOBOX TO CHOOSE "dihedral" OR "xtb"
        #
        ttk.Label(main_frame, text="Parameter:").grid(row=1, column=0, sticky="w")
        self.param_choice_var = tk.StringVar(value="dihedral")
        self.param_combo = ttk.Combobox(
            main_frame,
            textvariable=self.param_choice_var,
            values=["dihedral", "xtb"],
            state="readonly",
            width=15
        )
        self.param_combo.grid(row=1, column=1, sticky="w")
        self.param_combo.bind("<<ComboboxSelected>>", self.on_param_changed)

        #
        # ROW 2: XTB ENERGY (only relevant if param=xtb)
        #
        ttk.Label(main_frame, text="XTB Energy:").grid(row=2, column=0, sticky="w")
        self.xtb_var = tk.StringVar(value="0.0")
        self.xtb_entry = ttk.Entry(main_frame, textvariable=self.xtb_var, width=15)
        self.xtb_entry.grid(row=2, column=1, padx=5, sticky="w")

        #
        # ROW 3: COMPUTE BUTTON
        #
        compute_btn = ttk.Button(main_frame, text="Compute", command=self.on_compute)
        compute_btn.grid(row=3, column=0, pady=10)

        #
        # ROW 4: OUTPUT LABEL
        #
        self.output_label = ttk.Label(main_frame, text="", foreground="blue")
        self.output_label.grid(row=4, column=0, columnspan=3, sticky="w", pady=10)

        #
        # Make the window + columns resizable
        #
        self.root.columnconfigure(0, weight=1)          
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)         

        # Initialize XTB state
        self.on_param_changed()

    def browse_xyz(self):
        """
        Let user pick an .xyz file
        """
        filepath = filedialog.askopenfilename(
            title="Select XYZ file",
            filetypes=[("XYZ files", "*.xyz"), ("All files", "*.*")]
        )
        if filepath:
            self.xyz_path_var.set(filepath)

    def on_param_changed(self, event=None):
        """
        Called whenever the combobox changes, or at startup,
        to enable/disable the XTB input row.
        """
        mode = self.param_choice_var.get()
        if mode == "dihedral":
            self.xtb_entry.configure(state="disabled")
        else:
            self.xtb_entry.configure(state="normal")

    def on_compute(self):
        xyz_file = self.xyz_path_var.get().strip()
        if not xyz_file:
            messagebox.showerror("Error", "Please specify an XYZ file.")
            return

        mode = self.param_choice_var.get()
        try:
            # parse XTB even if disabled
            xtb_value = float(self.xtb_var.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid XTB Energy value.")
            return

        try:
            # Load the molecule
            mol = load_molecule_from_xyz(xyz_file)
            # analyze dihedrals
            results = analyze_all_dihedrals(mol)

            sum_dev = results["sum_all_deviations"]
            if mode == "dihedral":
                iso_energy, iso_low, iso_high = approximate_isomerization_energy_dihedral(sum_dev)
            else:
                # "xtb"
                iso_energy, iso_low, iso_high = approximate_isomerization_energy_xtb(sum_dev, xtb_value)

            msg = (
                f"Total dihedrals found: {results['num_dihedrals']}\n"
                f"Sum of <90° and >90° = {sum_dev:.2f}\n\n"
                f"Isomerization energy ~ {iso_energy:.2f} kcal/mol\n"
                f"Range: [{iso_low:.2f}, {iso_high:.2f}]\n"
            )
            self.output_label.config(text=msg)

        except Exception as e:
            messagebox.showerror("Error", f"Failed: {e}")

def main():
    root = tk.Tk()
    app = MyIsomerGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
