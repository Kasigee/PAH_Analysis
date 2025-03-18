# PAH_Analysis
Code for analysing polycyclic aromatic hydrocarbons

miller_PAH_analysis_wDPO.py was developed to run on a HPC across multiple CPUs. In the form here, it works, but has some redundant old functions that haven't been removed (at this stage).

The xyz_to_isomerisationEapprox.py can run a GUI to do the simpler calculations such as dihedrals and approximate the isomerisation energy from the xyz structure (or recognise the molecule from an xyz-smiles conversion function that then matches to the COMPAS database to search for that structure exisiting within the database). NOTE: That in its current form, it seems to perform best for PAHs with the chemical formula fitting the pattern: C16+4nH8+2n.
