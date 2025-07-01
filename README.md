# PAH_Analysis
Code for analysing polycyclic aromatic hydrocarbons (PAHs)

miller_PAH_analysis_wDPO.py and miller_PAH_analysis_wDPO4b.py was developed to run on a HPC across multiple CPUs. In the form here, it works, but has some redundant old functions that haven't been removed (at this stage).

MADetc_analysis8_R3.py runs the analysis just on the HOMA parameters calculated by test_diff_homa_calcs4.py, however, only small edits required to investigate the other properties as well. This is fairly slow to run, and a newer much much faster version was developed, but not fully implemented here for the same results. If needed, I can find and supply.

The xyz_to_isomerisationEapprox.py can run a GUI to do the simpler calculations such as dihedrals and approximate the isomerisation energy from the xyz structure (or recognise the molecule from an xyz-smiles conversion function that then matches to the COMPAS database to search for that structure exisiting within the database). NOTE: That in its current form, it seems to perform best for PAHs with the chemical formula fitting the pattern: C16+4nH8+2n.


Codes used directly in the PAHAPS webtool can be found at https://github.com/Kasigee/web-isomer-app
