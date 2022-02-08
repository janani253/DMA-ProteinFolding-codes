# DMA-ProteinFolding-codes

File: 0-run-measures-f1.py
This python code can be used to calculate measures like RMSD, d-RMSD, Q, hc-RMSD from the native configuration as well as the HISASA for each conformation.
This program further uses methods from _0_plot_measures_f1.py to plot the above mentioned measures. Additionally it uses classes and methods from Basic.py to calculate and record above mentioned measures.

File: 1-cluster.py 
This python code can be used to cluster each frame in the trajectory based on their d-RMSD from each other. The k-medoid clustering algorithm was used in R to do so. This program uses the file 1-pam-clustering.r to perform the clustering in R. This program uses methods from _1_raw_data_mats.py to calculated raw data matrices which will be used by R to perform the k-medoid clustering. Additionally this program also uses methods from _1_pam.py to generate dcd files representing the clusters and medoids calculated by R in (1-pam-clustering.r).

File : 2-DMAP.ipynb
This code (excuable in Jupyter notebook) was used to 
1. calculate the diffusion maps generated in this study.
2. select conformations to serve as starting points for subsequent rounds of Particle swarm evolution based on the diffusion maps calculated, and the ansatz.
1. The first cell in this Jupyter notebook was used to define the protein ID, the distance metric used and import necessary packages
2. The 2nd cell (method simpleTrajDMap) may be used to perform DMA from small trajectories (<2000 frames), (nystrom extension may be performed on the same)
3. the 3rd cell (method CGTrajDMap) may be used to perform DMA from coarse-grained representatives.
4. The final cell (method compareTrajDMap) in this Jupyter notebook was used to Compare diffusion maps between trajectories and reports their correlation coefficient.
This Jupyter notebook uses _2_dm_class.py to perform DMA and also uses Basic.py to calculate distance metrices.
