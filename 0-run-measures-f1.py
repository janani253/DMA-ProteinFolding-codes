import numpy as np
import mdtraj as md
from _0_plot_measures_f1 import *
import os

##############################################################################################################
# Define files
##############################################################################################################
protid = "2F4K"
native_fn = "../"+protid+"/"+protid+"-folded.pdb"   	## Reference structure/folded structure to compare against
native = md.load_pdb(native_fn) 

ifNameFormat = "../"+protid+"/"+protid+"-[0-9]/"+protid+"-[0-9].dcd"			## Sample folder format when calculating Q,RMSD for DCDs 

data = UnivData(native_fn)
digfmt = '{:03d}'

############################################################################################################
def main(oneOP=False):
	trajo = []
	M = Measures(data, "")
	traj = ""
	nl = []
	stride = 1
	
	
	fns = os.popen("ls "+ifNameFormat).read().split("\n")
	n = len(fns)

	for i in range(n):
		del traj
		if not oneOP:
			M = Measures(data, "")
		trj_fn = fns[i].strip()
		if len(trj_fn)==0:
			continue
		traj = Traj(trj_fn, native_fn, stride=stride)

		nl.append(traj.nl)

		M.traj = traj
		M.call(data.native, flag="calc")
		if not oneOP:
			M.calcRunningMeans()
			#plot(M)
			print2file(protid+"-"+str(i), "Q", 1-M.rmsd/np.max(M.rmsd), fflag='full',writeflag='w')
			M.resetMeasures()
	if oneOP:
		M.calcRunningMeans()			
		#plot(M)
		print2file(protid, "Q", 1-M.rmsd/np.max(M.rmsd), fflag='full',writeflag='w')	
	del traj	
	M.resetMeasures()
############################################################################################################

main(oneOP=False)		




