import numpy as np
import mdtraj as md
import os
from _1_raw_data_mats import *
from _1_pam import *


##############################################################################################################
# Define files
##############################################################################################################
protid = "2WXC"
native_filename = "../"+protid+"/"+protid+".pdb"     # reference structure  
native = md.load_pdb(native_filename) 

ifNameFormat = "../"+protid+"/"+protid+"-*.dcd"
nid = "0"
clustfName = "../"+protid+"/clustering/clustering"+nid+"/"

nc = "500"
# DEfine dist fun
fx = "dhd"
dist_fun = getDihed
#getRMSDMat    #getContactDists    #getInternalCoords

digfmt = '{:03d}'

############################################################################################################
def calcDMat(fx="", stride=1):
	
	# To Write raw data to file
	if stride>1:
		pid = clustfName+"raw_mats/"+protid+nid+"-"+str(stride)
	else:
		pid = clustfName+"raw_mats/"+protid+nid
	os.system("rm "+pid+"."+fx)
	
	fns = os.popen("ls "+ifNameFormat).read().split("\n")
	n = len(fns)
	for i in range(n):
		trj_fn = fns[i].strip()	
		if len(trj_fn)==0:
			continue			

		
		traj = md.load_dcd(trj_fn, top = native_filename, stride=stride)		
		
		# Calc raw data
		y = dist_fun(traj, native, fromnative=False)

		del traj
		# Write raw data
		print2file(pid,fx,y,fflag='f',writeflag='a')
	return pid+"."+fx
	
############################################################################################################


############################################################################################################
def calcClusts(iDMat_fn="", nc="sqrt"):
	op_fmt = clustfName+"clusters/"
	os.system("rm -rf "+op_fmt+"; mkdir -p "+op_fmt)
	op_fmt = op_fmt+protid+nid+"-"+nc+"."+fx+".pam"
	if nc == "sqrt":
	    print(os.popen("wc -l "+iDMat_fn).read().split(" ")[0])
	    nc = np.round(np.sqrt(int(os.popen("wc -l "+iDMat_fn).read().split(" ")[0]) )).astype(int)
	os.system("Rscript 1-pam-clustering.r "+iDMat_fn+" "+op_fmt+" "+str(nc))
	
	return op_fmt
	

############################################################################################################

DMat_fn = calcDMat(fx=fx, stride=1)

oclust_fmt = calcClusts(iDMat_fn=DMat_fn, nc=nc)

splitClusts(native_fn=native_filename, trj_fn_fmt=ifNameFormat, folder_na=clustfName+fx+"-op_DCDs-"+nc, clust_fn=oclust_fmt, stride=1)









