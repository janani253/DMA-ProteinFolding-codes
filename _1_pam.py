import numpy as np
import os
from Basic import *


######################################################################
# Read & split clusters
def splitClusts(native_fn, trj_fn_fmt, folder_na, clust_fn, stride=1, createClusts=False): 

    clusters = np.loadtxt(clust_fn+".clusters", dtype=int) - 1
    medoids = np.loadtxt(clust_fn+".medoids", dtype=int) - 1
    clussize = np.loadtxt(clust_fn+".clusinfo", dtype=int, skiprows=1, usecols=1)
    # Create temp dir for temp-clusts
    os.system("rm -rf "+folder_na+"; mkdir -p "+folder_na)

    nl1 = 0

    digfmt1 = str(len(str(clusters.max())))
    digfmt1 = '{:0'+digfmt1+'d}'
    trajm = [0]*len(medoids)
    medsizemat = [0]*len(medoids)  
        
    fns = os.popen("ls "+trj_fn_fmt).read().split("\n")
    n1 = len(fns)
    for i in range(n1):
        trj_fn = fns[i].strip()	
        if len(trj_fn)==0:
            continue
        traj1 = md.load_dcd(trj_fn, top=native_fn, stride=stride)

        nt = clusters[nl1:nl1 + traj1.n_frames ]

        
        l = range(nl1, nl1 + traj1.n_frames )        
        # adding medoids
        for j in sorted(medoids):
            if j in l:
                k = np.where(medoids==j)[0][0]
                nt2 = np.where(l==j)[0] 
                trajm[k] = traj1[nt2]
                medsizemat[k] = clussize[medoids==j]
                


        if createClusts:
            # adding strucs to clusters
            for j in range(nt.min(), nt.max()+1):
                nt2 = np.where(nt==j)[0]
                traj2 = ""
                if nt2.shape[0]>0:
                    temp_fn = folder_na+"/clust-"+digfmt1.format(j)+".dcd"               
                    
                    if os.path.exists(temp_fn):
                        traj2 = md.load_dcd(temp_fn, top = native_fn)
                        traj2 = traj2.join(traj1[nt2])
                    else:
                        traj2 = traj1[nt2]
    
                    traj2.save_dcd(temp_fn)
        
        
        
        nl1 = nl1 + traj1.n_frames
        del traj1

    temp_fn = folder_na+"/medoids.dcd"
    trajm = md.join(trajm)
    trajm.save_dcd(temp_fn)

    temp_fn = folder_na+"/clusts.size"    
    np.savetxt(temp_fn,medsizemat, fmt="%d")

    del trajm
    #del traj2
######################################################################
