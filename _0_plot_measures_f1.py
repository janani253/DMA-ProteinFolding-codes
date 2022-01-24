import numpy as np
import matplotlib.pyplot as plt
from Basic import *


######################################################################  
#Plotting Measures
def plot(M):
    Measures.qc = Measures.qc.tolist()
    """
    plt.scatter(range(len(M.q)),M.q,s=1,c='r')
    plt.scatter(range(len(M.dhd)),np.array(M.dhd),s=1,c='g')
    plt.scatter(range(len(M.rmsd)),np.array(M.rmsd),s=1,c='r')
    plt.show()
    """
    plt.scatter(range(len(M.q)),M.q,s=1,c=Measures.qc)
    plt.plot(M.qm,c='r')
    plt.xlabel('Frame', fontsize=14)
    plt.ylabel('Q(X)', fontsize=14)
    plt.show()
    """
    plt.scatter(range(len(M.dhd)),M.dhd,s=1)
    plt.plot(np.array(M.dhdm)/max(M.dhdm),c='r')
    plt.xlabel('Frame', fontsize=14)
    plt.ylabel('Dihedral_Distance(X)', fontsize=14)
    plt.show()
    """
    plt.scatter(range(len(M.rmsd)),M.rmsd,s=1)
    plt.xlabel('Frame', fontsize=14)
    plt.ylabel('RMSD(X)', fontsize=14)
    plt.show()
    
######################################################################



######################################################################
def writeDCDbyN(nqs, fn="t"):
                
    #n1 = str(len(nqs))
    #digfmt1 = str(len(n1))
    #digfmt1 = '{:0'+digfmt1+'d}'
                
    i = 0
    j = 0
    prevnl = 1
    for i in range(len(nqs)):
        trj_fn = []
        flag = False
        while nqs[i][0]>nlcum[j] and flag==False:
            prevnl = nlcum[j]
            j = j+1
        else:        
            flag = True
        while nqs[i][-1]>=nlcum[j] and flag==True:
            trj_fn.append(fnp+str(j)+".dcd")
            j = j+1
        else:
            trj_fn.append(fnp+str(j)+".dcd")
            flag = False
                
        traj = Traj(trj_fn, native_fn, stride=stride)
        traj.traj = md.join(traj.traj)
        nq1 = nqs[i]%prevnl
        traj.writeDCDbyN(nq1, fn=fn+digfmt.format(i))


######################################################################
  

