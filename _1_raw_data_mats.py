import numpy as np
import mdtraj as md
from itertools import combinations
from Basic import *


######################################################################
# Raw Data
######################################################################


######################################################################
# Get Contact distances
def getContactDists(traj,native,fromnative=True):
    NATIVE_CUTOFF = 11.1  # A    
    # get the indices of all of the CA
    alpha = native.topology.select_atom_indices('alpha')
    # get the pairs of CA atoms which are farther than 7 residues apart
    alpha_pairs = np.array(
        [(i,j) for (i,j) in combinations(alpha, 2)
            if abs(native.topology.atom(i).residue.index - \
                   native.topology.atom(j).residue.index) > 7])   
    # compute the distances between these pairs in the native state
    alpha_pairs_distances = md.compute_distances(native[0], alpha_pairs)[0] * 10.0
    # and get the pairs s.t. the distance is less than NATIVE_CUTOFF
    native_contacts = alpha_pairs[alpha_pairs_distances < NATIVE_CUTOFF]    
    # now compute these distances for the whole trajectory
    r = md.compute_distances(traj, native_contacts) * 10.0        
    return r
######################################################################


######################################################################
# Get Diheral angles along backbone C-alphas
def getPsi(traj,native,fromnative=True):
    # get the indices of all of the CA
    start_res = 6
    end_res = 61
    alpha = native.topology.select_atom_indices('alpha')[start_res - 2 : end_res + 1]
    alpha4 = [(alpha[i-1], alpha[i], alpha[i+1], alpha[i+2]) for i in range(1,len(alpha)-2)]
    psi = md.compute_psi(traj)[1]
    return psi   
######################################################################


######################################################################
# Get Diheral angles along backbone C-alphas
def getDihed(traj,native,fromnative=True):
    # get the indices of all of the CA
    start_res = 6
    end_res = 61
    alpha = native.topology.select_atom_indices('alpha')[2 : -2]
    alpha4 = [(alpha[i-1], alpha[i], alpha[i+1], alpha[i+2]) for i in range(1,len(alpha)-2)]
    dihed = md.compute_dihedrals(traj,alpha4)
    return dihed    
######################################################################


######################################################################
# Get Angles along backbone C-alphas
def getAngles(traj,native,fromnative=True):
    # get the indices of all of the CA

    alpha = native.topology.select_atom_indices('alpha')[1 : -1]
    alpha3 = [(alpha[i-1], alpha[i], alpha[i+1]) for i in range(1,len(alpha)-1)]
    angles = md.compute_angles(traj,alpha3)
    return angles 
######################################################################


######################################################################
# Get Distances between adjacent backbone C-alphas
def getDists(traj,native,fromnative=True):
    # get the indices of all of the CA

    alpha = native.topology.select_atom_indices('alpha')
    alpha2 = [(alpha[i-1], alpha[i]) for i in range(1,len(alpha))]
    dists = md.compute_distances(traj,alpha2)
    return dists
######################################################################


######################################################################
# Sligtly processed Data
######################################################################


######################################################################
# Calculate Trigonometric Properties (Sine and Cosine) of Diheral angles
def getDihedTrig(traj,native,fromnative=True):    
    return getTrig(getDihed(traj,native))
######################################################################


######################################################################
# Calculate Trigonometric Properties (Sine and Cosine) of Diheral angles
def getPsiTrig(traj,native,fromnative=True):    
    return getTrig(getPsi(traj,native))
######################################################################


######################################################################
# Calculate Trigonometric Properties (Sine and Cosine) of given angles
def getTrig(angle,fromnative=True):
    s = np.sin(angle)
    c = np.cos(angle)
    #return s*c
    return np.concatenate((s, c), axis=1)
######################################################################


######################################################################
# Collate distances, angles and dihecrals to get (modified) internal coordinates
def getInternalCoords(traj,native,fromnative=True):
    r = getDists(traj,native)
    t = getTrig(getAngles(traj,native))
    p = getTrig(getDihed(traj,native))
    rtp = np.concatenate((r,t,p), axis=-1)
    return rtp
###################################################################### 


######################################################################
# Matrices
######################################################################


######################################################################
# Calc Fraction of contacts matrix
def getDEShawQMat(traj,native,fromnative=True):
    BETA_CONST = 100.0  # 1/A
    r = getContactDists(traj)
    r0 = getContactDists(native)
    if fromnative == False:
        q = np.array([np.mean(1.0 / (1 + np.exp(BETA_CONST * (abs(r - r0) - 2))), axis=1) for ri in r])
    else:
        q = np.mean(1.0 / (1 + np.exp(BETA_CONST * (abs(r - r0) - 2))), axis=1)
    return q
######################################################################


######################################################################
# Get RMSD Matrix
def getRMSDMat(traj,native,fromnative=True):
    # get the indices of all of the CA

    nf = traj.n_frames
    alpha = native.topology.select_atom_indices('alpha')
    if fromnative == False:
        #rmsd = np.array([md.rmsd(traj, traj, i, atom_indices=alpha, ref_atom_indices=alpha).astype(float16) for i in range(nf)])  
        rmsd = np.array([md.rmsd(traj, traj, i, atom_indices=alpha, ref_atom_indices=alpha) for i in range(nf)])  
    else:
        rmsd = md.rmsd(traj, native, atom_indices=alpha, ref_atom_indices=alpha)  
    return rmsd 
######################################################################






