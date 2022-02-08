import numpy as np
import mdtraj as md
from scipy.signal import savgol_filter
from itertools import combinations 
import copy
import os

digfmt = '{:03d}'
##########################################################################
class UnivData:
    ######################################################################    
    #Init functions
    def __init__(self, native_fn=False):
        if native_fn:
            self.native = native = md.load_pdb(native_fn)
        self.dist_fun = 0 
        self.op_name = 0
        self.kernel = 0 
        self.eps = 0

    ######################################################################  
##########################################################################


##########################################################################    
class Traj:
    trajo = []
    nlo = 0

    ######################################################################    
    #Init functions
    def __init__(self, trj_fn, native_fn, stride=1):
        self.traj = []
        self.nl = []
        
        if isinstance(trj_fn, str):
            if ".dcd" in trj_fn:
                self.traj = md.load_dcd(trj_fn, top = native_fn, stride=stride) 
            elif ".xtc" in trj_fn:
                self.traj = md.load_xtc(trj_fn, top = native_fn, stride=stride) 
            elif  ".pdb" in trj_fn:
                self.traj = md.load_pdb(trj_fn, stride=stride) 
            self.nl = self.traj.n_frames
            #Traj.trajo.extend(self.traj)

        elif isinstance(trj_fn, list):
            for i in trj_fn:   
                traj = []             
                if ".dcd" in i:
                    traj = md.load_dcd(i, top = native_fn, stride=stride)
                    self.traj.extend(traj )
                elif ".xtc" in i:
                    traj = md.load_xtc(i, top = native_fn, stride=stride)
                    self.traj.extend(traj )
                elif  ".pdb" in i:
                    traj = md.load_pdb(i, stride=stride)
                    self.traj.extend(traj ) 
                self.nl.append(traj.n_frames)
                del traj
            #Traj.trajo.extend(self.traj)
            #self.traj = md.join(self.traj)
        #Traj.trajo = md.join(Traj.trajo)
        #self.nl = self.traj.n_frames
        #Traj.nlo = Traj.nlo + self.nl
    ######################################################################  


    ######################################################################    
    #Reset
    def resetTraj(self):
        Traj.trajo = []
        Traj.nlo = 0
        self.nl = 0
        self.traj = []
    ######################################################################    


    ######################################################################    
    #Write DCD
    def writeDCD(self, fn="t", traj=np.array([]), stride=1):
        folderna = os.path.split(fn)[0]
        if len(folderna)!=0:
            os.system("mkdir -p "+folderna)

        if traj.shape[0]==0:
            traj = traj[::stride]
        else:
            traj = Traj.trajo[::stride]
        traj.save_dcd(fn)
    ######################################################################  


    ######################################################################  
    # write/save DCDs by indices
    def writeDCDbyN(self, nq, fn="t"):
        folderna = os.path.split(fn)[0]
        if len(folderna)!=0:
            os.system("mkdir -p "+folderna)

        if not isinstance(nq[0], (list, np.ndarray, tuple)):
            trj_fn = fn+".dcd"
            traj = self.traj[nq]
            traj.save_dcd(trj_fn)
            return
        else: #if isinstance(nq[0], (list, np.ndarray, tuple)):
            n = str(len(nq))

            for i in range(len(nq)):
                trj_fn = fn+"-"+str(i)+".dcd"
                traj = self.traj[nq[i]]
                traj.save_dcd(trj_fn)

    ######################################################################

    ######################################################################  
    # write/save DCDs by indices
    def writePDBbyN(self, nq, fn="t"):
        folderna = os.path.split(fn)[0]
        if len(folderna)!=0:
            os.system("mkdir -p "+folderna)

        if not isinstance(nq[0], (list, np.ndarray, tuple)):
            trj_fn = fn+".pdb"
            traj = self.traj[nq]
            traj.save_pdb(trj_fn)
            return
        else: 
            n = str(len(nq))

            for i in range(len(nq)):
                trj_fn = fn+"-"+str(i)+".pdb"
                traj = self.traj[nq[i]]
                traj.save_pdb(trj_fn)

    ######################################################################  
    

    ######################################################################    
    #Extract Traj frames by Q limits
    def extractByQ(self, q, qrange=[0,1], qx=0.5):
        N = int(len(q)**0.5)
        q = np.array(q)
        qm = runningMean(q,N, centralFlag=True)
        qrange = sorted(qrange)
        

        n1 = np.where([q>qrange[0]])[-1]
        n2 = np.where([q<qrange[1]])[-1]
        n = np.intersect1d(n1,n2)

        if qx>qrange[1]: 
            print("Looking for unfolded structures")       
            m1 = np.where([qm>qrange[0]])[-1]
            m2 = np.where([qm<(2.5*qrange[1])])[-1]
        elif qx<qrange[0]:
            print("Looking for folded structures")
            m1 = np.where([qm>(qrange[0]/2)])[-1]
            m2 = np.where([qm<qrange[1]])[-1]
        
        m = np.intersect1d(m1,m2)
        n = np.intersect1d(m,n)

        n1 = []
        nq = []
        
        for i in range(len(qm)):
            if (qx>qrange[1] and qm[i]<=qx) or (qx<qrange[0] and qm[i]>=qx):
                n1.append(i)
            elif (qx>qrange[1] and qm[i]>qx) or (qx<qrange[0] and qm[i]<qx):
                n2 = np.intersect1d(n,n1)
                if n2.shape[0]>0:
                    nq.append(n2)
                    n1 = []
                    n2 = []
        
        n2 = np.intersect1d(n,n1)
        if n2.shape[0]>0:
            nq.append(n2)
            n1 = []
            n2 = []
        print(len(nq))
        nq = self.collate(n, nq, N//10)

        import matplotlib.pyplot as plt
        plt.plot(qm)
        plt.show()
        return nq
        
    ######################################################################  


    ######################################################################  
    # Collate distributed trajs into appropriate groups by size and gap
    def collate(self, n, nq, N):
        n1 = [nq[0]]
        if type(n) is np.ndarray:
            n = n.tolist()
        if type(nq) is np.ndarray:
            nq = nq.tolist()

        nl = len(n)/N
        print(nl)
        i = 1
        for i in range(1,len(nq)-1):
            if n1[-1].shape[0]<=nl:   
                n1[-1] = np.insert(n1[-1], n1[-1].shape[0], nq[i])
            elif nq[i].shape[0]<=nl:  
                if nq[i][0]-n1[-1][-1]<nq[i+1][0]-nq[i][-1]:
                    n1[-1] = np.insert(n1[-1], n1[-1].shape[0], nq[i])
                else:
                    n1.append(nq[i])
            else:
                n1.append(nq[i])
        
        if nq[-1][-1]!=n1[-1][-1]:
            if n1[-1].shape[0]<=nl:   
                n1[-1] = np.insert(n1[-1], n1[-1].shape[0], nq[-1])
            elif nq[i].shape[0]<=nl:  
                n1[-1] = np.insert(n1[-1], n1[-1].shape[0], nq[-1])
            else:
                n1.append(nq[-1] )
       
        nq = copy.deepcopy(n1)
        n1 = []

        for i in range(len(nq)):
            print(i, nq[i][0], nq[i][-1])

        return nq
        #self.writeDCD_by_N(nq,fn)
    ######################################################################  



    
    ######################################################################  
    #Compute the secondary structure content of each structure in the trajectory
    def computeDSSP(self, native_struc):
        ssn = md.compute_dssp(native_struc) 
        ss = md.compute_dssp(self.traj)
    
        ssn1 = copy.copy(ssn)
        ss1 = copy.copy(ss)
        ssn1[ssn1 != "H"] = 1
        ss1[ss1 != "H"] = 2
        h = np.count_nonzero(ss1 == ssn1, axis = -1)
        ssn1 = copy.copy(ssn)
        ss1 = copy.copy(ss)
        ssn1[ssn1 != "E"] = 1
        ss1[ss1 != "E"] = 2
        s = np.count_nonzero(ss1 == ssn1, axis = -1)
    
        return h, s
    ######################################################################
##########################################################################


##########################################################################
class Measures():

    nc = []
    order_param = []
    wts = []
    q = []
    rmsd = []
    dhd = []
    #hisasa = []
    hisasaall = []
    hisasahp = []
           
    ######################################################################    
    #Init functions        
    def __init__(self, data, traj):
        self.data = data
        self.traj = traj
        #self.order_param = []
        #self.wts = []
        self.q = [] 
        self.rmsd = [] 
        self.dhd = [] 
        #self.hisasa = []
        self.hisasaall = []
        self.hisasahp = []
        self.hpq = []
        self.qm = [] 
        self.rmsdm = [] 
        self.dhdm = [] 
        self.hisasaallm = []
        self.hisasahpm = []
        self.hpqm = []
        
        # Read HP scales frm file
        def read_hp_scales(hpid):       
            # Hydrophobicity index   
            hp_scales = np.loadtxt("hp_scales.csv", dtype=str, delimiter=",")
            res = np.char.upper(hp_scales[0][1:])
                      
            nhpid = np.where(hp_scales[:,0]==hpid)[0][0]
            hidict = {}
            hp = hp_scales[nhpid][1:]
            hp = hp.astype(float)  

            for i in range(len(res)):                    
                hidict[res[i]] = hp[i] #(hp[i] - hp.mean() )/ hp.std()
            if "NLE" not in hidict.keys():
                hidict["NLE"] = (hidict["ILE"]+hidict["LEU"])/2
                res = list(hidict.keys())  
            return hidict      
        
        # Effective partition energy (Miyazawa-Jernigan, 1985) - for HP scales
        hidict = read_hp_scales(hpid = "MIYS850101" )
        
        # Max SA of a residue
        #msa = {"ALA":1.29, "ARG":2.74, "ASN":1.95, "ASP":1.93, "CYS":1.67, 
        #       "GLN":2.25, "GLU":2.23, "GLY":1.04, "HIS":2.24, "ILE":1.97,
        #       "LEU":2.01, "LYS":2.36, "MET":2.24, "NLE":2.00, "PHE":2.40, "PRO":1.59, 
        #       "SER":1.55, "THR":1.72, "TRP":2.85, "TYR":2.63, "VAL":1.74}   
        msa = {"ALA":1.4720, "ARG":2.7427, "ASN":2.0270, "ASP":1.9300, "CYS":1.6717, 
               "GLN":2.2802, "GLU":2.3850, "GLY":1.0475, "HIS":2.3737, "ILE":1.9716,
               "LEU":2.4987, "LYS":2.5916, "MET":2.5413, "NLE":2.3000, "PHE":2.4415, "PRO":1.7332 , 
               "SER":1.5558, "THR":1.9359, "TRP":3.2901, "TYR":2.6323, "VAL":1.8838}          
        
        # Residue side-chain Length
        rldict = {"ALA":1.5,   "ARG":7.23,  "ASN":3.75,  "ASP":3.57,  "CYS":2.75, 
                  "GLN":4.89,  "GLU":4.96,  "GLY":0.5,   "HIS":4.62,  "ILE":3.91,
                  "LEU":3.90,  "LYS":6.29,  "MET":5.25,  "NLE":3.90,  "PHE":4.99, "PRO":1.75, 
                  "SER":2.46,  "THR":2.46,  "TRP":5.93,  "TYR":6.31,  "VAL":2.46}         
                                    
        
        ### Define alphas
        self.alpha = self.data.native.topology.select_atom_indices('alpha')     
        nres = 6
        self.NATIVE_CUTOFF = 11.0  # A
        self.b = 1
        self.c = 2
        if len(self.alpha)/2<nres:
            nres = np.round(len(self.alpha)**0.5)
            self.alpha = self.data.native.topology.select_atom_indices('heavy')    
            self.NATIVE_CUTOFF = 4.5  # A  
            self.b = 1.5
            self.c = 3       
        self.alpha_pairs = np.array(
        [(i,j) for (i,j) in combinations(self.alpha, 2)
            if abs(self.data.native.topology.atom(i).residue.index - \
                   self.data.native.topology.atom(j).residue.index) > nres])  
        
        ### Collate Distributed Residues        
        self.ri = []
        self.riall = []
        for i in self.data.native.topology.residues:
            if str(i) not in self.ri:
                self.ri.append(str(i))
            self.riall.append(str(i))
        self.ri = np.array(self.ri)
        self.riall = np.array(self.riall)
               
        ### Prep for HISASA
        self.resmsa = []
        self.reshi = []
        
        for i in self.ri:
            j = str(i)[:3]
            self.resmsa.append(msa[j])
            self.reshi.append(hidict[j])
            #self.reshi.append(1/(1-hidict[j]) ) 
                  
        #self.reshi = savgol_filter(np.array(self.reshi), 3, 1, mode="mirror").tolist()                     
        
        self.probe_radius = 0.14        
        allatoms = self.data.native.topology.select_atom_indices('all')
        if len(allatoms)==len(self.alpha):
            self.probe_radius = 0.5
        
        ### Prep for hpContacts   
        hydres = ["ALA", "CYS", "HIS", "ILE", "LEU", "MET", "NLE", "PHE", "TRP", "TYR", "VAL"] 
        #hydres = ["ALA", "ILE", "LEU", "MET", "NLE", "PHE", "TRP", "TYR", "VAL"]        
        #hydres = ["ALA", "CYS", "ILE", "LEU", "MET", "NLE", "PHE", "TRP", "VAL"]
        
        #Mean fractional area loss (Rose et al., 1985) for HP probabilities
        #hidict = read_hp_scales(hpid = "ROSG850102") #"BASU050101") 
        h1 = []
        for i in hidict.keys():
            if hidict[i]>0:
                h1.append(i)
        hydres = set(hydres) & set(h1)
        hydresno = []
        hi = []
        rl = []        
        res = []
        j = 0
        
        for i in self.ri:
            hi1 = hidict[str(i)[:3]]
            if str(i)[:3] not in hydres:
                hi1 = 0
            hi.append(hi1)
            rl.append(rldict[str(i)[:3]])
            res.append(str(i))
            if str(i)[:3] in hydres:
                hydresno.append(j)
            j = j+1
        hi = np.array(hi).clip(min=0)  
        hydresno = np.unique(hydresno)  
                                            
        self.hp_contacts = np.array(
        [(i,j) for (i,j) in self.alpha_pairs                          
               if self.data.native.topology.atom(i).residue.index in hydresno or 
                  self.data.native.topology.atom(j).residue.index in hydresno])  
                          
        self.natcont_hi = []
        self.minbondlen = []
        for (i,j) in self.hp_contacts:
            x = self.data.native.topology.atom(i).residue.index 
            y = self.data.native.topology.atom(j).residue.index 
            z = hi[x]+hi[y]  
            self.natcont_hi.append(z)
            z = rl[x]+rl[y]
            self.minbondlen.append(z)
            
        self.natcont_hi = np.array(self.natcont_hi)
        self.minbondlen = np.array(self.minbondlen)
        
        ### Prep for DEShawQ        
        # compute the distances between these pairs in the native state
        alpha_pairs_distances = md.compute_distances(self.data.native[0], self.alpha_pairs)[0] * 10.0
        # and get the pairs s.t. the distance is less than NATIVE_CUTOFF
        self.native_contacts = self.alpha_pairs[alpha_pairs_distances < self.NATIVE_CUTOFF]
        #print("Number of native contacts", len(native_contacts))
        
        
        ### Wts prep for dihedDist  
        alpha = self.alpha[1: -2]
        self.alpha4 = [(alpha[i-1], alpha[i], alpha[i+1], alpha[i+2]) for i in range(1,len(alpha)-2)]
        self.coeff = 1/(np.arange(len(self.alpha4)) +1)
        self.coeff = np.cumsum(self.coeff)
        self.coeff = self.coeff+self.coeff[::-1]
        self.coeff = self.coeff/self.coeff[0]     
        self.coeff = np.ones(len(self.alpha4))
    ###################################################################### 

    
    ######################################################################    
    #Assign Distance Function name and corresponding epsilon
    def assignFuncs(self, dist_fun, kernel): 
        self.data.kernel = kernel

        #dist_fun
        if dist_fun=="dihedDist":
            self.data.dist_fun = self.dihedDist
            self.data.op_name = "dihed"
            Measures.wts = copy.deepcopy(self.coeff)
        elif dist_fun=="RMSD":
            self.data.dist_fun = self.RMSD
            self.data.op_name = "rmsd" 
        elif dist_fun=="contactDist":
            self.data.dist_fun = self.DEShawQ
            self.data.op_name = "cd" 
            #Measures.wts = copy.deepcopy(self.natcont_hi)
        elif dist_fun=="hpqDist":
            self.data.dist_fun = self.hpContacts
            self.data.op_name = "hpq" 
            Measures.wts = copy.deepcopy(self.natcont_hi)  
        elif dist_fun=="HISASADist":
            self.data.dist_fun = self.HISASA
            self.data.op_name = "hisasa"
            Measures.wts = copy.deepcopy(self.reshi)
    ###################################################################### 

    
    ######################################################################    
    # Kernel functions
    ######################################################################    
    
    ######################################################################    
    #Gaussian kernel
    def gaussianKernel(self, x):
        return np.exp(-(x**2)/(2*self.data.eps))
        #return np.exp(-(x**2)/self.data.eps)
    ######################################################################    


    ######################################################################    
    #Sigmoid kernel
    #def sigmoidKernel(self, x, x0=0, const=100, b=1.8, c=-2):
    def sigmoidKernel(self, x, x0=0, const=10, b=1, c=2):
        return 1.0 / (1 + np.exp(const * (x - x0*b - c)))
    ######################################################################    

    
    ######################################################################    
    #Measure functions
    ######################################################################    


    ######################################################################    
    # Compute Hydrophobicity of the protein based on Solvent Accesible Surface Area
    def HISASA(self, traj1=False ):
        
        hydres = ["ALA", "ILE", "LEU", "MET", "NLE", "PHE", "TRP", "TYR", "VAL"] 
        
        if isinstance(self.traj.nl, list) :
            traj0 = md.join(self.traj.traj)
            sasa = md.shrake_rupley(traj0, probe_radius=self.probe_radius, n_sphere_points=960, mode="residue") 
        else:
            sasa = md.shrake_rupley(self.traj.traj, probe_radius=self.probe_radius, n_sphere_points=960, mode="residue")                
        sasat = np.zeros([sasa.shape[0], len(self.ri)])
        for i in range(len(self.riall)):
            j = self.riall[i]
            k = np.where(self.ri==j)[0][0]
            l = np.where(self.riall==j)[0]            
            sasat[:,k] = sasat[:,k] + sasa[:,i]
        sasa = sasat/self.resmsa  
                
        sasa[:,0] = sasa[:,0] / 1.5 
        sasa[:,-1] = sasa[:,-1] / 1.5 
        
                
        hisasa = sasa * self.reshi
        if self.data.op_name=="hisasa":  
            Measures.order_param.extend(sasa.tolist()) 
            
        
        hisasaall = hisasa.mean(axis=-1)    
        hisasahp = copy.deepcopy(hisasa)
        for i in range(len(self.ri)):
            if self.ri[i][:3] not in hydres:
                hisasahp[:,i] = 0
        hisasahp = hisasahp.mean(axis=-1)                  
        #hisasahp = hisasa.clip(min=0.0).mean(axis=-1)                
        
        
        self.hisasaall.extend(hisasaall)
        self.hisasahp.extend(hisasahp)
        self.hisasaallm.append((3*np.median(hisasaall)-(2*np.mean(hisasaall))))
        self.hisasahpm.append((3*np.median(hisasahp)-(2*np.mean(hisasahp))))
        #return hisasaall.tolist(), hisasahp.tolist()
    ######################################################################    
    
    
    ######################################################################    
    #Calc RMSD
    def RMSD(self, traj1=False, flag="calc"):
        # get the indices of all of the CA
        rmsd = []
        rmsd1 = []
        
        if not traj1:
            traj1 = self.data.native
            
        if isinstance(self.traj.nl, list):
            traj0 = md.join(self.traj.traj)
            if len(traj1)>0:
                for traj2 in traj1: 
                    t = md.rmsd(traj0, traj2, atom_indices=self.alpha, ref_atom_indices=self.alpha, parallel=True)               
                    rmsd.append(t)
                    rmsd1.extend(t)
        else:
            if len(traj1)>0:
                for traj2 in traj1:            
                    t = md.rmsd(self.traj.traj, traj2, atom_indices=self.alpha, ref_atom_indices=self.alpha, parallel=True)
                    rmsd.append(t)
                    rmsd1.extend(t)

        if flag=="clust":       
            return rmsd1 , rmsd
        elif flag=="matrix":     
            return np.array(rmsd)
        else:
            self.rmsd.extend(rmsd1)
            self.rmsdm.append((3*np.median(rmsd1)-(2*np.mean(rmsd1))))
    ######################################################################    

    
    ######################################################################    
    #Calc dihedral distance 
    def dihedDist(self, traj1=False):
        # get the indices of all of the CA        
        
        if isinstance(self.traj.nl, list):
            traj0 = md.join(self.traj.traj)
            dihed1 = md.compute_dihedrals(traj0,self.alpha4)
        else:
            dihed1 = md.compute_dihedrals(self.traj.traj,self.alpha4)
        
        if self.data.op_name=="dihed":  
            Measures.order_param.extend(dihed1.tolist())             
            
        if not traj1:
            traj1 = self.data.native
        dihed2 = md.compute_dihedrals(traj1,self.alpha4)    

        dhd = []

        if len(traj1)>0:           
            for i in range(len(traj1)):
               dhd1 = np.pi - abs(abs(dihed1 - dihed2[i]) - np.pi)
               dhd1 = np.abs(dhd1) * self.coeff     
               dhd1 = dhd1.mean(axis=1)
               dhd.extend(dhd1.transpose())   
        
        self.dhd.extend(dhd)            
        self.dhdm.append((3*np.median(dhd)-(2*np.mean(dhd))))            
        #return dhd
    ######################################################################     


    ######################################################################    
    #Calculate Fraction of contacts
    def hpContacts(self, traj1=False):
       
        # now compute these distances for the whole trajectory
        if isinstance(self.traj.nl, list):
            traj0 = md.join(self.traj.traj)
            r1 = md.compute_distances(traj0, self.hp_contacts) * 10
        else:
            r1 = md.compute_distances(self.traj.traj, self.hp_contacts) * 10
        BETA_CONST = 1.0  # 1/A   
        hpq = self.sigmoidKernel(x=r1,x0=self.minbondlen,const=BETA_CONST, c=4) #self.natcont_hi      
        
        
        if self.data.op_name in ["hpq","cd"]:
            Measures.order_param.extend(r1.tolist())          
        
        hpq = hpq.mean(axis=1)
        self.hpq.extend(hpq)
        self.hpqm.append((3*np.median(hpq)-(2*np.mean(hpq))))
      
    ######################################################################     
    
    
    ######################################################################    
    #Calculate Fraction of contacts
    def DEShawQ(self, traj1=False, flag="calc"):

        BETA_CONST = 10.0  # 1/A
        
        # define custom contacts
        if traj1:
            ### Prep  
            # compute the distances between these pairs in the native state
            alpha_pairs_distances = md.compute_distances(traj1, self.alpha_pairs)[0] * 10.0               
            # and get the pairs s.t. the distance is less than NATIVE_CUTOFF
            contacts = self.alpha_pairs[alpha_pairs_distances < self.NATIVE_CUTOFF]
        else:
            contacts = copy.deepcopy(self.native_contacts)
        
        # now compute these distances for the whole trajectory
        if isinstance(self.traj.nl, list):
            traj0 = md.join(self.traj.traj)
            r1 = md.compute_distances(traj0, contacts) * 10.0 
        else:
            r1 = md.compute_distances(self.traj.traj, contacts) * 10.0 
       
        if traj1:
            r2 = md.compute_distances(traj1, contacts) * 10.0             
        else:
            r2 = md.compute_distances(native, contacts) * 10.0        
            traj1 = native
               
        #q = self.sigmoidKernel(x=r1,x0=r2,const=BETA_CONST)
        q1 = []
        q2 = []
        if len(traj1)>0:
            for r3 in r2:   
                q = np.mean(self.sigmoidKernel(x=r1,x0=r3,const=BETA_CONST, b=self.b, c=self.c), axis=1)
                q1.extend(q)
                q2.append(q)
                
                
        
        if flag=="clust":   
            return q1, q2
        elif flag=="matrix":    
            return np.array(q2)
        else:
            self.q.extend(q1)
            self.qm.append((3*np.median(q1)-(2*np.mean(q1))))
        
        #return r1, r2
        
      
    ######################################################################  



    ######################################################################  
    #Extract specific frames of the trajectory
    def getQLims(self, q, qm):

        freqs, edges = np.histogram(qm, bins=20)
        mids = []
        Qunf = 0
        hf = 0
        Qf = 1
        for i in range(freqs.shape[0]):
            mids.append((edges[i]+edges[i+1])/2)

            if mids[i]<0.3 and hf<freqs[i]:
                hf = freqs[i]
                Qunf = mids[i]
            elif mids[i]>=0.3 and mids[i]<=0.7:
                hf = 0
            elif mids[i]>0.7 and hf<freqs[i]:
                hf = freqs[i]
                Qf = mids[i]

        return Qunf, Qf
    ######################################################################   

    
    ######################################################################  
    # Resetting class Measures
    def resetMeasures(self):
        Measures.q = []
        Measures.rmsd = []
        Measures.dhd = []
        Measures.hisasaall = []
        Measures.hisasahp = []
        Measures.hpq = []
        Measures.nc = []
        Measures.order_param = []

    ######################################################################   
    
    
    ######################################################################  
    # Resetting object measures to before NE state
    def reset2b4NE(self, maxnc):
                
        #self.q = np.array(self.q)[np.array(self.nc)<=maxnc].tolist()
        #self.rmsd = np.array(self.rmsd)[np.array(self.nc)<=maxnc].tolist()
        self.dhd = np.array(self.dhd)[np.array(self.nc)<=maxnc].tolist()
        #self.hisasaall = np.array(self.hisasaall)[np.array(self.nc)<=maxnc].tolist()
        #self.hisasahp = np.array(self.hisasahp)[np.array(self.nc)<=maxnc].tolist()
        #self.hpq = np.array(self.hpq)[np.array(self.nc)<=maxnc].tolist()
        Measures.order_param = np.array(Measures.order_param)[np.array(self.nc)<=maxnc].tolist()
        self.nc = np.array(self.nc)[np.array(self.nc)<=maxnc].tolist()
        

    ######################################################################   
 
   
    ######################################################################
    # Calc running means for all the 'distances' calc'd in 'calc' method
    def calcRunningMeans(self):
        # Calc running means 
        stride = 50   
        Measures.hisasaallm = runningMean(self.hisasaall, stride, centralFlag=True)     
        Measures.hisasahpm = runningMean(self.hisasahp, stride, centralFlag=True)    
        Measures.hpqm = runningMean(self.hpq, stride, centralFlag=True) 
        Measures.qm = runningMean(self.q, stride, centralFlag=True)       
        Measures.rmsdm = runningMean(self.rmsd, stride, centralFlag=True)
        Measures.dhdm = runningMean(self.dhd, stride, centralFlag=True)
        
        Measures.ts, Measures.qc = getTS(self.q, Measures.qm)

        #return qc, qm, qmr, rm, dhdm
    ######################################################################


    ######################################################################  
    #calling dist functions
    def call(self, traj1=False, flag=False):
        # Some stats to be later used in colouring plots
        
        if flag=="NE": 
            if isinstance(traj1, int):
                traj1 = [traj1]
            traj1 = np.unique(traj1)
            start = np.searchsorted(self.nc, traj1, side="left")
            end = np.searchsorted(self.nc, traj1, side="right")

            x = []         
            for i in range(len(traj1)): 
                x.extend(range(start[i],end[i]))  

            self.q = np.array(self.q)[x]
            self.rmsd = np.array(self.rmsd)[x]
            self.dhd = np.array(self.dhd)[x]
            self.hisasaall = np.array(self.hisasaall)[x]
            self.hisasahp = np.array(self.hisasahp)[x]
            self.hpq = np.array(self.hpq)[x]
            self.nc = np.array(self.nc)[x]
            
            self.nc = np.insert(self.nc, 0, [0]*len(self.dhdm))
            self.q = np.insert(self.q, 0, self.qm)
            self.rmsd = np.insert(self.rmsd, 0, self.rmsdm)
            self.dhd = np.insert(self.dhd, 0, self.dhdm)
            self.hisasaall = np.insert(self.hisasaall, 0, self.hisasaallm)
            self.hisasahp = np.insert(self.hisasahp, 0, self.hisasahpm)
            self.hpq = np.insert(self.hpq, 0, self.hpqm)
            
        else:
            #self.DEShawQ(traj1)            
            #self.RMSD(traj1)          
            self.dihedDist(traj1)
            #self.hpContacts(traj1)
            #self.HISASA()  
            pass          
            

        if flag in ["NE", "mean", "collate", "backup", "restart"]:
            return
        
        n = len(self.nc)
        if n==0:
            m = 0
        else:
            m = max(self.nc)

        if isinstance(self.traj.nl, (list,np.ndarray,tuple)):           
            for i in range(len(self.traj.nl)):
                n = len(self.nc)
                self.nc.extend([m+1]*self.traj.nl[i])
                m = max(self.nc)
        else:
            n = len(self.nc)
            self.nc.extend([m+1]*self.traj.nl)

    ###################################################################### 


##########################################################################
#Misc Functions
##########################################################################

##########################################################################
#Calc running mean of window N

def runningMean(x1, N=50, centralFlag=False):
    x = copy.deepcopy(x1)
    y = copy.deepcopy(x1)
    N = int(N)
   
    x = savgol_filter(x, 51, 1, mode="mirror").tolist()
    
    return x

##########################################################################


##########################################################################
#Given running mean calculate, transition states
def getTS(x, xm, xunf=0.2, xf=0.9, stride=50):
    a = []
    b = []
    i = 0

    while i<len(x)-1:
        if x[i]<=xunf and x[i+1]>=xunf:
            while i<len(x)-1:
                b.append(i)
                #i = i+1
                if x[i]>xunf and x[i+1]<xunf:
                    b = []
                    #i = i+1
                    break
                if x[i]>xf:
                    a.append(b)
                    b = []
                    #i = i+1
                    break
                else:
                    i = i+1
                    
        i = i+1 
                
    ts = list(flattenList(a))
    c = np.array(['b']*len(x))
    for i in ts:
        c[i] = 'r'
        
    return ts, c
##########################################################################


##########################################################################
def flattenList(nested_list):

    nested_list = copy.deepcopy(nested_list)
    
    while nested_list:
        sublist = nested_list.pop(0)
        if isinstance(sublist, list):
            nested_list = sublist + nested_list
        else:
            yield sublist
##########################################################################

  
##########################################################################
#Print to file
def print2file(pid,fx,x,fflag='full',writeflag='w'):

    folderna = os.path.split(pid)[0]
    if len(folderna)!=0:
        os.system("mkdir -p "+folderna)

    import time
    x = np.array(x)
    if len(x.shape)==1:
        x = x.reshape(x.shape[0],1)
    fo=open(pid+"."+fx,writeflag)
    print("Writing to file "+pid+"."+fx+" @ "+str(time.process_time()))
    #print(x.shape)
    for i in range(0,x.shape[0]): 
        lt=[]  
        for j in range(0,x.shape[1]):
            if fflag=='full':
                lt.append(str(x[i][j])) 
            else:
                lt.append('{:f}'.format(x[i][j]))        
        fo.write(" ".join(lt)+"\n")
    fo.close()
##########################################################################

