import numpy as np
import mdtraj as md
from matplotlib.pyplot import *
import time
import sys
from Basic import *


class DMAPs():
    ######################################################################    
    #Init function
    def __init__(self, data):
        self.data = data
        self.n1 = 0
        self.n2 = 0
        self.op1 = 0
        self.op2 = 0
    ######################################################################   

    
    ######################################################################    
    #Nystrom ext ension
    def nystromE(self, Pn, eVeco, eVal):
        #so = eVal.shape[0]
        #sn = Pn.shape[0]
        Fn = Pn#[so:sn,0:so]
        eVecn = np.dot(Fn,eVeco)/eVal
        #eVecn = np.concatenate((eVeco,eVecn), axis=0)
        #print2file('eVecn',eVecn) #.transpose()
        return eVecn
    ######################################################################    

    
    ######################################################################    
    #Functions related to diffusion map calculation
    ######################################################################    

    ######################################################################    
    #Calc Ai
    def calcAi(self, a):
        a = a.transpose()
        n = a.shape[0]
        ai = np.zeros(n) #, dtype=np.float32)
        for i in range(0,n): 
            ai[i] = np.sum(a[i])
        return ai
    ######################################################################    

    
    ######################################################################    
    #Calc P for simple single traj
    def calcP(self, a, ai, aj, flag='calc'):
        p = 0
        n1 = a.shape[0]
        n2 = a.shape[1]
        if flag!='file':
            p = np.zeros((n1, n2), dtype=np.float32)

        for i in range(0,n1): 
            if flag!='file':
                p[i] = self.calcPWDi(a[i],ai, i, aj)
            else:
                p = self.calcPWDi(a[i], ai, i, aj)
                print2file('p',np.array([p]),fflag='full',writeflag='a')
        a = np.delete(a, np.s_[::])
        ai = np.delete(ai, np.s_[::])
        
        if flag!='file':
            return p
        else:
            return 0
    ######################################################################    


    ######################################################################    
    #Calc P,W,D
    def calcPWDi(self, a, ai, i, aj):
        alpha = 0.5
        n = ai.shape[0]
        w = np.zeros(n, dtype=np.float16)
        d = 0
        
        for j in range(0,n):
            w[j] = a[j]/((aj[i]**alpha)*(ai[j]**alpha))
            d = d+w[j]
        p = w/d
        w = np.delete(w, np.s_[::])
        d = np.delete(d, np.s_[::])

        return p
    ######################################################################    


    ######################################################################    
    #Calc Ev
    def calcEV(self, p):
        eVal,eVec = np.linalg.eig(p)
        p = np.delete(p, np.s_[::])
        return eVal,eVec
    ######################################################################    


    ######################################################################    
    #Get A from file
    def getDFrmFile(self, fi, nl):
        
        fi = open(fi,'r')
        
        d = np.zeros([nl,nl])#, dtype=np.float16)
        i = 0
        j = 0
        for l in fi.readlines():
            if i>=nl:
               break
            lt = l.strip().split(" ")
            for j in range(0,i+1):
                d[i][j] = lt[j]
                d[j][i] = lt[j]
            i = i+1
        del lt
        fi.close()
        return d
    ######################################################################    

    
    ######################################################################    
    #Get Matrix
    def getDMat(self, n1=1, n2=1, flag="calc"):
        if self.data.op_name in ["RMSD","rmsd"]:
            return 0
        a = []

        nc = np.array(Measures.nc)
        if n1!=self.n1:
            self.op1 = np.array(Measures.order_param)[nc==n1]
            self.n1 = n1
        if n2!=self.n2:
            self.op2 = np.array(Measures.order_param)[nc==n2]
            self.n2 = n2

        for i in range(len(self.op2)):        
            r = self.op1 - self.op2[i] 
            if self.data.op_name == "dihed":
                r = np.pi - abs(np.abs(r) - np.pi)  

            r = (r**2) * Measures.wts
            r = r.sum(axis=-1)/np.sum(Measures.wts)
            r = np.sqrt(r)   
            self.d1.extend(r)           
            a.append(r.transpose().tolist())                 
                                
        if flag=="clust": 
            return a #np.mean(a, axis=-1), np.mean(a, axis=0)
        else:
            return np.array(a)
    ######################################################################

    
    ######################################################################    
    #Calc and plot Epsilon
    def toCalcEpsilon(self, verbose=True):
        #global eps
        a = [] 
        ec = []
        d = copy.deepcopy(self.d)   
        #self.d1 = copy.deepcopy(self.d)
        #for i in range(self.d1.shape[0]):
        #    self.d1[i,i] = 0    

        for e1 in range(-5,5,1):            
            for e2 in range(1,10,1):
                self.data.eps = e2*(10**e1)
                c = self.data.kernel(np.array(self.d1))
                c = c.sum()
                a.append(np.log(c))
                ec.append(np.log(self.data.eps))
        if verbose:
            plot(ec,a, c='b')

        from scipy.interpolate import interp1d
        f = interp1d(a, ec)
        amid = (max(a)+min(a))/2
        ecmid = f(amid)
        if verbose:
            scatter(ecmid,amid, s=10, c='r')
            show()
            print(amid, np.exp(ecmid))
        return np.exp(ecmid)
        
    ######################################################################    
    
    
    ######################################################################    
    #calculating distances for simple trajectories
    def calcDMatSimple(self, traj1, n1=1, n2=False, fn="t", fx="RMSD", verbose=True, returnDMat=False):
                    
        if self.data.op_name in ["RMSD","rmsd","cd","CD"]:                          
            d = self.data.dist_fun(traj1.traj, flag="matrix").transpose()
            self.d1 = copy.deepcopy(d)
                            
        else:
            if isinstance(n1,int):
                n1 = [n1]               
            if not n2:
                n2 = copy.deepcopy(n1)
            if isinstance(n2,int):                     
                n2 = [n2]               
            
            n1 = np.unique(n1)
            n2 = np.unique(n2)

            for i in n2:
                d = np.array([])
                for j in n1:                
                    if d.shape[0]==0:
                        d = self.getDMat(n1=j, n2=i)
                    else: 
                        d1 = self.getDMat(n1=j, n2=i)
                        d = np.concatenate((d,d1), axis=-1)
                      
        """            
        # make symmetric   
        for i in range(self.d.shape[0]):
            for j in range(i,self.d.shape[1]):             
                 if self.d[i,j]!=self.d[j,i]:                        
                     self.d[i,j] = (self.d[i,j]+self.d[j,i])/2
                     self.d[j,i] = self.d[i,j]     
        """                                                         
        if verbose:
            print("No. of samples: "+str(len(d)))
            

        #for i in range(d.shape[0]):           
        #    d[i,i] = sorted(d[i])[1] **2                 
        return d
    ######################################################################    
    
    
    ######################################################################  
    #calling DMAP functions for simple trajectories
    def callSimple(self, traj1, n1=1, fn="t", fx="RMSD", wFlag=False, verbose=True):
        flag = 'calc'       
        if verbose: 
            print("Calculating A @ "+str(time.process_time()))
        
        self.d1 = [] 
        self.d = self.calcDMatSimple(traj1, n1=n1, n2=False, fn=fn, fx=fx, verbose=verbose) 
                
        
        if wFlag:
            print2file(fn,fx,self.d,fflag='full',writeflag='w')
        
                           
        self.data.eps = self.toCalcEpsilon(verbose=verbose)

        if verbose:
            print("Epsilon : "+str(self.data.eps))    
        
           
        self.a = self.data.kernel(np.array(self.d))
        
        for i in range(self.a.shape[0]):           
            self.a[i,i] = np.sqrt(sorted(self.a[i],reverse=True)[1] )
            self.d[i,i] = (-2 * self.data.eps * np.log(self.a[i,i]) )**0.5

                   
        ai = self.calcAi(self.a) 
        aj = self.calcAi(self.a.transpose()) 

        if verbose:
            print("Calculating Po @ "+str(time.process_time()))
        self.P = self.calcP(self.a, ai, aj, flag=flag)
        if verbose:
            print("Calculating eVeco @ "+str(time.process_time()))
        self.eVal, eVec = self.calcEV(self.P)
        self.eVec = eVec.transpose()
        if verbose:
            print("eVeco shape: "+str(self.eVec.shape))               
        
        self.trajo = traj1
        
        ai = np.delete(ai, np.s_[::])
        aj = np.delete(aj, np.s_[::])  
        del traj1
        
        return self.eVal, self.eVec
        
    ######################################################################
    
    
    ######################################################################  
    #calling functions to calc DMAP for simple trajectories using Nystrom Etension
        
    def callSimpleNE(self, traj1, n1=1, n2=2, fn="t", fx="RMSD", wFlag=False, verbose=True):
        flag = 'calc1' 
        
        if verbose:
            print("Calculating An @ "+str(time.process_time()))
        
            
        d = self.calcDMatSimple(traj1, n1=n1, n2=n2, fn=fn, fx=fx, verbose=verbose) 
        a = self.data.kernel(d)  
        self.a = np.concatenate((self.a,a), axis=0)
        #self.a = np.insert(self.a, self.a.shape[0], a, axis=0).transpose()
        
        if wFlag:
            print2file(fn, fx, d, fflag='full', writeflag='w')
        
              
        ai = self.calcAi(self.a)
        aj = self.calcAi(self.a.transpose())

        if verbose:
            print("Calculating Pn @ "+str(time.process_time()))
        Pn = self.calcP(self.a, ai, aj, flag=flag) #.transpose()

        
        if verbose:
            print("Calculating eVecn using Nystrom Extension @ "+str(time.process_time()))
        eVec = self.nystromE(Pn, self.eVec.transpose(), self.eVal)
        eVec = eVec.transpose()
        if verbose:
            print("eVecn shape: "+str(eVec.shape))
               
        
        self.a = self.a[:self.a.shape[1]]

        ai = np.delete(ai, np.s_[::])
        aj = np.delete(aj, np.s_[::])
        Pn = np.delete(Pn, np.s_[::])  
        del traj1
        self.n1 = 0
        self.n2 = 0
               
        return eVec
    ######################################################################  
    
    
    ######################################################################  
    #calling DMAP functions for clusters/CG trajectories
    def callCG(self, native_fn, ifNameFormat, dist_fun="RMSD", nclusts=20, fn="t", fx="RMSD", stride=1, wFlag=False, verbose=True):
               
        M = Measures(self.data, "")
        M.resetMeasures()
        M.assignFuncs(dist_fun = dist_fun, kernel = M.gaussianKernel)
                           
        self.d = []
        self.d1 = []
        self.a = []
        self.a1 = []

        if verbose:
            print("Calculating A @ "+str(time.process_time()))
            print("Processing Clusters")
        for i in range(nclusts):
            d = []
            a = []
            a1 = []
            for j in range(nclusts):
                d.append(0)
                a.append(0)
                a1.append(0)
            self.d.append(d)
            self.a.append(a) 
            self.a1.append(a1)           
        
        self.dext = []
        self.a2 = []
        
        fns = os.popen("ls "+ifNameFormat).read().split("\n")
        n = len(fns)
        trj_fn = []
        for i in range(n):
            if len(fns[i].strip())==0:
                continue       
            trj_fn.append(fns[i].strip())

        if self.data.op_name in ["RMSD","rmsd"]:
        
            for i in range(nclusts):
                clust_fn = trj_fn[i] #digfmt1.format(i)+".dcd"
                clust1 = Traj(clust_fn, native_fn, stride=stride)
          
                M.traj = clust1     
                M.call(self.data.native, flag="calc")
             
                if verbose:
                    print("Cluster # :"+str(i)+"\tQ shape:"+str(len(M.q))+"\tCluster size:"+str(clust1.nl), end='\r')
                                
                for j in range(i,nclusts):               
                    clust_fn = digfmt1.format(j)+".dcd"
                    clust2 = Traj(clust_fn, native_fn, stride=stride)
                    d = M.RMSD(clust2.traj, flag="clust")              
                    self.d1.extend(d[0])
                    self.d[i][j] = copy.deepcopy(d[1])
                    self.d[j][i] = copy.deepcopy(d[1])
                    
                
        else:  
            for i in range(nclusts):
                clust_fn = trj_fn[i] #clust_fn = digfmt1.format(i)+".dcd"
                clust1 = Traj(clust_fn, native_fn, stride=stride)
                if verbose:
                    print("Cluster # :"+str(i)+"\tQ shape:"+str(len(M.q))+"\tCluster size:"+str(clust1.nl), end='\r')

                M.traj = clust1     
                M.call(self.data.native, flag="calc")   

            for i in range(nclusts):             
                if verbose:
                    print("Cluster # :"+str(i)+"\tQ shape:"+str(len(M.q))+"\tCluster size:"+str(clust1.nl), end='\r')

                for j in range(i,nclusts):  
                                               
                    d = self.getDMat(i+1, j+1, flag="clust")                     
                    self.d[i][j] = copy.deepcopy(d)
                    self.d[j][i] = copy.deepcopy(d)
                            
        

        if wFlag:
            print2file(fn+"-clust",fx,self.d,fflag='full',writeflag='w')
            
        self.data.eps = self.toCalcEpsilon() 

        if verbose:
            print("Epsilon : "+str(self.data.eps))

        for i in range(nclusts):
            for j in range(i,nclusts):   
                a = self.data.kernel(np.array(self.d[i][j]))
                
                self.a[i][j] = np.mean(a) 
                self.a[j][i] = np.mean(a) 
                
                self.a1[i][j] = np.mean(a,axis=-1)
                self.a1[j][i] = np.mean(a,axis=0)
                
        for i in range(nclusts):            
            a = np.concatenate((np.array(self.a1[i])), axis=0)  
            self.a2.append(np.array(a))
        self.a2 = np.array(self.a2) 
        self.a = np.array(self.a)


        ai = self.calcAi(self.a)
        aj = self.calcAi(self.a.transpose())
                
        if verbose:
            print("Calculating Po @ "+str(time.process_time()))
        self.P = self.calcP(self.a, ai, aj, flag=flag)
        #del clust1, clust2
        
        if verbose:
            print("Calculating eVeco @ "+str(time.process_time()))
        self.eVal, eVec = self.calcEV(self.P)
        self.eVec = eVec.transpose()
        if verbose:
            print("Calculated eVeco @ "+str(time.process_time()))
            print("eVeco shape: "+str(self.eVec.shape))        
               
        
        d = np.delete(d, np.s_[::])
        ai = np.delete(ai, np.s_[::])
        aj = np.delete(aj, np.s_[::])  

        return self.eVal, self.eVec, M
        
    ######################################################################  
    
    
    ######################################################################  
    #calling functions to calc DMAP for clusters/CG trajectories using Nystrom Etension
    def callCGNE(self, native_fn, digfmt1, M, nclusts=[1], fn="t", fx="RMSD", stride=1, wFlag=False, verbose=True):
                
        #M = Measures(self.data, "")

        flag = 'calc'
        if verbose:
            print("Calculating An @ "+str(time.process_time()))

        if isinstance(nclusts,int):
            nclusts = [nclusts]
        nclusts = np.unique(nclusts)
        M.call(nclusts, flag="NE")                
        
        extstruct = []
        for i in nclusts:
            extstruct.extend(np.where([M.nc==i])[1]-len(nclusts))

        self.a2 = self.a2[:,extstruct]
       
        
        self.a = np.insert(self.a, self.a.shape[0], self.a2.transpose(), axis=-1)

        ai = self.calcAi(self.a)
        aj = self.calcAi(self.a.transpose())
 
        if verbose:
            print("Calculating Pn @ "+str(time.process_time()))
        Pn = self.calcP(self.a, ai, aj, flag=flag).transpose()      
        
        if verbose:
            print("Calculating eVecn using Nystrom Extension @ "+str(time.process_time()))
        eVec = self.nystromE(Pn, self.eVec.transpose(), self.eVal)
        eVec = eVec.transpose() 
        if verbose:
            print("Calculated eVecn using Nystrom Extension @ "+str(time.process_time()))
            print("eVecn shape: "+str(eVec.shape))
                

        self.dext = np.delete(self.dext, np.s_[::])
        self.d = np.delete(self.d, np.s_[::])
        self.a = np.delete(self.a, np.s_[::])
        ai = np.delete(ai, np.s_[::])
        aj = np.delete(aj, np.s_[::])
        Pn = np.delete(Pn, np.s_[::])    
        
        return eVec, M
                
    ######################################################################  

    
    ######################################################################  
    #Extracting significant Evecs
    def selDMapExtremes(self, DMap, c=False):
        n = len(DMap)
        m = self.eVal.shape[0]
        all_pts = []
        sel_pts1 = []
        
        """
        if self.eVec.shape[0]!=self.eVec.shape[-1]:        
            #DMap = DMap[:,m:]
            #c = c[m:]
            NEFlag = True
        """        
        
        if n<2 and c:
            for i in range(n):
                t = alphaShape2(e1=DMap[i], e2=c, c=c, plotFlag=False)
                                                
                t = t.tolist()
                if len(sel_pts1)>0:
                    sel_pts1 = np.intersect1d(t, sel_pts1).tolist()
                else:
                    sel_pts1 = t
                all_pts.extend(t)
            
            if len(sel_pts1)>0:
                all_pts = sel_pts1
            
            all_pts = np.unique(all_pts).tolist()
                
            col = np.array(['b']*len(DMap[0]))
            col[all_pts] = 'r'
            all_pts = [[i] for i in all_pts]
            return all_pts, col

        
        sel_pts2 = []
        figure(figsize=(5*n,4)) 
        for i in range(1,n):
            j = 100+(n*10)+(i+1)                
            subplot(j)                         
            t = alphaShape2(e1=DMap[i], e2=DMap[i-1], c=c, plotFlag=True)   
            #t = alphaShape2(e1=DMap[i-1], e2=DMap[i], c=c, plotFlag=True)
            #scatter(DMap[i], DMap[i-1], c=c)
            
            t = t.tolist()
            if len(sel_pts2)>0:
                sel_pts2 = np.intersect1d(t, sel_pts2).tolist()
            else:
                sel_pts2 = t
            all_pts.extend(t)
        #colorbar()
        #show()
        
        all_pts = np.unique(all_pts).tolist()
        try:
            sel_pts1.extend(sel_pts2.tolist())
        except:
            sel_pts1.extend(sel_pts2)
        sel_pts1 = np.unique(list(flattenList(sel_pts1))).tolist()
        
        if len(sel_pts1)>0:
            all_pts = sel_pts1
        col = np.array(['b']*len(DMap[0]))
        col[all_pts] = 'r'
        
        all_pts = [[i] for i in all_pts] 
        
        #scatter(DMap[1], DMap[0], c=c)
        #scatter(DMap[2], DMap[1], c=c)
        #show()
        
        return all_pts, col
    ###################################################################### 
    
    
    ######################################################################  
    #Extracting significant Evecs
    def extractDMap(self):

        # Assign top eVals and eVecos
        sorted_eVal = sorted(self.eVal, reverse=True)
        
                
        # Plot eVals     
        figure(figsize=(8,5))
        subplot(121)
        scatter(range(len(sorted_eVal)),sorted_eVal)
        xlabel("N")
        ylabel("eVal")
        if len(sorted_eVal)>10:
            subplot(122)
            scatter(range(10),sorted_eVal[1:11])
            xlabel("N")
            ylabel("eVal")
        show()
        
        d = [sorted_eVal[i]-sorted_eVal[i+1] for i in range(1,6)]
        n = np.where(d==max(d))[0][0]+1
        n = 3
        print("Dimention of DMAP: "+str(n))

        
        # Get relevant/significant eVecs
        eVal = np.empty((n), dtype=int)
        m = self.eVal.shape[0]
        eVec = np.empty((n,self.eVec.shape[-1]))

        
        for i in range(n):
            eVal[i] = np.where(np.array(self.eVal) == sorted(self.eVal, reverse=True)[i+1])[0]
            eVec[i] = self.eVec[eVal[i]]
  
        return eVec
    ######################################################################  

    
    ######################################################################  
    #Plotting Eigen Vectors
    def plot(self, eVec, c={}, cpairs=[]):  
        nc = len(cpairs)        
        if nc==0:
            c = {'c':'c'}
        ckeys = c.keys()
        
        m = self.eVal.shape[0]
        
        n = len(eVec)
        
        #Plot eVecos w/ frame index
        print("Plot eVecos w/ frame index")
        for ck in ckeys:              
            figure(figsize=(5*n,4))
            print("Color:"+ck)
            for i in range(n):
                j = 100+(n*10)+(i+1)
                subplot(j)
                scatter(range(len(eVec[i])),eVec[i], c=c[ck])
                #if nc>0:
                try:
                    colorbar()
                except:
                    pass
                xlabel("N")
                ylabel("eVec"+str(i+1))
            show()
        
        #Plot eVecos w/ c       
        if nc>0:
            print("Plot eVecos w/ c")
            for cp in cpairs: 
                figure(figsize=(5*n,4))
                for i in range(n):
                    j = 100+(n*10)+(i+1)
                    subplot(j)
                    
                    if i==0:
                        print("Color: "+cp[0]) 
                    scatter(eVec[i], c[cp[1]], c=c[cp[0]])
                    colorbar()
                                                                             
                    xlabel("eVec"+str(i+1))
                    ylabel(cp[1])                    
                show()
                
                figure(figsize=(5*n,4))                
                for i in range(n):
                    j = 100+(n*10)+(i+1)
                    subplot(j)
                    
                    if i==0:
                        print("Color: "+cp[1]) 
                    scatter(eVec[i], c[cp[0]], c=c[cp[1]])
                    colorbar()
                                                                             
                    xlabel("eVec"+str(i+1))
                    ylabel(cp[0])                    
                show()

        
        if n<2:
            return
        #Plot 2 eVecos w/ each other
        #figure(figsize=(4,4*n))
        print("Plot 2 eVecos w/ each other")
        for ck in ckeys:  
            figure(figsize=(5*n,4))
            print("Color:"+ck)
            for i in range(1,n):
                #j = (n*100)+10+(i+1)
                j = 100+(n*10)+(i+1)                
                subplot(j)  
                
                scatter(eVec[i],eVec[i-1], c=c[ck])
                #if nc>0:
                try:
                    colorbar()    
                except:
                    pass
                xlabel("eVec"+str(i))
                ylabel("eVec"+str(i-1))
                
                """
                if self.eVec.shape[0]==self.eVec.shape[-1]:
                    if 'q' in ckeys:
                        sel_pts = alphaShape2(e1=eVec[i], e2=eVec[i-1], c=c['q'], plotFlag=True)                    
                    else:
                        alphaShape2(eVec[i], eVec[i-1], c=False, plotFlag=True)
                else:
                    if 'q' in ckeys:
                        sel_pts = alphaShape2(e1=eVec[i][m:], e2=eVec[i-1][m:], c=c['q'], plotFlag=True)# + m                   
                    else:
                        alphaShape2(eVec[i][m:], eVec[i-1][m:], c=False, plotFlag=True)
                """
            show()
        
        """
        if n<3:
            return
        #Plot 3 eVecos w/ each other
        print("Plot 3 eVecos w/ each other")
        from mpl_toolkits import mplot3d        
        for ck in ckeys:
            #figure(figsize=(5*n,8))
            fig = figure(figsize=(30, 5*n)) #plt.figure()
            for i in range(n-2):
                j = (n*100)+(30)+((3*i)+1)
                ax = fig.add_subplot(j, projection='3d')
                m = ax.scatter3D(eVec[i], eVec[i+1], eVec[i+2], c=c[ck])   
                fig.colorbar(m)
             
                j = (n*100)+(30)+((3*i)+2)
                ax = fig.add_subplot(j, projection='3d')
                m = ax.scatter3D(eVec[i+2], eVec[i], eVec[i+1], c=c[ck])   
                fig.colorbar(m)
            
                j = (n*100)+(30)+((3*i)+3)
                print(j)
                ax = fig.add_subplot(j, projection='3d')
                m = ax.scatter3D(eVec[i+1], eVec[i+2], eVec[i], c=c[ck])   
                fig.colorbar(m)
        
         
            show()
        """
        
    ######################################################################  
    
    
    ######################################################################  
    # Destructor
    def delete(self):
        self.a = np.delete(self.a, np.s_[::])
        self.P = np.delete(self.Po, np.s_[::])
    ######################################################################  


    
######################################################################    
#Misc Functions
######################################################################    


######################################################################    
def alphaShape(points, alpha, only_outer=True):
    from scipy.spatial import Delaunay

    """
    Compute the alpha shape (concave hull) of a set of points.
    :param points: np.array of shape (n,2) points.
    :param alpha: alpha value.
    :param only_outer: boolean value to specify if we keep only the outer border
    or also inner edges.
    :return: set of (i,j) pairs representing edges of the alpha-shape. (i,j) are
    the indices in the points array.
    """
    assert points.shape[0] > 3, "Need at least four points"

    def addEdge(edges, i, j):
        """
        Add a line between the i-th and j-th points,
        if not in the list already
        """
        if (i, j) in edges or (j, i) in edges:
            # already added
            assert (j, i) in edges, "Can't go twice over same directed edge right?"
            if only_outer:
                # if both neighboring triangles are in shape, it's not a boundary edge
                edges.remove((j, i))
            return
        edges.add((i, j))

    tri = Delaunay(points)
    edges = set()
    # Loop over triangles:
    # ia, ib, ic = indices of corner points of the triangle
    for ia, ib, ic in tri.vertices:
        pa = points[ia]
        pb = points[ib]
        pc = points[ic]
        # Computing radius of triangle circumcircle
        # www.mathalino.com/reviewer/derivation-of-formulas/derivation-of-formula-for-radius-of-circumcircle
        a = np.sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
        b = np.sqrt((pb[0] - pc[0]) ** 2 + (pb[1] - pc[1]) ** 2)
        c = np.sqrt((pc[0] - pa[0]) ** 2 + (pc[1] - pa[1]) ** 2)
        s = (a + b + c) / 2.0
        area = np.sqrt(s * (s - a) * (s - b) * (s - c))
        circum_r = a * b * c / (4.0 * area)
        if circum_r < alpha:
            addEdge(edges, ia, ib)
            addEdge(edges, ib, ic)
            addEdge(edges, ic, ia)
    return edges  
######################################################################    


######################################################################    
def alphaShape2(e1,e2,c, plotFlag = False):    
    points = np.vstack([e1, e2]).T
     
    edges = alphaShape(points, alpha=5.0, only_outer=True)
    figure(figsize=(5,4))
    if plotFlag:
        for i, j in edges:
            plot(points[[i, j], 0], points[[i, j], 1], c='b')


    from scipy import stats
    kde = stats.gaussian_kde(points.T)
    kde = np.reshape(kde(points.T).T, points.T.shape[-1])
    
    sel_edges = []
    edges1 = np.unique(list(flattenList(list(edges))))
    
    n = edges1.shape[0]

    
    cmin = np.where(np.array(c)==min(c))[0][0]  
    kdemax = np.where(np.array(kde)==max(kde))[0][0]  
    e1cmin = e1[cmin]
    e2cmin = e2[cmin]
    e1kdemax = e1[kdemax]
    e2kdemax = e2[kdemax]
    cmin = min(c)
    kdemax = max(kde)

    DfromC = np.sqrt( (np.array(e1)[edges1]-e1cmin)**2 + (np.array(e2)[edges1]-e2cmin)**2 )
    DfromKDE = np.sqrt( (np.array(e1)[edges1]-e1kdemax)**2 + (np.array(e2)[edges1]-e2kdemax)**2 )
    lim1 = sorted(DfromC)[n//2]
    
    lim2 = sorted(DfromKDE)[n//2]
    for i in range(len(edges1)):
        if DfromC[i]<lim1: #DfromKDE[i]<lim2: #
            sel_edges.append(edges1[i])
    
    
    edges1 = copy.deepcopy(sel_edges)
    sel_edges = []
    
    csort = np.argsort(np.array(c)[edges1])[:len(edges1)//2]
    for i in csort:
        sel_edges.append(edges1[i])
    
    #print(np.where(e2==sorted(e2)[1]), sorted(e2)[1], np.where(e1==sorted(e1)[-2]), sorted(e1)[-2])
    if plotFlag:
        for i, j in edges:
            if (i in sel_edges) or (j in sel_edges):
                plot(points[[i, j], 0], points[[i, j], 1], c='r')
    
    kdeorder = np.argsort(np.array(kde))
    e1 = e1[kdeorder]
    e2 = e2[kdeorder]
    kde = kde[kdeorder]
    
    scatter(e1,e2,c=c) 
    colorbar()            
    show() 
        
    return np.array(edges)
    
######################################################################         


