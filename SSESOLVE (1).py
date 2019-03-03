
# coding: utf-8

# In[68]:


import time
import argparse
import numpy as np
from qutip import *
from math import sqrt
from scipy import *
from scipy.special import factorial
import matplotlib.pyplot as plt
import pandas as pd

#parser = argparse.ArgumentParser(description='Adiabatic Quantum Computing using Spin Ensembles', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#parser.add_argument('--decoherence', action='store_true', help='Enables decoherence')
#pargs = parser.parse_args()

#print('decoherence: ' + str(pargs.decoherence))
##*******************************      set up the parapeteres 


count_time=0

result=[]
for taumax in [100]:
    count_time=count_time+1
    print('taulist',taumax)
    taulist= np.linspace(0,taumax,20)
    num_trial=1                    #number of instances
    p1_final=[]
    q=2                            #Ensemble size
    M=3                            #Problem size 
    for N in range(1,q):
        Mp=(N+1)**M
        probability=[]
        print('N=',N)
        a=destroy(N+1)
    #********************************** Construct the  Initial State
        g1=np.zeros((N+1,1),float)
        for i in range(0,N+1):
            g1[i]=((1./np.sqrt(2))**i)*((1./np.sqrt(2))**(N-i))*sqrt(factorial(N)/((factorial(i))*(factorial(N-i))))
        psi_list=[Qobj(g1).unit() for n in range(M)]
        psii=tensor(psi_list)

    #*********************************  Construct the Hamiltonian
        I1=qeye(N+1)
        z=(2*( a.dag()*a)-(N)*I1)
        x=np.zeros((N+1,N+1), complex)
        for i in range (0,N):
            x[i+1][i]=np.sqrt(N-i)*np.sqrt(i+1)
            x[i][i+1]=np.sqrt(N-i)*np.sqrt(i+1)
        y=np.zeros((N+1,N+1),complex)
        for i in range (0,N):
            y[i+1][i]=-1j*np.sqrt(N-i)*np.sqrt(i+1)
            y[i][i+1]=1j*np.sqrt(N-i)*np.sqrt(i+1)
        sx = Qobj(x)
        sy = Qobj(y)
        sz = Qobj(z)
        sx_list = []
        sy_list = []
        sz_list = []

        for n in range(M):
            op_list = []
            for m in range(M):
                op_list.append(I1)
            #print(op_list)    
            op_list[n] = sx
            sx_list.append(tensor(op_list))

            op_list[n] = sy
            sy_list.append(tensor(op_list))

            op_list[n] = sz
            sz_list.append(tensor(op_list))

        H0=0
        for n in range(M):
            H0 += -sx_list[n]

        count=0
        np.random.seed(10)
        number=[]
        n_ground=[]
        success_eachtrial=[]
        regime2=[]
      #*********************************************************        
        for i in range(1,num_trial+1):
            count=count+1
            #print(count)  
            Jmat=np.random.uniform(low=-1, high=1, size=(M,M) )
            Jmat=Jmat+Jmat.transpose()
            for i in range(0,len(Jmat)):
                Jmat[i,i]=0

            Kmat=np.random.uniform(low=-1, high=1, size=(1,M) )
            Jmat[1,0]= 0.5#-1.64333884      
            Jmat[0,1]=0.5# -1.64333884
            Jmat[2,0]=0#0.4581942
            Jmat[0,2]=0#0.45819421
            Jmat[1,2]=0.5#0.36196838
            Jmat[2,1]=0.5#0.36196838

            Kmat[0,0]=-1#-0.73236837 
            Kmat[0,1]=-1# 0.35603385
            Kmat[0,2]=-1# 0.70586361
            H1 = 0
            for n in range(M):
                H1 += Kmat[0,n]*sz_list[n]
            for n1 in range(M) :
                for n2 in range(M) :
                    H1 +=(1./N)* (Jmat[n1, n2]*sz_list[n1]*sz_list[n2])
             #****************************** Hamiltonian for exact cover instances

            args = {'t_max': max(taulist)}
            h_t = [[H0, lambda t, args :( 1-(t/args['t_max']))],[H1, lambda t, args : (t/args['t_max'])]]

            #print(h_t) 

      #*********************************************************        


            evals1, ekets1=H1.eigenstates()  
            ex=[]
            for m in range(M):
                #print('m',m)
                ex.append([])
                for n in range(Mp): # for Calculating the number of Replica states
                    ex[m].append(expect(sz_list[m],ekets1[n]))


            args = {'t_max': max(taulist)}
            h_t = H = [[H0, lambda t, args :( 1-(t/args['t_max']))],[H1, lambda t, args : (t/args['t_max'])]]


    #***************************************** set parameters
            evals_mat = np.zeros((len(taulist),Mp))
            P_n=np.zeros((len(taulist),Mp))
            P_mat = np.zeros((len(taulist),Mp))
            success_probability=np.zeros((len(taulist),Mp))
            #success_p=np.zeros((1,Mp))
            c_ops = [np.sqrt(0.0001)*sz_list[0],np.sqrt(0.0001)*sz_list[1],np.sqrt(0.0001)*sz_list[2]]
            e_ops=[]
            options = Options( average_states=False)
            R=ssesolve(h_t, psii, taulist, c_ops,[],method='homodyne' ,args= {'t_max': max(taulist)}, options=options)
            print(R.states[5])

            
            
            for i in range(100):
                tau=taulist[i]
    #******************************* evaluate the Hamiltonian with gradually switched on interaction 
                H = qobj_list_evaluate(h_t, tau, args)

    #******************************   eigenvalues of the Hamiltonian
                evals, ekets = H.eigenstates(eigvals=Mp)
                evals_mat[i,:] = real(evals)
    #***************************************  occupation probabilities of the energy levels without decoherence
               
                for n, eket in enumerate(ekets):
                      P_mat[i,n] =(abs((ekets[n].dag() * R.states[i]*ekets[n])[0,0]))


    #***********************************  finding the number of replica states
            num=[]
            for m in range(M):        #using the output of the adiabatic evolution
                num.append([])
                for n in range(Mp):
                    if  ex[m][n]* ex[m][0]<=0:
                        num[m].append(n)
            for i in range(M):
                number.append(num[i][0])
            ng=min(number[:])        #number of replica states

            
            success_probability=(sum(P_mat[len(taulist)-1,n] for n in range(0,ng)))
            success_eachtrial.append(success_probability)

        print('success_eachtrial',success_eachtrial)
        print('success_eachtrial',size(success_eachtrial))
        success_alltrials=sum(success_eachtrial[i] for i in range(1))
        print('success_alltrials',success_alltrials)

        p1_final.append(success_alltrials/1)

        print('p_final',p1_final)

     #********************************   save success probability
    result.append(('tau%d'%taumax, (p1_final)))
df_in = pd.DataFrame.from_items(result)
df_in.to_csv('successprobability.dat', index=False, sep='\t')





