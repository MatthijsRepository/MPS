import numpy as np
from Operators_and_Hamiltonian import Create_Sz, Create_Sx, Create_Sy
from Expectations_and_Dotproducts import expectation,Singlesite_Expectation, Singlesite_Expectation_Chain



"""
Contains the time evolution process and the two-site operator
"""


def Two_site_Operator(i,gammas,lambdas,O_arr,d,chi,loc_size,normalize):
    """The result of a two-site operator is computed with the process explained in the Bachelor thesis. Then a SVD is computed wherafter a truncation process is done and the objects are reshaped in the input form"""
    theta = np.tensordot(np.diag(lambdas[i,:]), gammas[i,:,:,:], axes=(1,0)) 
    theta = np.tensordot(theta,np.diag(lambdas[i+1,:]),axes=(1,0))
    theta = np.tensordot(theta, gammas[i+1,:,:,:],axes=(2,0))
    theta = np.tensordot(theta,np.diag(lambdas[i+2,:]), axes=(2,0))     
    theta_prime = np.tensordot(theta,O_arr[i,:,:,:,:],axes=([1,2],[2,3]))               # Two-site operator
    theta_prime = np.reshape(np.transpose(theta_prime, (2,0,3,1)),(d*chi,d*chi)) # danger!
    #Singular value decomposition
    X, Y, Z = np.linalg.svd(theta_prime); Z = Z.T
    #truncation
    if normalize:
        lambdas[i+1,:] = Y[:chi]*1/np.linalg.norm(Y[:chi])
    else:
        lambdas[i+1,:] = Y[:chi]
    X = np.reshape(X[:d*chi,:chi], (d, chi,chi))  # danger!
    inv_lambdas = lambdas[i,:loc_size[i]]**(-1)
    inv_lambdas[np.isnan(inv_lambdas)]=0
    tmp_gamma = np.tensordot(np.diag(inv_lambdas),X[:,:loc_size[i],:loc_size[i+1]],axes=(1,1))
    gammas[i,:loc_size[i],:loc_size[i+1],:] = np.transpose(tmp_gamma,(0,2,1))
    Z = np.reshape(Z[0:d*chi,:chi],(d,chi,chi))
    Z = np.transpose(Z,(0,2,1))
    inv_lambdas = lambdas[i+2,:loc_size[i+2]]**(-1)
    inv_lambdas[np.isnan(inv_lambdas)]=0
    tmp_gamma = np.tensordot(Z[:,:loc_size[i+1],:loc_size[i+2]], np.diag(inv_lambdas), axes=(2,0))
    gammas[i+1,:loc_size[i+1],:loc_size[i+2],:] = np.transpose(tmp_gamma,(1, 2, 0))    
    return gammas,lambdas


def Time_evolution(gammas,lambdas,T,O_arr_ev,L,d,chi,loc_size, Ham_meas, s ,Nin, Meas, normalize):
    """The Odd-Even time evoltuon is done here, after each time step the Ground state energy is computed and printed"""
    #time = [];Energy=np.zeros((T,len(Nin)-(len(Nin)-1)*Meas),dtype=complex)
    time=[]
    if Meas==0:
        Energy=np.zeros((T,len(Nin)), dtype=complex)
    else:
        Energy=np.zeros((T,1), dtype=complex)
            
    dsmall=int(2*s+1)
    Sz=np.kron(Create_Sz(s), np.eye(dsmall, dtype=complex))
    for t in range(T):
        #Time evolution
        for i in range(1,L-1,2):            # Odd bonds
            gammas, lambdas = Two_site_Operator(i,gammas,lambdas,O_arr_ev,d,chi,loc_size, normalize)
        for i in range(0,L-1,2):            # Even bonds
            gammas, lambdas = Two_site_Operator(i,gammas,lambdas,O_arr_ev,d,chi,loc_size, normalize)

        time.append(t)

        if Meas==0:         #Calculation of <S_z>
            for i,n in enumerate(Nin): 
                E = Singlesite_Expectation(n,gammas,lambdas,Sz) 
                Energy[t,i]=E
        elif Meas==2:       #Calculation of <S_z> for the entire chain
            E = Singlesite_Expectation_Chain(gammas, lambdas, Sz, L)
            Energy[t,0]=E
        else:               #Calculation of chain energy
            #Note that the measurement Hamiltonian may be different than the time evolution Hamiltonian. In the case that the ancilla's evolve backwards application of Hamiltonian will always yield zero
            E = expectation(gammas,lambdas,Ham_meas,L)
            Energy[t,0]=E
        if np.isnan(Energy[t,0]):                   # Break before the code crashes
            return time,Energy
    return np.array(time),Energy
   
  


"""

#Higher order Suzuki-Trotter decompositions
#Not used during this research

def Time_evolutionST2(gammas,lambdas,T,O_arr1,O_arr2,L,d,chi,loc_size, Ham,Nin, Meas, normalize=False):
    #The Odd-Even time evoltuon is done here, after each time step the Ground state energy is computed and printed
    time = [];Energy=np.zeros((T,len(Nin)-(len(Nin)-1)*Meas),dtype=complex)
    
    print("%5s,%15s" % ("Time","Value"))
    Sz=Create_Sz(s=2)
    for t in range(T):
        for i in range(0,L-1,2):            # Even bonds
            gammas, lambdas = Two_site_Operator(i,gammas,lambdas,O_arr2,d,chi,loc_size, normalize)
        for i in range(1,L-1,2):   # Odd bonds
            gammas, lambdas = Two_site_Operator(i,gammas,lambdas,O_arr1,d,chi,loc_size, normalize)
        for i in range(0,L-1,2):            # Even bonds
            gammas, lambdas = Two_site_Operator(i,gammas,lambdas,O_arr2,d,chi,loc_size, normalize)
        time.append(t)
        
        if Meas==0:
            for i,n in enumerate(Nin):
                E = Singlesite_Expectation(n,gammas,lambdas,Sz)
                
                Energy[t,i]=E
        else:
            E = expectation(gammas,lambdas,Ham,L)
            Energy[t,0]=E
        if np.isnan(Energy[t,0]):                   # Break before the code crashes
            return time,Energy
        print("%5d,%15.10f" % (t,np.real(Energy[t,0])))
        
    return np.array(time),Energy
   
    
    
def Time_evolutionST4(gammas,lambdas,T,O_arr1,O_arr2,O_arr3,O_arr4,L,d,chi,loc_size, Ham,Nin, Meas, normalize=False):
    #The Odd-Even time evoltuon is done here, after each time step the Ground state energy is computed and printed
    time = [];Energy=np.zeros((T,len(Nin)),dtype=complex)
   
    print("%5s,%15s" % ("Time","Value"))
    Sz=Create_Sz(s=2)
    for t in range(T):
        for i in range(0,L-1,2):            # Even bonds
            gammas, lambdas = Two_site_Operator(i,gammas,lambdas,O_arr1,d,chi,loc_size, normalize)
        for i in range(1,L-1,2):            # Odd bonds
            gammas, lambdas = Two_site_Operator(i,gammas,lambdas,O_arr2,d,chi,loc_size, normalize)
        for i in range(0,L-1,2):            # Even bonds
            gammas, lambdas = Two_site_Operator(i,gammas,lambdas,O_arr2,d,chi,loc_size, normalize)
        for i in range(1,L-1,2):            # Odd bonds
            gammas, lambdas = Two_site_Operator(i,gammas,lambdas,O_arr2,d,chi,loc_size, normalize)
        for i in range(0,L-1,2):            # Even bonds
            gammas, lambdas = Two_site_Operator(i,gammas,lambdas,O_arr3,d,chi,loc_size, normalize)
        for i in range(1,L-1,2):            # Odd bonds
            gammas, lambdas = Two_site_Operator(i,gammas,lambdas,O_arr4,d,chi,loc_size, normalize)
        for i in range(0,L-1,2):            # Even bonds
            gammas, lambdas = Two_site_Operator(i,gammas,lambdas,O_arr3,d,chi,loc_size, normalize)
        for i in range(1,L-1,2):            # Odd bonds
            gammas, lambdas = Two_site_Operator(i,gammas,lambdas,O_arr2,d,chi,loc_size, normalize)
        for i in range(0,L-1,2):            # Even bonds
            gammas, lambdas = Two_site_Operator(i,gammas,lambdas,O_arr2,d,chi,loc_size, normalize)
        for i in range(1,L-1,2):            # Odd bonds
            gammas, lambdas = Two_site_Operator(i,gammas,lambdas,O_arr2,d,chi,loc_size, normalize)
        for i in range(0,L-1,2):            # Even bonds
            gammas, lambdas = Two_site_Operator(i,gammas,lambdas,O_arr1,d,chi,loc_size, normalize)
        
        time.append(t)
        
        if Meas==0:
            for i,n in enumerate(Nin):
                E = Singlesite_Expectation(n,gammas,lambdas,Sz)
                
                Energy[t,i]=E
        else:
            E = expectation(gammas,lambdas,Ham,L)
            Energy[t,0]=E
        if np.isnan(Energy[t,0]):                   # Break before the code crashes
            return time,Energy
        print("%5d,%15.10f" % (t,np.real(Energy[t,0])))
        
    return np.array(time),Energy


"""
