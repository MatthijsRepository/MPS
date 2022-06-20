import numpy as np
import scipy.constants as sc


"""
Contains code related to the creation of the operators and the hamiltonian
"""


def two_site_Justin(H,delta,d): 
    """
    Returns the operator in the correct shape.
    """
    Hop=Operator(H,delta)
    return np.reshape(Hop,(d,d,d,d))    

    
def Operator(H,dt): 
    """
    The Crank-Nicolson operator. 
    """    
    H_top=np.eye(H.shape[0])-1j*dt*H/2
    H_bot=np.eye(H.shape[0])+1j*dt*H/2
    return np.linalg.inv(H_bot).dot(H_top)

def Raising_Op(s, ms):
    """
    The prefactor that results from using the S+ operator on a given |s,ms> state
    """
    return np.sqrt(s*(s+1)-ms*(ms+1))
    
def Create_Sp(s):
    """Creates the Splus matrix"""
    d = int(2*s+1)
    Sp = np.zeros((d,d), dtype=complex)
    for i in range(d-1):
        Sp[i,i+1]=Raising_Op(s,-s+i)
    return Sp
    
def Create_Sm(s):
    """Creates the Sminus matrix"""
    d = int(2*s+1)
    Sm = np.zeros((d,d), dtype=complex)
    for i in range(d-1):
        Sm[i+1,i]=Raising_Op(s,-s+i)
    return Sm

def Create_Sz(s, diag = False):
    """
    Creates the Sz matrix for a spin-s system.
    diag=True: Returns diagonal. diag=False: Returns diagonal matrix.
    """
    d = int(2*s+1) 
    Sz_Diag=np.linspace(s,-s,d,dtype=complex)
    if diag:
        return Sz_Diag
    return np.diag(Sz_Diag)

def Create_Sx(s): 
    """
    Creates the Sx matrix for a spin-s system. For s=1/2 we get our spin-1/2 pauli matrices
    """
    Sx=0.5*(Create_Sp(s)+Create_Sm(s))
    return Sx

def Create_Sy(s):
    Sy=0.5j*(Create_Sm(s)-Create_Sp(s)) 
    return Sy


    

def Create_Opparray(H_arr, L, d, delta):
    """Creates a time operator array using the given Hamiltonian."""
    O_arr = np.zeros((L-1,d,d,d,d), dtype=complex)
    for i in range(0,L-1):
        O_arr[i,:,:,:,:] = two_site_Justin(np.reshape(H_arr[i,:,:,:,:],(d**2,d**2)),delta,d)       
    return O_arr
    


def Ham_Experiment_Extended(L, J, h, s, ancilla_H):
    """
    Creates the Hamiltonian used during the simulations
    """
    d = int(2*s+1)
    dnew = int(d**2)
    
    #Creating Pauli matrices
    Sx = Create_Sx(s) *2
    Sy = Create_Sy(s) *2
    Sz = Create_Sz(s, diag = False) *2

    Id = np.eye(dnew, dtype=complex)
    
    #Creating physical-system spin matrices and ancillary-system spin matrices
    Sx_p = np.kron(Sx, np.eye(d))
    Sy_p = np.kron(Sy, np.eye(d))
    Sz_p = np.kron(Sz, np.eye(d))
    
    Sx_a = np.kron(np.eye(d), Sx)
    Sy_a = np.kron(np.eye(d), Sy)
    Sz_a = np.kron(np.eye(d), Sz)
    
    #Creating the operators that work on two particles
    Sz1_p = np.kron(Sz_p, Id)
    Sz2_p = np.kron(Id, Sz_p)
    Sz1_a = np.kron(Sz_a, Id)
    Sz2_a = np.kron(Id, Sz_a)
        
    H_X_p = np.kron(Sx_p, Sx_p)
    H_Y_p = np.kron(Sy_p, Sy_p)
    H_X_a = np.kron(Sx_a, Sx_a)
    H_Y_a = np.kron(Sy_a, Sy_a)
    
    #Creating z-component of Hamiltonian
    H_Z_middle_p = h * (Sz1_p/2 + Sz2_p/2)
    H_Z_left_p = h * (Sz1_p + Sz2_p/2)
    H_Z_right_p = h * (Sz1_p/2 + Sz2_p)
    
    H_Z_middle_a = h * (Sz1_a/2 + Sz2_a/2)
    H_Z_left_a = h * (Sz1_a + Sz2_a/2)
    H_Z_right_a = h * (Sz1_a/2 + Sz2_a)
    
    
    #Creating Hamiltonian
    H_XY_model_middle_p = J*(H_X_p + H_Y_p + H_Z_middle_p)
    H_XY_model_left_p = J*(H_X_p + H_Y_p + H_Z_left_p)
    H_XY_model_right_p = J*(H_X_p + H_Y_p + H_Z_right_p)
    
    H_XY_model_middle_a = J*(H_X_a + H_Y_a + H_Z_middle_a)
    H_XY_model_left_a = J*(H_X_a + H_Y_a + H_Z_left_a)
    H_XY_model_right_a = J*(H_X_a + H_Y_a + H_Z_right_a)
    
    
    
    H_XY_model_middle = H_XY_model_middle_p 
    H_XY_model_left = H_XY_model_left_p
    H_XY_model_right = H_XY_model_right_p
    
    if ancilla_H == True:
        H_XY_model_middle -= H_XY_model_middle_a
        H_XY_model_left -= H_XY_model_left_a
        H_XY_model_right -= H_XY_model_right_a
    
    H_arr = np.zeros((L-1, dnew,dnew,dnew,dnew), dtype=complex)
    
    for i in range(1,L-2):
        H_arr[i,:,:,:,:] = np.reshape(H_XY_model_middle, (dnew,dnew,dnew,dnew))
    H_arr[0,:,:,:,:] = np.reshape(H_XY_model_left, (dnew,dnew,dnew,dnew))
    H_arr[L-2,:,:,:,:] = np.reshape(H_XY_model_right, (dnew,dnew,dnew,dnew))
    
    return H_arr






def Single_Site_Operator(i,gammas,lambdas,O_arr):
    """Applies a single site operator. Used to flip a site in the ground state."""
    theta = np.tensordot(np.diag(lambdas[i,:]), gammas[i,:,:,:], axes=(1,0)) 
    theta = np.tensordot(theta,np.diag(lambdas[i+1,:]),axes=(1,0))
    theta_prime = np.tensordot(theta,O_arr[:,:],axes=(1,1))
    theta_prime = theta_prime / np.linalg.norm(theta_prime)
    inv_lambdas=lambdas[i]**(-1)
    inv_lambdas[np.isnan(inv_lambdas)]=0
    theta_prime=np.tensordot(np.diag(inv_lambdas), theta_prime, axes=(1,0))
    inv_lambdas=lambdas[i+1]**(-1)
    inv_lambdas[np.isnan(inv_lambdas)]=0
    theta_prime=np.tensordot(theta_prime, np.diag(inv_lambdas), axes=(1,0))
    theta_prime=np.transpose(theta_prime,(0,2,1))
    return theta_prime
    

def Excite_site(i,s,L,lambdas,gammas, start):
    """Creates an excitation at site i, applying a Splus or Smin operator to flip the state"""
    if i>=L:
        print("Given exited state is outside of the chain")
        return gammas
    if (i+start)%2==1: #If the site is in the "up" state
        gammas[i,:,:,:] = Single_Site_Operator(i,gammas,lambdas,Create_Sm(s))
    elif (i+start)%2==0: #If the site is in the "down" state
        gammas[i,:,:,:] = Single_Site_Operator(i,gammas,lambdas,Create_Sp(s))
    return gammas
    
 
def Prob_state(i,lambdas,gammas,state):
    """Gives the probability that site 'i' is in a given state."""
    theta = np.tensordot(np.diag(lambdas[i,:]), gammas[i,:,:,:], axes=(1,0)) 
    theta = np.tensordot(theta,np.diag(lambdas[i+1,:]),axes=(1,0))
 
    return np.linalg.norm(theta[:,state,:])**2
    
















"""
def Ham_new(L, delta, h, s, realtime):
    d = int(2*s+1)
    dnew = int(d**2)

    #Creating Pauli matrices
    Sx = Create_Sx(s) *2
    Sy = Create_Sy(s) *2
    Sz = Create_Sz(s, diag = False) *2

    Id = np.eye(d)

    Sz1 = np.kron(Sz, Id)
    Sz2 = np.kron(Id, Sz)
    
    H_X = np.kron(Sx, Sx)
    H_Y = np.kron(Sy, Sy)
    
    #H_X = np.eye(dnew**2)
    #H_Y = np.eye(dnew**2)
    
    H_Z_middle = h * (Sz1/2 + Sz2/2)
    H_Z_left = h * (Sz1 + Sz2/2)
    H_Z_right = h * (Sz1/2 + Sz2)
    
    H_XY_model_middle = H_X + H_Y + H_Z_middle
    H_XY_model_left = H_X + H_Y + H_Z_left
    H_XY_model_right = H_X + H_Y + H_Z_right
    
    
    if realtime==False:
        H_anc = np.eye(dnew)
        H_XY_model_middle = np.kron
    
    return
"""

    
    

def Ham_Experiment(L, delta, h, s):
    d = int(2*s+1)
    
    #NOTE THAT THE SPIN MATRICES HAVE A FACTOR 1/2 BEFORE THEM - PAULI MATRICES SIGMA DO NOT HAVE THESE!!!!!!!!
    Sx = Create_Sx(s)
    Sy = Create_Sy(s)
    Sz = Create_Sz(s, diag = False)
    
    
    #for symmetric split of single site operators as TEBD is done with two site operators
    Sx1 = np.kron(Sx, np.eye(d))
    Sy1 = np.kron(Sy, np.eye(d))
    Sz1 = np.kron(Sz, np.eye(d))
    Sx2 = np.kron(np.eye(d), Sx)
    Sy2 = np.kron(np.eye(d), Sy)
    Sz2 = np.kron(np.eye(d), Sz)
    
    
    H_arr = np.zeros((L-1,d,d,d,d), dtype=complex)
    #why this shape? Comes from code Pim Vree
    
    #H_Heis_Justin = np.kron(Sx, Sx) + np.kron(Sy, Sy) + np.kron(Sz, Sz)
    
    H_X = np.kron(Sx, Sx)
    H_Y = np.kron(Sy, Sy)
    
    H_Z_middle = h * (Sz1/2 + Sz2/2)
    H_Z_left = h * (Sz1 + Sz2/2)
    H_Z_right = h * (Sz1/2 + Sz2)
    
    H_XY_model_middle = H_X + H_Y + H_Z_middle
    H_XY_model_left = H_X + H_Y + H_Z_left
    H_XY_model_right = H_X + H_Y + H_Z_right
    
    #H_XY_model = H_X + H_Y + H_Z
    
    
    H_arr = np.zeros((L-1, d,d,d,d), dtype=complex)
    
    for i in range(1,L-2):
        H_arr[i,:,:,:,:] = np.reshape(H_XY_model_middle, (d,d,d,d))
    H_arr[0,:,:,:,:] = np.reshape(H_XY_model_left, (d,d,d,d))
    H_arr[L-2,:,:,:,:] = np.reshape(H_XY_model_right, (d,d,d,d))
    
    """
    #H_arr = np.reshape(H_Heis_Justin,(d,d,d,d))
    #H_arr = np.reshape(H_XY_model, (d,d,d,d))

    #print(H_arr)
    
    #H_arr_new = np.reshape(H_arr, (d**2,d**2))
    H_arr_new = np.zeros((L-1,d**2,d**2), dtype=complex)
    for i in range(L-1):
        H_arr_new[i,:,:] = np.reshape(H_arr[i], (d**2,d**2))
    
    O_arr = np.zeros((L-1, d,d,d,d), dtype=complex)
    for i in range(L-1):
        O_arr[i,:,:,:,:] = two_site_Justin(H_arr_new[i], delta, d)
    
    #print(O_arr[1])
    """
    return H_arr
    

#Ham_Experiment(5, 1e-2, 3, 1/2)


def Ham_Experiment_old(L,delta,s=2, Jglob=0.7, g=2.11, Bglob=np.array([0,0,1]), muB=10**3*sc.value("Bohr magneton in eV/T"), D=-1.77 ,E=0.33,Jprime=-0.05,LocB=0,LocJ=[0],Jloc=0):
    """
    Creates the hamiltonian that has been used in the experiment. The Hamiltonian is split into an odd and even part symmetrically.
    """
    d = int(2*s+1)
    B = np.full((L,3),Bglob,dtype=float)
    B_loc = np.array([0,0,-Jprime*s/(g*muB)])
    if 0<=LocB<L:
        B[LocB] += B_loc
    J = np.full(L-1,Jglob)
    for i in LocJ:
        if 0<=i<L-1:
            J[i] = Jloc
        
    Sx = Create_Sx(s)
    Sy = Create_Sy(s)
    Sz = Create_Sz(s, diag = False)
    Sx1 = np.kron(Sx, np.eye(d))
    Sy1 = np.kron(Sy, np.eye(d))
    Sz1 = np.kron(Sz, np.eye(d))
    Sx2 = np.kron(np.eye(d), Sx)
    Sy2 = np.kron(np.eye(d), Sy)
    Sz2 = np.kron(np.eye(d), Sz)
    
    H_arr = np.zeros((L-1,d,d,d,d), dtype=complex)
    
    #Anisotropic term ------------------- FUN FACT: EVERYTHING AFTER THE E*(...) IS ZERO
    H_anisleft = D*(Sz1.dot(Sz1) + Sz2.dot(Sz2)/2) + E*((Sx1.dot(Sx1) + Sx2.dot(Sx2)/2) - (Sy1.dot(Sy1) + Sy2.dot(Sy2)/2))
    H_anismid = D*(Sz1.dot(Sz1)/2 + Sz2.dot(Sz2)/2) + E*((Sx1.dot(Sx1)/2 + Sx2.dot(Sx2)/2) - (Sy1.dot(Sy1)/2 + Sy2.dot(Sy2)/2))
    H_anisright = D*(Sz1.dot(Sz1)/2 + Sz2.dot(Sz2)) + E*((Sx1.dot(Sx1)/2 + Sx2.dot(Sx2)) - (Sy1.dot(Sy1)/2 + Sy2.dot(Sy2)))
    
    #Middle of chain
    for i in range(1,L-2):
        H_Heis = J[i]*(np.kron(Sx, Sx) + np.kron(Sy, Sy) + np.kron(Sz, Sz))
        H_zeemanmid = -g*muB*((B[i,0]*Sx1/2 + B[i+1,0]*Sx2/2) + (B[i,1]*Sy1/2 + B[i+1,1]*Sy2/2) + (B[i,2]*Sz1/2 + B[i+1,2]*Sz2/2))
        H_mid = H_anismid + H_Heis + H_zeemanmid
        H_arr[i,:,:,:,:] = np.reshape(H_mid,(d,d,d,d))

    
    #Left edge of chain
    H_Heis = J[0]*(np.kron(Sx, Sx) + np.kron(Sy, Sy) + np.kron(Sz, Sz))
    H_zeemanleft = -g*muB*((B[0,0]*Sx1 + B[1,0]*Sx2/2) + (B[0,1]*Sy1 + B[1,1]*Sy2/2) + (B[0,2]*Sz1 + B[1,2]*Sz2/2))
    H_left = H_anisleft + H_Heis + H_zeemanleft
    H_arr[0,:,:,:,:] = np.reshape(H_left,(d,d,d,d))
    
    #Right edge of chain
    H_Heis = J[L-2]*(np.kron(Sx, Sx) + np.kron(Sy, Sy) + np.kron(Sz, Sz))
    H_zeemanright = -g*muB*((B[L-2,0]*Sx1/2 + B[L-1,0]*Sx2) + (B[L-2,1]*Sy1/2 + B[L-1,1]*Sy2) + (B[L-2,2]*Sz1/2 + B[L-1,2]*Sz2))    
    H_right = H_anisright + H_Heis + H_zeemanright
    H_arr[L-2,:,:,:,:] = np.reshape(H_right,(d,d,d,d))
    
    """
    print("begin")
    print(H_arr[0])
    print()
    print(H_arr[1])
    print()
    print(H_arr[2])
    """
    return H_arr
