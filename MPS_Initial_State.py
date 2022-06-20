import numpy as np
from TimeEvolution_and_twosite import Two_site_Operator

"""
Contains the creation of our initial MPS state
"""



def Initializing_State(L,chi,d):
    """For the OBC, the maxium bond size is smaller than chi at the boundaries and here we compute what the maximum bond size can be for every site"""
    arr = np.arange(0,L+1)
    arr = np.minimum(arr, L-arr)
    arr = np.minimum(arr,chi)               # For large L, d**arr returns negative values, this line prohibits this effect
    loc_size = np.minimum(d**arr, chi)
    """
    Here  a random state Phi is constructed which is in the canoncial form from the bachelor thesis. Only the information up to loc_size is created.
    """
    global lambdas                                  #Made it global for ease of checking
    global gammas                                   #Made it global for ease of checking
    lambdas = np.zeros((L+1,chi), dtype = complex)
    gammas  = np.zeros((L,chi,chi,d), dtype = complex)
    np.random.seed()
    for j in np.arange(0,L):
        lef_size = loc_size[j]
        rig_size = loc_size[j+1]
        lambdas[j,:lef_size] = np.random.rand(lef_size)                                     # last lambda is neglected
        gammas[j,:lef_size,:rig_size,:d]  = np.random.rand(lef_size, rig_size,d)
    lef_size = loc_size[L]
    #lambdas[L,:lef_size] = np.random.rand(lef_size)                                         # last lambda is neglected
    lambdas[0,0] = 1
    lambdas[L,0] = 1 

    
    #Creation of an Identity operator
    Id = np.zeros((L-1,d,d,d,d), dtype=complex) #Identity array)
    for i in range(1,L-2):
        Id[i,:,:,:,:] = np.reshape(np.eye(d**2),(d,d,d,d))       
    Id[0,:,:,:,:] = np.reshape(np.eye(d**2),(d,d,d,d))      
    Id[L-2,:,:,:,:] = np.reshape(np.eye(d**2),(d,d,d,d)) 
    
    for t in range(1):
        for i in range(1,L-1,2):            # Odd bonds
            gammas, lambdas = Two_site_Operator(i,gammas,lambdas,Id,d,chi,loc_size,normalize=True)
        for i in range(0,L-1,2):            # Even bonds
            gammas, lambdas = Two_site_Operator(i,gammas,lambdas,Id,d,chi,loc_size,normalize=True)
    
    
    return gammas,lambdas,loc_size



def Create_flipstate(L,chi,d,Start=0):
    """
    Creates a state in the form |0,d-1,0,d-1,...>. Start determines what direction the first particle points in.
    """
    lambdas = np.zeros((L+1,chi), dtype = complex)
    gammas  = np.zeros((L,chi,chi,d), dtype = complex)
    for i in range(0,L,2): #Particles in up state
        gammas[i,0,0,(d-1)*(1-Start)]=1
    for i in range(1,L,2): #Particles in down state
        gammas[i,0,0,(d-1)*Start]=1
    for i in range(0,L+1):
        lambdas[i,0]=1
        
    arr = np.arange(0,L+1)
    arr = np.minimum(arr, L-arr)
    arr = np.minimum(arr,chi)               # For large L, d**arr returns negative values, this line prohibits this effect
    loc_size = np.minimum(d**arr, chi)
    return lambdas,gammas,loc_size


def Create_chainstate(L,chi,d,Start=0):  ##The edit has made the function equal to Create_flipstate
    """
    Creates a state in the form |0,d-1,0,d-1,...>. Start determines what direction the first particle points in,
    LocJ determines the positions at which the chain is split.
    """
    lambdas = np.zeros((L+1,chi), dtype = complex)
    gammas  = np.zeros((L,chi,chi,d), dtype = complex)
    #LocJ=np.append(np.append(-1,LocJ[LocJ<L]),L-1)
    
    for i in range(0, L-1, 2): #Particles in up state
        gammas[i,0,0,(d-1)*(1-Start)]=1
    for i in range(1, L-1, 2): #Particles in down state
        gammas[i,0,0,(d-1)*Start]=1
    """
    for j in range(len(LocJ)-1):
        for i in range(LocJ[j]+1,LocJ[j+1]+1,2): #Particles in up state
            gammas[i,0,0,(d-1)*(1-Start)]=1
        for i in range(LocJ[j]+2,LocJ[j+1]+1,2): #Particles in down state
            gammas[i,0,0,(d-1)*Start]=1
    """     
            
    for i in range(0,L+1):
        lambdas[i,0]=1
    
    arr = np.arange(0,L+1)
    arr = np.minimum(arr, L-arr)
    arr = np.minimum(arr,chi)               # For large L, d**arr returns negative values, this line prohibits this effect
    loc_size = np.minimum(d**arr, chi)
    return lambdas,gammas,loc_size




def Create_Max_Mixed_State_Purified(L, s, chi):       #creates  1/sqrt(2) ( |up up> + |down down> ) states at each site
    """
    Creates the purification of the maximally mixed state (infinite temperature state) in MPS form
    """
    d = 2*s+1    
    dnew = int(d**2)         #two systems of dimension d are combined, resulting in a new system of dimension d^2
    lambdas = np.zeros((L+1,chi), dtype = complex)
    gammas  = np.zeros((L,chi,chi,dnew), dtype = complex)
    
    onesite_gamma = np.zeros((chi,chi,dnew))
    onesite_gamma[0,0,0] = 1/np.sqrt(2)
    onesite_gamma[0,0,dnew-1] = 1/np.sqrt(2)
    gammas[:] = onesite_gamma
    lambdas[:, 0] = 1
    
    arr = np.arange(0,L+1)
    arr = np.minimum(arr, L-arr)
    arr = np.minimum(arr,chi)               # For large L, d**arr returns negative values, this line prohibits this effect
    loc_size = np.minimum(dnew**arr, chi)   #here also a dnew!
    return gammas, lambdas, loc_size
















