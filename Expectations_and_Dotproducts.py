import numpy as np


"""
Contains all code related to the expectation of an operator and the calculation of a dot product
"""


def calc_norm(gammas, lambdas, L,chi):
    """ function for calculating the norm of a state Phi with the 'zipping' method described in the thesis."""
    m_total = np.eye(chi)
    for j in range(0, L):        
        sub_tensor = np.tensordot(gammas[j,:,:,:],np.diag(lambdas[j+1,:]), axes=(1,0))
        mp = np.tensordot(np.conj(sub_tensor),sub_tensor,axes = (1,1))
        m_total = np.tensordot(m_total,mp,axes=([0,1],[0,2]))
    if np.isnan(m_total[0,0]):                                                        # For large N value can be too large
        lambdas[1:L] = lambdas[1:L]/1.5                                                 # These lines make sure that the function returns a value
        m_total[0,0] = calc_norm(gammas,lambdas,L, chi)           
    return m_total[0,0]                                                                 # Only the first element is non-zero

def dotproduct(gammas1,lambdas1,gammas2,lambdas2,L,chi):
    """ This function can be used to calculate the dotproduct between two states"""
    m_total = np.eye(chi)
    for j in range(0,L):
        sub_tensor1 = np.tensordot(gammas1[j,:,:,:],np.diag(lambdas1[j+1,:]), axes=(1,0))
        sub_tensor2 = np.tensordot(gammas2[j,:,:,:],np.diag(lambdas2[j+1,:]), axes=(1,0))
        mp = np.tensordot(np.conj(sub_tensor1),sub_tensor2,axes = (1,1))
        m_total = np.tensordot(m_total,mp,axes=([0,1],[0,2]))
    return m_total[0,0]    

def expectation(gammas,lambdas,M_arr,L):
    "Calculates the expectation of the two-site operator M_arr over the full chain"
    Energy = 0
    for i in range(0,L-1):
        Energy += Twosite_Expectation(i,gammas,lambdas,M_arr)
    return Energy
        
def Twosite_Expectation(i,gammas,lambdas,M_arr):
    """Calculates the expectation of a two-site operator on sites i:i+1 using the method described by Andre melo. M_arr is the measured operator"""
    theta = np.tensordot(np.diag(lambdas[i,:]), gammas[i,:,:,:], axes=(1,0))
    theta = np.tensordot(theta,np.diag(lambdas[i+1,:]),axes=(1,0))
    theta = np.tensordot(theta, gammas[i+1,:,:,:],axes=(2,0))
    theta = np.tensordot(theta,np.diag(lambdas[i+2,:]), axes=(2,0)) 
    theta_prime = np.tensordot(theta,M_arr[i,:,:,:,:],axes=([1,2],[2,3])) 
    result = np.tensordot(np.conj(theta_prime),theta,axes=([0,1,2,3],[0,3,1,2]))
    return result


def Twosite_norm(i,gammas,lambdas):
    """Calculates the norm using the method described by andre melo"""
    theta = np.tensordot(np.diag(lambdas[i,:]), gammas[i,:,:,:], axes=(1,0))
    theta = np.tensordot(theta,np.diag(lambdas[i+1,:]),axes=(1,0))
    theta = np.tensordot(theta, gammas[i+1,:,:,:],axes=(2,0))
    theta = np.tensordot(theta,np.diag(lambdas[i+2,:]), axes=(2,0)) 
    result = np.tensordot(np.conj(theta),theta,axes=([0,1,2,3],[0,1,2,3]))
    return result

def Singlesite_Expectation(i,gammas,lambdas,M_arr):
    """Calculates the expectation of an operator that works on a single site"""
    theta = np.tensordot(np.diag(lambdas[i,:]), gammas[i,:,:,:], axes=(1,0))
    theta = np.tensordot(theta,np.diag(lambdas[i+1,:]),axes=(1,0))    
    theta_prime = np.tensordot(theta,M_arr,axes=(1,1)) 
    result = np.tensordot(np.conj(theta_prime),theta,axes=([0,1,2],[0,2,1]))
    return result

def Singlesite_Expectation_Chain(gammas, lambdas, M_arr, L):
    """Similar to "expectation" but for a singlesite operator"""
    Energy = 0
    for i in range(0,L):
        Energy += Singlesite_Expectation(i, gammas, lambdas, M_arr)
    return Energy / L




    