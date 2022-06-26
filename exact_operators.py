import numpy as np



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






def Create_Ham(s,L,h,J):
    d= int(2*s+1)
    Id = np.eye(d, dtype=complex)
    Sx = Create_Sx(s) *2
    Sy = Create_Sy(s) *2
    Sz = Create_Sz(s) *2
    
    Ham = np.zeros((d**L, d**L), dtype=complex)
    
    Sx_2 = np.kron(Sx, Sx)
    Sy_2 = np.kron(Sy, Sy)

    
    for i in range(0, L-1):            
        term = np.ones(1, dtype=complex)
        for j in range(0,i):
            term = np.kron(term, Id)
        term = np.kron(term, Sx_2)
        for j in range(i+2, L):
            term = np.kron(term, Id)
        Ham += J*term
        
    for i in range(0, L-1):            
        term = np.ones(1, dtype=complex)
        for j in range(0,i):
            term = np.kron(term, Id)
        term = np.kron(term, Sy_2)
        for j in range(i+2, L):
            term = np.kron(term, Id)
        Ham += J*term


    for i in range(0,L):
        term = np.ones(1, dtype=complex)
        for j in range(0,i):
            term = np.kron(term, Id)
        term = np.kron(term, Sz)
        for j in range(i+1, L):
            term = np.kron(term, Id)
        Ham += J*h*term
    
    eigval, eigvec = np.linalg.eig(Ham)
    eigvec_inv = np.linalg.inv(eigvec)
    
    return Ham, eigval, eigvec, eigvec_inv









def Chain_Operator(i,A, s,L):
    d= int(2*s+1)
    Id = np.eye(d, dtype=complex)
    
    Op = np.ones(1, dtype=complex)
    for j in range(0,i):
        Op = np.kron(Op, Id)
    Op = np.kron(Op, A)
    for j in range(i+1, L):
        Op = np.kron(Op, Id)
    return Op





def expectation(rho, op):
    return np.real(np.trace(np.matmul(rho,op)))


def expectation_site(rho, i, A, s, L):
    Op = Chain_Operator(i,A,s,L)
    E = expectation(rho, Op)
    return np.real(E)

def expectation_chain(rho, A, s, L):
    E=0+0j
    for i in range(0, L):
        Op = Chain_Operator(i,A,s,L)
        E += expectation(rho, Op)
    E=E/L
    return np.real(E)

















