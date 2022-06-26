import os
os.chdir(os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt
import time



from operators import Create_Ham, expectation_chain, expectation_site, Create_Sz, Create_Sx, Create_Sy




s=1/2                           # s=1/2 during the experiment
d = int(2*s+1)                  # Dimension of the spins



L = 13                           # Length of the chain
J = 1                           # Coupling constant in Hamiltonian, 1 = Anti ferromagnetic coupling in chain, -1 = ferromagnetic coupling
h = 3  


temperature = 0.5                 # Temperature of the system
steps = 800                     # Number of imaginary time steps taken to reach the system temperature
temp_plot_bound = 400           # Results of imaginary time evolution are plotted for the temperature interval (0, temp_plot_bound)


T = 2000                                        # Number of real time steps to be taken
time_step = 1e-2                                # Size of the time step
time_array = np.linspace(0, T*time_step, T)     # Array containing all time steps, used for plotting real time evolution


Nin = np.arange(L)


Sx = Create_Sx(s)
Sy = Create_Sy(s)
Sz = Create_Sz(s)




def Thermal_Matrix(Ham, eigval, eigvec, eigvec_inv, beta):
    time = -1*beta
    
    exp_matrix = np.diag(np.exp(eigval*time))

    #eigval_matrix = np.diag(eigval)
    
    rho = np.matmul(eigvec, exp_matrix)
    rho = np.matmul(rho, eigvec_inv)
    
    rho = rho / np.trace(rho)
    return rho



def Time_Evolution(rho, Ham, eigval, eigvec, eigvec_inv, t):
    exp_matrix = np.diag(np.exp(eigval*-1j*t))
    #exp_matrix_2 = np.diag(np.exp(eigval*1j*t))
    
    U = np.matmul(np.matmul(eigvec, exp_matrix), eigvec_inv)
    Ut = np.transpose(np.conj(U))
    
    rho = np.matmul(np.matmul(U, rho), Ut)
    
    return rho


"""
rho, Ham = Thermal_Matrix()
print(rho)

Sx = Create_Sx(s)
Sy = Create_Sy(s)
Sz = Create_Sz(s)

print(expectation_chain(rho, Sz, s, L))
    
print(np.trace(np.matmul(rho, Ham)))
"""

def h_vs_sz_compare():
    h_array = np.linspace(0,3.5, 70)
    temp_array = np.array([0.5, 2.5, 5])
    
    results = np.zeros((len(temp_array), len(h_array)))
        
    for i in range(0, len(h_array)):
        h=h_array[i]
        Ham, eigval, eigvec, eigvec_inv = Create_Ham(s, L, h, J)
        
        for j in range(0, len(temp_array)):
            temperature = temp_array[j]
            beta = 1/temperature
            rho = Thermal_Matrix(Ham, eigval, eigvec, eigvec_inv, beta)
            results[j,i] = expectation_chain(rho, Sz, s, L)

    
    plt.figure(dpi=200)
    for j in range(0, len(temp_array)):
        plt.plot(h_array, results[j], label = "T= " + str(temp_array[j]))
    plt.xlabel("magnetic field strength h")
    plt.ylabel(r'$<S_z>$  of chain')
    plt.legend()
    plt.show()
        
    filename = "h_vs_Sz"
    saveloc = "C:\\Users\\matth\\OneDrive\\Documents\\TUDelft\\jaar3\\BEP\\BEP_work\\code\\data\\exact\\" + filename
    #np.save(saveloc, results)
        
    return



def energy_plot():
    beta = 1/temperature
    beta_array = np.linspace(0, beta, steps)
    
    results = np.zeros(steps)
    
    Ham, eigval, eigvec, eigvec_inv = Create_Ham(s, L, h, J)
    
    for i in range(0,steps):
        beta = beta_array[i]
        rho = Thermal_Matrix(Ham, eigval, eigvec, eigvec_inv, beta)
        results[i] = np.trace(np.matmul(rho, Ham))



    temp_step = beta/(steps-1)

    
    temperature_array =  np.ones(steps) / (np.arange(steps) * temp_step)   #array that contains the temperature at each time step
    
    print(temperature_array)
    
    plt.figure(dpi=200)
    plt.plot(temperature_array, results, linewidth = 0.7)
    plt.xlim(400, 0)
    
    filename = "Energy"
    saveloc = "C:\\Users\\matth\\OneDrive\\Documents\\TUDelft\\jaar3\\BEP\\BEP_work\\code\\data\\exact\\" + filename
    #np.save(saveloc, results)
    
    return



def total_Sz_plot():
    beta = 1/temperature
    beta_array = np.linspace(0, beta, steps)
    
    results = np.zeros(steps)
    
    Ham, eigval, eigvec, eigvec_inv = Create_Ham(s, L, h, J)
    
    for i in range(0,steps):
        beta = beta_array[i]
        rho = Thermal_Matrix(Ham, eigval, eigvec, eigvec_inv, beta)
        results[i] = expectation_chain(rho, Sz, s, L)



    temp_step = beta/(steps-1)

    
    temperature_array =  np.ones(steps) / (np.arange(steps) * temp_step)   #array that contains the temperature at each time step
    
    print(temperature_array)
    
    plt.figure(dpi=200)
    plt.plot(temperature_array, results, linewidth = 0.7)
    plt.xlim(400, 0)
    
    filename = "Sz"
    saveloc = "C:\\Users\\matth\\OneDrive\\Documents\\TUDelft\\jaar3\\BEP\\BEP_work\\code\\data\\exact\\" + filename
    #np.save(saveloc, results)
    
    return




def System_Time_Evol():

    
    beta = 1/temperature
    
    Ham, eigval, eigvec, eigvec_inv = Create_Ham(s, L, h, J)
    
    rho = Thermal_Matrix(Ham, eigval, eigvec, eigvec_inv, beta)
    
    #print(expectation_chain(rho, Sz, s, L))
    #print(expectation_site(rho, 0, Sz, s, L))
    
    
    results = np.zeros((L, T))
    for i in range(0, T):
        rho = Time_Evolution(rho, Ham, eigval, eigvec, eigvec_inv, time_step)
        #print(expectation_site(rho, 0, Sz, s, L))
        for j in range(0, len(Nin)):
            results[j, i] = expectation_site(rho, Nin[j], Sz, s, L)
    
    #print(expectation_chain(rho, Sz, s, L))
    plt.figure(dpi=200)
    
    for j in range(0, len(Nin)):
        plt.plot(np.arange(T), results[j], label="Site " + str(Nin[j]))
        
    plt.legend()
    plt.show()
    return


time1  = time.time()

#h_vs_sz_compare()
#energy_plot()
System_Time_Evol()
#total_Sz_plot()

print("Elapsed time: " + str(time.time()-time1))




