import os
os.chdir(os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt

import scipy.constants as sc
import time as timing
import string

from MPS_Initial_State import Create_chainstate, Create_flipstate, Initializing_State, Create_Max_Mixed_State_Purified
from TimeEvolution_and_twosite import Time_evolution
from Expectations_and_Dotproducts import Singlesite_Expectation, calc_norm, Singlesite_Expectation_Chain
from Operators_and_Hamiltonian import Ham_Experiment_Extended,Create_Opparray,Excite_site, Create_Sz, Prob_state, Single_Site_Operator, Create_Sm, Create_Sp




###################




"""
The different values for the parameters and variables are created here
"""
#Chain Parameters

s=1/2                           # s=1/2 during the experiment
d = int(2*s+1)                  # Dimension of the spins
dnew = int(d**2)                # Dimension of purified system
snew = int((dnew-1)/2)          # 'spin' of the combined physical ancilla site


L = 6                           # Length of the chain
J = 1                           # Coupling constant in Hamiltonian, 1 = Anti ferromagnetic coupling in chain, -1 = ferromagnetic coupling
h = 3                           # Magnetic field strength in the z-direction


#Simulation Parameters


normalize = False               # Continuously maintain normalization. Required for imaginary time evolution.
anc_H = True                    # If True: ancillary system is subject to reverse time evolution    If False: ancillary system is subject to identity Hamiltonian
ST = 1                          # Order of Suzuki trotter expansion used. Global error is equal to order. Order 1,2,4 available. Order 1 highly advised.
chi = 12                        # Bond dimension. Determines where we truncate our Hilbert space


temperature = 2                 # Temperature of the system
steps = 800                     # Number of imaginary time steps taken to reach the system temperature
temp_plot_bound = 400           # Results of imaginary time evolution are plotted for the temperature interval (0, temp_plot_bound)


T = 2000                                        # Number of real time steps to be taken
time_step = 1e-2                                # Size of the time step
time_array = np.linspace(0, T*time_step, T)     # Array containing all time steps, used for plotting real time evolution


Meas_1=2                # After imaginary time evolution           #0 :Measure Sz.             1: Measure Energy           2: Measure total Sz
Meas_2=0                # After real time evolution                #0 :Measure Sz.             1: Measure Energy           2: Measure total Sz 
plot_type = 1           # 0: plot over time steps    1: plot over Temperature

Nin = np.arange(L)      #Index of sites you want to measure <Sz> for




###################




def Initialize_temp_step_array():
    beta = 1/temperature
    
    #-1 is required so that desired temperature is actually reached, otherwise the loop will stop slightly above the desired temperature
    temp_step = beta/(2*(steps-1))
    temp_step = temp_step*(-1j)
    
    temperature_array =  np.ones(steps) / (np.arange(steps) * 2 * temp_step/(-1j) )   #array that contains the temperature at each time step
    
    print("Temperature to be reached: " + str(np.real(temperature_array[-1])))
    print("Number of steps to be taken: " + str(steps))
    print("The size of the step in beta is: " + str(-temp_step/1j))
    return temp_step, temperature_array


def Initialize_temp_step_array_h_vs_sz(choice, temperature, steps):
    beta = 1/temperature
    
    #-1 is required so that desired temperature is actually reached, otherwise the loop will stop slightly above the desired temperature
    temp_step = beta/(2*(steps-1))
    temp_step = temp_step*(-1j)
    
    temperature_array =  np.ones(steps) / (np.arange(steps) * 2 * temp_step/(-1j) )   #array that contains the temperature at each time step
    
    print("Temperature to be reached: " + str(np.real(temperature_array[-1])))
    print("Number of steps to be taken: " + str(steps))
    print("The size of the step in beta is: " + str(-temp_step/1j))
    return temp_step, temperature_array



def Initialize_system_and_Finite_Temperature(temp_step, temperature_array):
    #Creating Hamiltonian used during the simulation
    H_arr = Ham_Experiment_Extended(L,J,h,s, False)
    #Creation of time operators
    O_arr = Create_Opparray(H_arr,L,dnew,temp_step)
    #Creating initial state
    gammas, lambdas, loc_size = Create_Max_Mixed_State_Purified(L, s, chi)
    
    print()
    print("Starting imaginary time evolution to desired temperature")
    if ST==0:
        return lambdas, gammas, loc_size, H_arr, O_arr
    if ST==1:
        Res = Time_evolution(gammas,lambdas,steps,O_arr,L,dnew,chi,loc_size, H_arr, s, Nin, Meas_1, normalize=True)
    
    time,Energy = Res
    return lambdas, gammas, loc_size, H_arr, O_arr, Res
    



def Initialize_system_and_Finite_Temperature_h_vs_sz(h, steps, temp_step, temperature_array):
    """
    This part is relatively unchanged from Justin's code
    Initializes the system and evolves to ground state using imaginary time evolution
    """
    #Creating Hamiltonian used during the simulation
    H_arr = Ham_Experiment_Extended(L,J,h,s, False)
    
    
    #Creation of time operators
    O_arr = Create_Opparray(H_arr,L,dnew,temp_step)
    
    #Creating initial state
    gammas, lambdas, loc_size = Create_Max_Mixed_State_Purified(L, s, chi)
    #gammas, lambdas, loc_size = Initializing_State(L,chi,dnew)
    
    print("Starting imaginary time evolution to desired temperature")
    ###
    #gammas en lambdas are updated inside this function but not returned
    ###
    if ST==0:
        return lambdas, gammas, loc_size, H_arr, O_arr
    
    if ST==1:
        #Res = Time_evolution(gammas,lambdas,T_ground,O_arrground,L,d,chi,loc_size, H_arr, s, Nin, Meas, normalize=True)
        Res = Time_evolution(gammas,lambdas,steps,O_arr,L,dnew,chi,loc_size, H_arr, s, Nin, Meas_1, normalize=True)
    
    
    time,Energy = Res
    
    #plot_measured(Meas, Meas_2, Res, temperature_array, Nin)
    
    #M_arr = np.kron(Create_Sz(s), np.eye(d))
    
    
    return lambdas, gammas, loc_size, H_arr, O_arr, Res




    
def Real_Time_Evolution(lambdas, gammas, loc_size, h, T, time_step, anc_H):
    #Creating Hamiltonian used during the simulation
    #Note that the measurement Hamiltonian may be different than the time evolution Hamiltonian. In the case that the ancilla's evolve backwards application of Hamiltonian will always yield zero
    H_arr_ev = Ham_Experiment_Extended(L,J,h,s, anc_H)              #Hamiltonian used for time evolution
    H_arr_meas = Ham_Experiment_Extended(L,J,h,s, ancilla_H=False)  #Hamiltonian used to measure the energy

    #Creation of time operators
    O_arr = Create_Opparray(H_arr_ev,L,dnew,time_step)
    
    print()
    print("Starting real time evolution")
    if ST==0:
        return lambdas, gammas, loc_size, H_arr_meas, O_arr
    if ST==1:
        Res = Time_evolution(gammas,lambdas,T,O_arr,L,dnew,chi,loc_size, H_arr_meas, s, Nin, Meas_2, normalize)

    time,Energy = Res
    return lambdas, gammas, loc_size, H_arr_meas, O_arr, Res


def plot_measured(Meas, plot_type, Res, temperature_array,  Nin):
    plt.figure(dpi=200)
    plot_colors = ["orange", "blue", "green", "red", "pink", "purple", "brown"]
    
    if Meas==0:             #Plot <Sz> for each particle in Nin
        if plot_type==0:
            for i in range(len(Nin)):   #Plot result against the time steps (time steps on x-axis)
                #plt.plot(Res[0][:],np.real(Res[1][:,i]),label="site "+str(Nin[i]+1) , linewidth=0.8, color=plot_colors[i])
                plt.plot(Res[0][:],np.real(Res[1][:,i]),label="site "+str(Nin[i]+1), linewidth=0.8)
            plt.xlim(Res[0][0], Res[0][-1])
            plt.xlabel('Time step t')
            plt.ylabel('$<S_z>$')
            plt.legend()
        else:
            for i in range(len(Nin)):   #Plot results against temperature (temperature on x-axis)
                plt.plot(temperature_array, np.real(Res[1][:,i]), label="site "+str(Nin[i]+1) , linewidth=0.7)
            plt.xlim(temp_plot_bound, 0)
            plt.xlabel('Temperature (K)')
            plt.ylabel('$<S_z>$')
            plt.legend()
    
    if Meas==1:           #Plot energy of the entire chain
        if plot_type==0:                #Plot result against the time steps (time steps on x-axis)
            plt.plot(Res[0][:],np.real(Res[1][:,0]))
            plt.xlim(Res[0][0], Res[0][-1])
            plt.xlabel('Time step t')
            plt.ylabel('Energy')
        else:                           #Plot results against temperature (temperature on x-axis)
            plt.plot(temperature_array,np.real(Res[1][:,0]))
            plt.xlim(temp_plot_bound, 0)
            plt.xlabel('Temperature (K)')
            plt.ylabel('Energy')

    if Meas==2:                   #Plot <Sz> of the entire chain
        if plot_type==0:                #Plot result against the time steps (time steps on x-axis)
            plt.plot(Res[0][:],np.real(Res[1][:,0]))
            plt.xlim(Res[0][0], Res[0][-1])
            plt.xlabel('Time step t')
            plt.ylabel('$<S_z>$ of chain')
        else:                           #Plot results against temperature (temperature on x-axis)
            plt.plot(temperature_array,np.real(Res[1][:,0]))
            plt.xlim(temp_plot_bound, 0)
            plt.xlabel('Temperature (K)')
            plt.ylabel('$<S_z>$ of chain')
        
    #plt.grid()
    plt.show()
    return
    


def Main_imtime_retime():
    #Initializing system and imaginary time evolution to desired temperature
    temp_step, temperature_array = Initialize_temp_step_array()
    lambdas, gammas, loc_size, H_arr, O_arr, Res = Initialize_system_and_Finite_Temperature(temp_step, temperature_array)
    plot_measured(Meas_1, plot_type, Res, temperature_array, Nin)
    
    #Real time evolution
    lambdas, gammas, loc_size, H_arr, O_arr, Res = Real_Time_Evolution(lambdas, gammas, loc_size, h, T, time_step, anc_H)
    plot_measured(Meas_2, 0, Res, time_array, Nin)

    return lambdas, gammas, loc_size





def Main_h_vs_sz():
    h_array=np.linspace(0, 3.5, 70)
    
    choice = 1
    
    T_array=np.array([0.5, 2.5, 5])
    steps=800
    #temp_step=1e-3
    
    results = np.zeros((len(T_array), len(h_array)))
    
    for i in range(len(T_array)):
        print(i)
        temperature = T_array[i]
        for j in range(len(h_array)):
            h=h_array[j]
            temp_step, temperature_array = Initialize_temp_step_array_h_vs_sz(choice, temperature, steps)
            lambdas, gammas, loc_size, H_arr, O_arr, Res = Initialize_system_and_Finite_Temperature_h_vs_sz(h, steps, temp_step, temperature_array)
            
            results[i, j] = np.real(Res[1][-1])
   
    plt.figure(dpi=200)
    for i in range(len(T_array)):
        plt.plot(h_array, results[i,:], label="T= " + str(T_array[i]))
    
    plt.xlabel("magnetic field strength h")
    #plt.ylabel("measured value")
    #plt.ylabel("<Sz of chain with L=" + str(L))
    plt.ylabel(r'$<S_z>$  of chain')
    plt.legend()
    plt.show()
    
    filename = "h_vs_Sz"
    saveloc = "C:\\Users\\matth\\OneDrive\\Documents\\TUDelft\\jaar3\\BEP\\BEP_work\\code\\data\\MPS\\" + filename
    np.save(saveloc, results)
       
    return















#"""
T1=timing.time()

if max(Nin)>L:
    print("Nin has to be smaller than or equal to L")
else:
    Main_h_vs_sz()

print("Time taken:",timing.time()-T1)
















