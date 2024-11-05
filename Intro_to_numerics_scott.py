# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 16:48:55 2024

@author: sl8924
"""

import numpy as np
import matplotlib.pyplot as plt


#Set up parameters of the linear shallow-water equations and discretisation for the test of stability
def setup_stability(runnumber):
    #Parameters
    g = 10       # Gravitational acceleration (rounded to 10 for convenience of numbers)
    H = 10       # Mean water depth
    #Set up discretisation
    nx = 200
    x = np.linspace(0.0, np.pi*2, nx+1)
    dx = 1/nx
    if runnumber == 1:
        nt = 8000
    elif runnumber ==2:
        nt = 4000
    elif runnumber == 3:
        nt = 3999
    runlength = nt
    endTime = 4
    dt = endTime/nt
    k = 1
    waveSpeed = np.sqrt(g*H)
    c = dt*waveSpeed/dx
    return g, H, nx, x, dx, nt, runlength, endTime, dt, k, waveSpeed, c

#Set up parameters of the linear shallow-water equations and discretisation for the test of dispersion
def setup_dispersion_relation():
    #Parameters 
    g = 10       # Gravitational acceleration (rounded to 10 for convenience of numbers)
    H = 10       # Mean water depth
    #Set up discretisation
    nx = 200
    x = np.linspace(0.0, np.pi*2, nx+1)
    dx = 1/nx
    nt = 50000
    runlength = 1000
    endTime = 4
    dt = endTime/nt
    return g, H, nx, x, dx, nt, runlength, endTime, dt

#Set up parameters of the linear shallow-water equations and discretisation for the test of grid-scale waves
def setup_grid_scale_waves():
    #Parameters
    g = 10       # Gravitational acceleration (rounded to 10 for convenience of numbers)
    H = 10       # Mean water depth
    #Set up discretisation
    nx = 6
    x = np.linspace(0.0, 1.0, nx+1)
    dx = 1/nx
    nt = 500
    runlength = nt
    endTime = 0.1
    dt = endTime/nt
    return g, H, nx, x, dx, nt, runlength, endTime, dt

#Initialise disturbance as either a sinusoid or a grid-scale wave
def init_eta(wavetype):
    u = np.zeros(nx+1)
    if wavetype == "sinusoidal":
        #Initialise eta as a sinusoid 
        eta = np.sin(k*x)   
    else:
        eta = []
        #Initialise eta as a grid-scale wave
        for i in range(len(x)):
                eta.append((-1)**(i))
    return eta, u

#Run equations and store solution in a list
def run_equations(eta,u):
    #Create lists to store the new values of eta and u 
    eta_new = eta
    u_new = np.zeros(nx+1)
    #List to store solution
    eta_list = []
    #Compute solution
    for n in range(runlength):
        #Calculate new value of eta 
        for j in range(1,nx):
            eta_new[j] = eta[j] - (H * dt / ( 2*dx)) * (u[j+1] - u[j -1])
        #Apply periodic boundary conditions
        eta_new[-1] = eta_new[-1] - (H * dt / ( 2*dx)) * (u[1] - u[-2])
        eta_new[0] =  eta_new[-1]
        #Calculate new value of u 
        for j in range(1,nx):
            u_new[j] = u[j] - g*(dt/(2*dx)) *(eta_new[j+1]-eta_new[j-1])
        #Apply periodic boundary conditions
        u_new[-1] = u_new[-1] - g*(dt/(2*dx)) *(eta_new[1]-eta_new[-2])
        u_new[0] =  u_new[-1]
        
        #Update eta and u to their new values 
        eta = eta_new.copy()
        u = u_new.copy()
        #Append solution
        eta_list.append(eta) 
    return eta_list

#Find the frequency of oscillation for a given wavenumber of initial sine wave
def find_frequency(k):
    eta = np.sin(k*x)
    u = np.zeros(nx+1)
    y = run_equations(eta, u)
    #We find frequency from oscillations of the point x[10]; store these in a list
    eta_10 = []
    for i in range(runlength):
        eta_10.append(y[i][10])
    #Find the frequency from the times that the wave crosses 0
    zero_crossings = np.where(np.diff(np.sign(eta_10)))[0]
    time_diff = []
    for i in range(1, len(zero_crossings)):
        time_diff.append(zero_crossings[i] - zero_crossings[i-1])
    av_time_diff = np.mean(time_diff)
    period = 2* av_time_diff * dt
    frequency = 1 / period 
    return frequency
 
#Plot the dispersion relation for the numerical scheme against the analytic gravity waves, for wavenumbers 
#up to k   
def plot_dispersion_relation(k):
    #Store wavenumbers, computed frequencies, analytic frequencies
    k_list = []
    freqs = []
    analytic = []
    #Calculate and plot the dispersion relation
    for i in range(1,k+1):
        k_list.append(i*dx)
        freqs.append(find_frequency(i)*dx)
        analytic.append(i*dx*np.sqrt(g*H))
    plt.figure(figsize=(8, 5))
    plt.plot(k_list, freqs, label='Simulated')
    plt.plot(k_list, analytic, label= 'Analytic solution')
    plt.xlabel('k dx')
    plt.ylabel('$\omega$ dx')
    plt.legend()    

#Visualise solutions' evolution in time (only applicable for stability test and grid-scale waves test)
def visualise(eta_list):
    for n,eta in enumerate(eta_list):
        #Visualise solution 
        if (n+1) % 25 == 0:
            
            plt.figure(figsize=(8, 4))
            plt.plot(x, eta, label='Wave Height')
           
            plt.ylim([-2, 2])
            plt.legend()
            plt.title(f"     Time: {n*dt:.3f} s")
            plt.xlabel("x")
            plt.ylabel("Wave height")
            plt.grid()
            plt.show()
            
#Plot individual stills of simulation for stability and grid-scale waves
def plotter(eta_list, time, c):
    plt.figure(figsize=(8, 4))
    plt.plot(x, eta_list[time], label='Wave Height')
    plt.ylim([-2, 2])
    plt.legend()
    if c == 'N/A':
        plt.title(f"  Time: {time*dt:.3f} s")
    else:
        plt.title(f"  c = {c}   Time: {time*dt:.3f} s")
    plt.xlabel("x")
    plt.ylabel("Wave height")
    plt.grid()
    plt.show()

#Stability plot for c = 1    
g, H, nx,x, dx, nt, runlength, endTime, dt, k, waveSpeed, c = setup_stability(1)
eta, u = init_eta("sinusoidal")      
y = run_equations(eta, u)
plotter(y, 7999, c)

#Stability plot for c = 2
g, H, nx,x, dx, nt, runlength, endTime, dt, k, waveSpeed, c = setup_stability(2)
eta, u = init_eta("sinusoidal")      
y = run_equations(eta, u)
plotter(y, 3999, c)

#Stability plot for c = 2.0005
g, H, nx,x, dx, nt, runlength, endTime, dt, k, waveSpeed, c = setup_stability(3)
eta, u = init_eta("sinusoidal")      
y = run_equations(eta, u)
plotter(y, 689, 2.0005)
plotter(y, 789, 2.0005)
plotter(y, 839, 2.0005)


#Dispersion relation plot
g, H, nx,x, dx, nt, runlength, endTime, dt = setup_dispersion_relation()
plot_dispersion_relation(50) 
  
#Grid-scale waves plot
g, H, nx, x, dx, nt, runlength, endTime, dt = setup_grid_scale_waves()
eta, u = init_eta("grid-scale")
y = run_equations(eta,u)
plotter(y, 0, 'N/A')
plotter(y, 150, 'N/A')
plotter(y, 300, 'N/A')
plotter(y, 499, 'N/A')

