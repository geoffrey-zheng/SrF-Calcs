"""
Analysis of in-cell absorption spectroscopy of Sr on the 1S0 to 1P1 transition at 461 nm.
Deduce number of Sr atoms produced per pulse in the SrF experiment.
Date: Dec 9, 2022

Note: at least on my work desktop, need to run this in Anaconda Command Prompt 
(something to do with how I downloaded h5py package). works normally on macbook vscode ide though with conda environment.
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import scipy.optimize as optimize
import h5py

"""
Import the absorption data from the HDF5 file. 
HDF5 file = dictionary of (key, group). Key = name of each group, e.g. runHH:MM:SS.
Group = dictionary of (key, dset). Key = name of each dset, e.g. 0, 1, ... 
Dataset = list of tuples. Each tuple is one data point during a scan. 5000 data points per dataset.

So in our hierarchy: 
HDF5 = file with absorption data over all scanned frequencies.
Group = specific frequency. Each element of group is specific shot. Shots 0-1 were with YAG off (bkgnd). Shots 2-11 were with YAG on (actual data).
Dataset = data for each specific shot.
"""

absolute_path = os.path.dirname(__file__)
rel_path = "Sr absorption Dec 7 2022.hdf" #name of HDF5 file where our absorption data was written.
filename = os.path.join(absolute_path, rel_path)
hdf = h5py.File(filename, 'r') 

#create dictionary of run names and corresponding frequency detunings. Also make lists for corresponding average OD and Sr atom number (+ std dev).
run_names = ['run16:40:57', 'run16:42:43', 'run16:43:33', 'run16:44:08', 'run16:44:46', 
'run16:45:27', 'run16:46:21', 'run16:47:02', 'run16:47:40', 'run16:48:31', 'run16:53:11', 
'run16:54:02','run16:54:49', 'run16:55:27', 'run16:56:09', 'run16:57:12', 'run16:57:44', 
'run16:58:16']
freq_detunings = [-1.679e9, -1.479e9, -1.279e9, -1.079e9, -0.879e9, -0.679e9, -0.479e9, -0.279e9,
-0.079e9, 0.121e9, 0.543e9, 0.743e9, 0.943e9, 1.143e9, 1.343e9, 1.543e9, 1.743e9, 1.943e9]
name_to_detune_dict = {run_names[i] : freq_detunings[i] for i in range(len(run_names))}
avg_opt_depths = []
avg_Sr_num = []
stdev_opt_depths = []
stdev_Sr_num = []

for key in name_to_detune_dict:
    grp = hdf[key]
    tot_num_shots = len(grp)
    dset_list = list(np.empty(tot_num_shots)) #create list of datasets with length = total num of shots
    for key2 in grp:
        intkey = int(key2)
        dset_list[intkey] = grp[key2] #make list sorted by numerical key order (not "string numerical" order)

    #go through each dataset and extract time and PD voltage data
    len_dset = (dset_list[0].shape)[0]
    bkgnd_t_arrays = [] #list of numpy arrays for time (x-axis), for background data
    bkgnd_pdvolt_arrays = [] #list of numpy arrays for photodetector voltage (y-axis), for background data
    shot_t_arrays = [] #list of numpy arrays for time (x-axis), for shot data
    shot_pdvolt_arrays = [] #list of numpy arrays for pd voltage (y-axis), for shot data
    num_bkgnds = 2
    num_shots = 10

    for i in range(0, tot_num_shots):
        times, pd_voltages = zip(*dset_list[i])
        if (i < num_bkgnds):
            bkgnd_t_arrays.append(np.asarray(times))
            bkgnd_pdvolt_arrays.append(np.asarray(pd_voltages))
        else:
            shot_t_arrays.append(np.asarray(times))
            shot_pdvolt_arrays.append(np.asarray(pd_voltages))
    
    """
    #test raw data plots
    for i in range(0, num_bkgnds):
        plt.plot(bkgnd_t_arrays[i], bkgnd_pdvolt_arrays[i])
        plt.show()

    for i in range(0, num_shots):
        plt.plot(shot_t_arrays[i], shot_pdvolt_arrays[i])
        plt.show() 
    """
    
    #compute background subtraction and normalization of signal for time integral of pulse
    avg_bkgnd_volt = (np.average(bkgnd_pdvolt_arrays[0])+np.average(bkgnd_pdvolt_arrays[1]))/num_bkgnds
    #print(np.round(avg_bkgnd_volt,3))
    bkgnd_volt_tolerance = 0.05 #max diff allowed btwn background voltage as calculated from background shots vs actual shots
    shot_bkgnd_time = 3 #consider first 3 ms to be background in the shot data
    total_time = 50 #each shot was 50 ms in length
    slicenum = int(shot_bkgnd_time*len_dset/total_time)
    shot_absorb_arrays = []
    shot_recipabsorb_arrays = [] #equivalent to P0/P, i.e. the exponential of the optical depth OD

    for i in range(0, num_shots):
        shot_bkgnd_volt = np.average(shot_pdvolt_arrays[i][:slicenum])
        if (np.abs(shot_bkgnd_volt - avg_bkgnd_volt) < bkgnd_volt_tolerance):
            shot_bkgnd_volt = avg_bkgnd_volt
        #subtract out background voltage from shot voltage and compute the normalized %absorption, being sure to clip any voltage above background to the background value.
        shot_absorb_arrays.append(np.clip((shot_bkgnd_volt-shot_pdvolt_arrays[i])/shot_bkgnd_volt, a_min = 0, a_max = None))
        shot_recipabsorb_arrays.append(1.0/(1.0-shot_absorb_arrays[i]))
        #plt.plot(shot_t_arrays[i], shot_recipabsorb_arrays[i])
        #plt.show()

    #compute number of Sr atoms per pulse
    shot_num_atoms = []

    #experimental parameters
    Ad = 4.91 # in cm^2
    vel_par = 14000.0 # in cm/s
    Ls = 3.0 # in cm
    wavelen = 460.8623665e-7  # in cm
    wavenum = 2*np.pi/wavelen
    freq = 650.50323e12 # in Hz
    delta = name_to_detune_dict.get(key) # detuning in Hz
    crosssec = 3*(wavelen**2)/(2*np.pi) # in cm^2
    natlinewidth = 2*np.pi*32.0e6 # in Hz
    vel_perpspread = 3200.0 #transverse velocity std dev in cm/s

    #compute Doppler broadened cross section due to transverse velocity spread and accounting for detuning
    def Dopp_crosssec_integrand(vel_perp):
        prefactor = 1/(np.sqrt(2*np.pi)*vel_perpspread)
        multfactor1 = np.exp(-vel_perp**2/(2*vel_perpspread**2))
        multfactor2 = crosssec/(1+4*((delta + wavenum*vel_perp)/natlinewidth)**2)
        return prefactor*multfactor1*multfactor2

    Dopp_crosssec = integrate.quad(Dopp_crosssec_integrand,-np.infty,np.infty)[0]

    #fit OD plot to phenomenological function
    peak_od = []

    #fit function
    def od_fit(time_var, amp, tau1, tau2):
        return amp*(1 - np.exp(-(time_var-3.0)/tau1))*np.exp(-(time_var-3.0)/tau2)

    #fit function with computed optimized parameters
    def computed_od_fit(time_var):
        return ampfit*(1 - np.exp(-(time_var-3.0)/tau1fit))*np.exp(-(time_var-3.0)/tau2fit)

    #compute atom number from Beer-Lambert law for each shot and peak OD for each shot
    for i in range(0, num_shots):
        #fit data to OD fit function
        try:
            popt, pcov = optimize.curve_fit(od_fit, shot_t_arrays[i][shot_t_arrays[i] > 3.0], np.log(shot_recipabsorb_arrays[i][shot_t_arrays[i] > 3.0]))
            ampfit = popt[0]
            tau1fit = popt[1]
            tau2fit = popt[2]
        except RuntimeError:
            ampfit = 0.0
            tau1fit = 1.0
            tau2fit = 1.0
        
        """
        ***
        Comment this out if you don't want to see the absorption trace for each shot.
        ***

        print("Freq detuning (GHz):", f"{name_to_detune_dict.get(key):.3e}")
        print("Fitted OD Amp:", np.round(ampfit, 3))
        print("Fitted tau1 (ms):", np.round(tau1fit, 3))
        print("Fitted tau2 (ms):", np.round(tau2fit, 3))
        plt.plot(shot_t_arrays[i], np.log(shot_recipabsorb_arrays[i]), 'o', markersize = 2, label='data')
        plt.plot(shot_t_arrays[i], np.concatenate((np.zeros(len(shot_t_arrays[i][shot_t_arrays[i] <= 3.0])), od_fit(shot_t_arrays[i][shot_t_arrays[i] > 3.0], ampfit, tau1fit, tau2fit)), axis=None), label='fit')
        plt.xlabel('Time (ms)')
        plt.ylabel('Optical Depth')
        plt.legend()
        plt.show()
        """
    
        peak_od.append(np.amax(od_fit(shot_t_arrays[i][shot_t_arrays[i] > 3.0], ampfit, tau1fit, tau2fit)))

        #numerically integrate over all time slices in one shot
        init_time = shot_t_arrays[i][len(shot_t_arrays[i][shot_t_arrays[i] <= 3.0])] 
        fin_time = shot_t_arrays[i][len(shot_t_arrays[i]) - 1] 
        timeinteg = np.trapz(np.log(shot_recipabsorb_arrays[i]), x = shot_t_arrays[i]) #directly numpy integrate over the P0/P data
        timeintegfit = integrate.quad(computed_od_fit, init_time, fin_time)[0] #scipy integrate over the fitted OD function 
        #print(timeinteg)
        #print(timeintegfit)

        #compute atom number
        N_Sr = Ad*vel_par*(timeinteg/1000.0)/(Ls*Dopp_crosssec) #use the scipy integration over fitted OD function. Divide by 1000 to convert ms to s.
        shot_num_atoms.append(N_Sr)
        #print("Number of Sr atoms per pulse:", f"{N_Sr:.2e}")
        #print("Cross sec (cm^2):", f"{crosssec:.2e}")
        # print("Doppler broadened cross sec (cm^2):", f"{Dopp_crosssec:.2e}")

    #convert lists to arrays and replace any zero values with NaN for further averaging
    peak_od_nparray = np.array(peak_od)
    shot_num_atoms_nparray = np.array(shot_num_atoms)
    peak_od_nparray[peak_od_nparray==0] = np.nan
    shot_num_atoms_nparray[shot_num_atoms_nparray==0] = np.nan

    avg_pk_od = np.round(np.nanmean(peak_od_nparray), 3)
    mdn_pk_od = np.round(np.nanmedian(peak_od_nparray), 3)
    stdev_pk_od = np.round(np.nanstd(peak_od_nparray), 3)
    avg_num_atoms = np.nanmean(np.array(shot_num_atoms_nparray))
    mdn_num_atoms = np.nanmedian(np.array(shot_num_atoms_nparray))
    stdev_num_atoms = np.nanstd(np.array(shot_num_atoms_nparray))

    """
    print("Freq detuning (GHz):", f"{name_to_detune_dict.get(key):.3e}")
    print("Average peak OD of pulse:", avg_pk_od)
    print("Median peak OD of pulse:", mdn_pk_od)
    print("Std Dev peak OD of pulse:", stdev_pk_od)
    print("Average number of Sr atoms per pulse:", f"{avg_num_atoms:.2e}")
    print("Median number of Sr atoms per pulse:", f"{mdn_num_atoms:.2e}")
    print("Std Dev number of Sr atoms per pulse:", f"{stdev_num_atoms:.2e}")
    """
    
    avg_opt_depths.append(avg_pk_od)
    avg_Sr_num.append(avg_num_atoms)
    stdev_opt_depths.append(stdev_pk_od)
    stdev_Sr_num.append(stdev_num_atoms)

#plot frequency detuning vs OD. Compute fitted gaussian function for absorption spectrum.
freq_detunings_arr = np.array(freq_detunings)
avg_opt_depths_arr = np.array(avg_opt_depths)
stdev_opt_depths_arr = np.array(stdev_opt_depths)
avg_Sr_num_arr = np.array(avg_Sr_num)
stdev_Sr_num_arr = np.array(stdev_Sr_num)

def dopp_broadened_spec(freq, amp, centerfreq, stddev):
    return amp*np.exp(-(freq-centerfreq)**2/(2*stddev**2))


popt2, pcov2 = optimize.curve_fit(dopp_broadened_spec, freq_detunings_arr/1.0e9, avg_opt_depths_arr, p0=[2.0, 0.0, 0.2])
fitted_od_amp = popt2[0]
fitted_od_ctrfreq = popt2[1]
fitted_od_stdevfreq = popt2[2]

plt.errorbar(freq_detunings_arr/1.0e9, avg_opt_depths_arr, yerr = stdev_opt_depths_arr, fmt='o', capsize = 2.0, label='data')
plt.plot(np.linspace(-2.0, 2.0, 1000), dopp_broadened_spec(np.linspace(-2.0, 2.0, 1000), fitted_od_amp, fitted_od_ctrfreq, fitted_od_stdevfreq), label='fit')
plt.xlabel('Frequency Detuning (GHz)')
plt.ylabel('Optical Depth')
plt.legend()
plt.show()

print("Fitted OD amplitude :", np.round(fitted_od_amp, 3))
print("Fitted Center Freq (GHz):", np.round(fitted_od_ctrfreq, 3))
print("Fitted Std Dev Freq (GHz):", np.round(fitted_od_stdevfreq, 3))
print("FWHM (GHz):", np.round(fitted_od_stdevfreq*2*np.sqrt(2*np.log(2)), 3))

#also compute average Sr atom number across all detunings.
print("Average number of Sr atoms across all scans:", f"{np.average(avg_Sr_num_arr):.2e}")
print("Median number of Sr atoms across all scans:", f"{np.median(avg_Sr_num_arr):.2e}")
print("Std dev of num of Sr atoms across all scans:", f"{np.std(avg_Sr_num_arr):.2e}")
print("Std error of mean num of Sr atoms:", f"{np.std(avg_Sr_num_arr)/np.sqrt(tot_num_shots):.2e}")
print(tot_num_shots)

#plot frequency detuning vs Sr atom number. 
plt.errorbar(freq_detunings_arr, avg_Sr_num_arr, yerr = stdev_Sr_num_arr, fmt='o', capsize = 2.0)
plt.plot(freq_detunings_arr, np.full(len(freq_detunings_arr), np.average(avg_Sr_num_arr)), label='avg')
plt.fill_between(freq_detunings_arr, np.full(len(freq_detunings_arr), np.average(avg_Sr_num_arr))-np.std(avg_Sr_num_arr)/np.sqrt(tot_num_shots), np.full(len(freq_detunings_arr), np.average(avg_Sr_num_arr))+np.std(avg_Sr_num_arr)/np.sqrt(tot_num_shots), alpha=0.3)
plt.plot(freq_detunings_arr, np.full(len(freq_detunings_arr), np.median(avg_Sr_num_arr)), label = 'mdn')
plt.xlabel('Frequency Detuning (GHz)')
plt.ylabel('Number of Sr Atoms per Pulse')
plt.legend()
plt.show()

