"""
Code for communicating with R&S NRP18S MW power meter.
Has capability to read a continuous average power measurement and a triggered trace measurement.

Purpose: record and log to hdf file the microwave power trace measurement during each shot of the experimental cycle.
should enable us to see if there are shot-to-shot power fluctuations.

RsInstrument package found at: https://pypi.org/project/RsInstrument/
RsInstrument read the docs: https://rsinstrument.readthedocs.io/
Subsets of this code borrowed from: https://github.com/Rohde-Schwarz/Examples/tree/main

GZ, Mar. 2024
"""

from RsInstrument import *
import numpy as np
import matplotlib.pyplot as plt
import time
import math
import csv
import h5py

"""
Library of user-defined functions for using R&S NRPxxS MW power meter to take power measurement.
Can probably be streamlined into a class later.
Remaining: incorporate into main experimental program so it automatically accepts trigger from there.
"""

# -----------------------------------------------------------
# Query VISA resource string for connection to NRP
# -----------------------------------------------------------
def queryNRPstring() -> str:
    instr_list = RsInstrument.list_resources("?*") 
    return instr_list[0]
    

# -----------------------------------------------------------
# Initialize USB connection to NRP
# -----------------------------------------------------------
def USBconnectNRP() -> RsInstrument:
    instr = None
    RsInstrument.assert_minimum_version('1.21.0.78')
    try:
    # Make sure VISA resource string for USB connection to NRP18S is correct
        resource_string = queryNRPstring()
        instr = RsInstrument(resource_string, True, False)
        idn = instr.query_str('*IDN?')
        print(f"\nHello, I am: '{idn}'")
        #print(f'RsInstrument driver version: {instr.driver_version}')
        #print(f'Visa manufacturer: {instr.visa_manufacturer}')
        print(f'Instrument full name: {instr.full_instrument_model_name}')
        #print(f'Instrument installed options: {",".join(instr.instrument_options)}')
        instr.visa_timeout = 5000  # Timeout for VISA Read Operations in ms
        instr.instrument_status_checking = True  # Error check after each command
    except ResourceError as ex:
        print('Error initializing the instrument session:\n' + ex.args[0])
        exit()

    instr.write_str("*RST")  # Reset the instrument to system default
    instr.write_str("INIT:CONT OFF")  # Set instrument to idle

    return instr

# -----------------------------------------------------------
# Initialize settings for a single "continuous average" measurement
# See R&S NRPxxS Three-Path Power Sensors User Manual pg. 179 for list of power sensor commands
# -----------------------------------------------------------
def NRPAvgModeSettings(instr: 'RsInstrument', freq: 'float', countsForAvg: 'int', apertime: 'float') -> None:
    instr.write_str("*RST")  # Reset the instrument to system default
    instr.write_str("INIT:CONT OFF")  # Set instrument to idle
    instr.write_str('SENS:FUNC \"POW:AVG\"')  # set continuous average mode
    instr.write_str("SENS:FREQ " + str(freq)) #set carrier freq to be measured
    instr.write_str('SENS:AVER:STAT ON')  # turn on "averaging mode"
    instr.write_str('SENS:AVER:COUNT:AUTO OFF')  # turn off "auto-averaging" (where length of filter is determined by RS)
    instr.write_str("SENS:AVER:COUN " + str(countsForAvg))  # manually set how many readings are averaged for one measurement
    instr.write_str("SENS:POW:AVG:APER " + str(apertime))  # set time the sensor is "exposed to one measurement"
    instr.write_str('SENS:AVER:TCON REP')  # only output result after the measurement has been completed


# -----------------------------------------------------------
# Initialize settings for a trace measurement
# See R&S NRPxxS Three-Path Power Sensors User Manual pg. 179 for list of power sensor commands
# -----------------------------------------------------------
def NRPTraceModeSettings(instr: 'RsInstrument', freq: 'float', countsForAvg: 'int', points: 'int', duration: 'float', realtime: 'bool') -> None:
    if realtime:
        realtimestr = "ON"
        averaging = "OFF"
    else:
        realtimestr = "OFF"
        averaging = "ON"
    instr.write_str("*RST")  # Reset the instrument to system default
    instr.write_str("INIT:CONT OFF")  # Set instrument to idle
    instr.write_str("SENS:FUNC \"XTIM:POW\"") # set trace mode
    instr.write_str("SENS:FREQ " + str(freq)) #set carrier freq to be measured
    instr.write_str("SENS:TRAC:AVER:STAT " + averaging) #manually set whether want averaging to be on or not
    instr.write_str("SENS:TRAC:REAL " + realtimestr) #manually set whether want each measurement in trace to be an averaged one or not
    instr.write_str("SENS:TRAC:AVER:COUN " + str(countsForAvg)) #manually set how many readings are averaged for one measurement
    instr.write_str("SENS:TRAC:POIN " + str(points)) #manually set number of data points taken per trace sequence
    instr.write_str("SENS:TRAC:TIME " + str(duration)) #manually set duration of trace measurement

    #configure trigger so the trace is only measured when triggered by experiment
    instr.write_str("TRIG:SOUR EXT2") # set external trigger from SMB connector on power meter
    instr.write_str("TRIG:EXT2:IMP HIGH") #set SMB connector impedance as 10 kOhm

    instr.write_str("SENS:RANG:AUTO OFF") #Set auto-range to off
    instr.write_str("SENS:RANG 0") #Set sensitivity to most sensitive (needed for measuring after 20 dB pickoff, at least for the lowest powers out of Agilent)

    instr.write_str("SENS:CORR:OFFS:STAT ON") #set offset correction to on when looking after Qorvo amp
    instr.write_str("SENS:CORR:OFFS 20") #set level offset to 20 dB to account for directional coupler

# -----------------------------------------------------------
# Tell NRP to begin measurement and check whether measurement was successful
# -----------------------------------------------------------
def beginNRPmeasure(instr: 'RsInstrument', waitpts: 'int') -> None:
    instr.write_str("INIT:IMM")  # starts measurement by changing from idle to waiting for trigger

    success = False
    for i in range(0, waitpts):
        status = instr.query_int('STAT:OPER:TRIG:COND?')
        if (status != 2):
            # status = 2 if not triggered, 0 once triggered
            # once it has received trigger, it executes the measurement and is done
            success = True
            break
        time.sleep(0.02)

    if not success:
        raise TimeoutError("Measurement timed out")


# -----------------------------------------------------------
# Fetch (but not log) measurement results for NRP continuous average measurement.
# Returned format is string of (measured power in mW, measured power in dBm)
# -----------------------------------------------------------
def fetchNRPContAvgMeasure(instr: 'RsInstrument') -> str:
    instr.write_str('FORMAT ASCII')
    results = instr.query_str('FETC?').split(',')
    power_mw = float(results[0])*1e3
    if power_mw < 0:
        power_mw = 1e-6
    power_dbm = 10 * math.log10(power_mw)
    return str(f'Measured power: {power_mw:.6f} mW, {power_dbm:.3f} dBm')

# -----------------------------------------------------------
# Fetch (but not log) measurement results for NRP trace measurement.
# Returned format is numpy array of powers in mW
# -----------------------------------------------------------
def fetchNRPTraceMeasure(instr: 'RsInstrument') -> list:
    instr.write_str('FORMAT ASCII')
    results = instr.query_str('FETC?').split(',')
    power_mw = np.array([float(ele) for ele in results])*1e3
    power_mw[power_mw < 0] = 1e-6
    return power_mw
    
    
# -----------------------------------------------------------
# Fetch and log measurement results for NRP.
# Returned format is list of [timestamp, measured power in mW, measured power in dBm]
# -----------------------------------------------------------
def logNRPresults(instr: 'RsInstrument', timeZero: 'float') -> list:
    pwrdata = [round((time.time() - timeZero)*1e3, 6)] #timestamp in ms, rounded to 6 digits after decimal pt
    pwrresult = fetchNRPContAvgMeasure(instr)
    for t in pwrresult.split():
        try:
            pwrdata.append(float(t))
        except ValueError:
            pass
    
    return pwrdata


# -----------------------------------------------------------
# Main code
# -----------------------------------------------------------

#connect instrument and initialize settings to do a trace measurement
myinstr = USBconnectNRP()
points = 200 #num of points in the trace measurement
duration = 100e-6 #in seconds
NRPTraceModeSettings(myinstr, freq=14.89012e9, countsForAvg=1, points=points, duration=duration, realtime=True)


timings = []
powers = []

#hard code the number of experimental shots we want traces for
numshots = 168
for i in range(numshots):
    beginNRPmeasure(myinstr, 2000) #each trigger has up to 2000*0.02 = 40 s timeout
    results = fetchNRPTraceMeasure(myinstr)
    timings.append(np.linspace(0, duration*1e6, points))
    powers.append(results)
    """
    plt.plot(np.linspace(0, duration*1e6, points), results)
    plt.xlabel("Time (us)")
    plt.ylabel("Power (mW)")
    plt.title("MW Power Trace Measurement After Switch")
    plt.show()
    """

#for given number of experimental shots, record traces and save in hdf file
path = "C:/Users/BufferLab/Desktop/RS-NRP-MW/"
hdf_filename = path + "MWPowerTraces_" + time.strftime("%Y%m%d") + ".hdf"
dtp = [("Time (us)", 'f'), ("Power (mW)", 'f')]
data = np.stack((timings, powers), axis=1)

with h5py.File(hdf_filename, "a") as hdf_file:
    #group corresponds to the run
    group = hdf_file.create_group(name="run_" + time.strftime("%H%M%S"))
    for i in range(0, numshots):
        rec_data = np.core.records.fromarrays(data[i], dtype=dtp) 
        group.create_dataset(
                name                    =   str(i),
                data                    =   rec_data,
                shape                   =   rec_data.shape,
                compression             =   "gzip",
        )

# Close NRP instrument session
myinstr.close()
print("measurements and recording complete!")

"""
NRPAvgModeSettings(myinstr, freq=14.89012e9, countsForAvg=1, apertime=70e-6)
#prepare file for logged results  
timeZero = time.time()
filename = "contavg.csv"
f = open(filename, "w+") #creates new file and allows to be overwritten subsequently
f.close()

#read NRP, fetch result, and log result to csv file. Record 50 points.
rawdata = [] # format: list of [timestamp, power in mW, power in dBm]
timeZero = time.time()
for i in range(500):
    beginNRPmeasure(myinstr, 200)
    #print(fetchNRPresults(myinstr))
    rawdata.append(logNRPresults(myinstr, timeZero))

# append measured data to our file for logging
# ultimately, if we have a lot of data, it might be wiser to write this to an hdf file
with open(filename,"a",newline='') as my_csv: 
    csvWriter = csv.writer(my_csv, delimiter=',')
    field = ["Time (ms)", "Power (mW)", "Power (dBm)"]
    csvWriter.writerow(field)
    csvWriter.writerows(rawdata)
    my_csv.close()


#plot result just to see 
datatime = np.array([i[0] for i in rawdata])
pwrmw = np.array([i[1] for i in rawdata])
plt.plot(datatime, pwrmw, 'o-')
plt.xlabel('Time (ms)')
plt.ylabel('Power (mW)')
plt.title('Single Shot Measurement, 70 us aperture time')
plt.show()

print("Average measured power (mW):", round(np.average(pwrmw), 3))
print("% Max Deviation from Avg:", round(100*max(np.abs(np.max(pwrmw) - np.average(pwrmw)), np.abs(np.min(pwrmw) - np.average(pwrmw)))/np.average(pwrmw), 3))
"""

# Other potentially relevant code (may or may not work):

#
#
# global data_trace, y_val
#
# def com_check():
#     """Check communication with the device"""
#     # Just knock on the door to see if instrument is present
#     print('Hello, I am ' + instr.idn_string)
#     if not instr.full_instrument_model_name.endswith('P'):
#         raise ValueError(f'Instrument "{instr.full_instrument_model_name}" is not a pulse-power sensor. This example only works with instr-xxP power sensors.')
#
#
# def meas_setup():
#     global y_val
#     # Reset
#     instr.write_with_opc("*RST;*CLS")
#     # Single sweep
#     instr.write("INIT:CONT OFF")
#     # Trace function
#     instr.write("SENS:FUNC 'XTIM:POW'")
#     # Frequency
#     instr.write("SENS:FREQ 1.000000e+009")
#     # Trace points
#     instr.write("SENS:TRAC:POIN 500")
#     number_points = instr.query_int('SENS:TRAC:POIN?')
#     # Trace length
#     instr.write('SENS:TRAC:TIME 10e-6')
#     trace_length_s = instr.query_float('SENS:TRAC:TIME?')
#     # Trace offset time
#     instr.write_with_opc("SENS:TRAC:OFFS:TIME 0")
#     y_val = np.linspace(0, trace_length_s, number_points)
#
#
# def trig_setup():
#     # Trigger settings
#     instr.write("TRIG:ATR:STAT OFF")
#     instr.write("TRIG:SOUR INT")
#     instr.write("TRIG:LEV 10e-6")
#     instr.write("TRIG:SLOP POS")
#     instr.write("TRIG:COUN 1")
#     instr.write("TRIG:DELay -10e-6")
#     instr.write("TRIGger:HYSTeresis 0.5")
#     instr.write("TRIGger:DTIME 0")
#     instr.write_with_opc("TRIG:Hold 0")
#
#
# def aver_setup():
#     # averaging settings
#     instr.write("SENS:TRACE:AVER:COUN 8")
#     instr.write("SENS:TRACE:AVER:STAT ON")
#     instr.write("SENS:TRACE:AVER:TCON REP")
#     instr.write("SENS:AVER:RES")
#     instr.write("SENS:TRACE:REAL OFF")
#     #instr.write_with_opc("SENS:TRACE:MEAS:STAT OFF")
#
#
# def measurement():
#     # Initiate a measurement
#     instr.write("FORM REAL,32")
#     instr.write("FORM:BORD NORM")
#     instr.write("STAT:OPER:MEAS:NTR 2")
#     instr.write("STAT:OPER:MEAS:PTR 0")
#     instr.write("STAT:OPER:TRIG:NTR 2")
#     instr.write("STAT:OPER:TRIG:PTR 0")
#     instr.write("INIT:IMM")
#
#
# def end_meas():
#     # Waiting for the end of the measurement
#     n = 0
#     while n < 2:
#         n = instr.query_int('STAT:OPER:MEAS:EVEN?')
#         time.sleep(0.010)
#
#
# def get_res():
#     # Get the result
#     global data_trace
#     data_trace = instr.query_bin_or_ascii_float_list('FETC?')
#
#
# def plot():
#     # Plot the results
#     global y_val
#     global data_trace
#     plt.plot(y_val, data_trace)
#     plt.ylabel('power / W')
#     plt.xlabel('time / us')
#     plt.show()
#
# meas_setup()
# trig_setup()
# aver_setup()
# measurement()
# end_meas()
# get_res()
# instr.close()
# plot()


