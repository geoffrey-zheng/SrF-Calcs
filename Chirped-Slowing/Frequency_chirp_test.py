import nidaqmx.constants
import numpy as np
import nidaqmx, time

samp_rate = 750000
samp_num = int(samp_rate * 40e-3)

ai_task = nidaqmx.Task("ai task "+time.strftime("%Y%m%d_%H%M%S"))
# add cavity ai channel to this task
ai_task.ai_channels.add_ai_voltage_chan("Dev3/ai0", min_val=-0.5, max_val=1.2, units=nidaqmx.constants.VoltageUnits.VOLTS)
# add laser ai channel to this task
ai_task.ai_channels.add_ai_voltage_chan("Dev3/ai3", min_val=-0.5, max_val=1.2, units=nidaqmx.constants.VoltageUnits.VOLTS)
# use the configured counter as clock and make acquisition type to be FINITE
ai_task.timing.cfg_samp_clk_timing(
                                    rate = samp_rate,
                                    # source = parent.config["counter PFI line"],
                                    active_edge = nidaqmx.constants.Edge.RISING,
                                    sample_mode = nidaqmx.constants.AcquisitionType.FINITE,
                                    samps_per_chan = samp_num
                                )



ao_task = nidaqmx.Task("ao task "+time.strftime("%Y%m%d_%H%M%S"))
# add cavity ao channel to this task
ao_task.ao_channels.add_ao_voltage_chan("Dev3/ao0", min_val=-10, max_val=10, units=nidaqmx.constants.VoltageUnits.VOLTS)
#add laser ao channel to this task
ao_task.ao_channels.add_ao_voltage_chan("Dev3/ao3", min_val=-10, max_val=10, units=nidaqmx.constants.VoltageUnits.VOLTS)
# use the configured counter as clock and make acquisition type to be FINITE
ao_task.timing.cfg_samp_clk_timing(
                                    rate = samp_rate,
                                    source = "/Dev3/ai/SampleClock",
                                    active_edge = nidaqmx.constants.Edge.RISING,
                                    sample_mode = nidaqmx.constants.AcquisitionType.FINITE,
                                    samps_per_chan = samp_num
                                )
ao_task.triggers.start_trigger.cfg_dig_edge_start_trig(trigger_source="/Dev3/ai/StartTrigger", trigger_edge=nidaqmx.constants.Edge.RISING)


cav_amp = 4.35 #in volts
cav_offset = -1.65 #in volts
chirp_amp = 2.0 # in volts
cav_waveform = np.tile((np.linspace(cav_amp, 0, 1500, dtype=np.float64)+cav_offset), 20)
chirp_waveform = np.concatenate((np.linspace(0, chirp_amp, 15000, dtype=np.float64), np.linspace(chirp_amp, 0, 15000, dtype=np.float64)))

ao_task.write(np.vstack((cav_waveform, chirp_waveform)), auto_start=True)
ai_task.start()
data = ai_task.read(number_of_samples_per_channel=30000, timeout=10)
print(data)
#AO0: write ramp cavity voltage by same amount every time. reverse sawtooth. 1 ms period (1 kHz). Voltage from 6 V to 0 V, with samp_num samples
#AO3: write chirp ramp to laser PZT. make symmetric triangle wave, ramp from 0 to 2 V in 20 ms and back down again.
#during every cavity ramp period, read HeNe signal from AI0 and laser signal from AI3, and use to calculate frequency of laser
ao_task.wait_until_done(timeout=2)
ao_task.stop()
ai_task.stop()
ai_task.close()
ao_task.close()

np.savetxt('chirp_test3.txt', data)