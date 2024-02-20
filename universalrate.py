"""
compute universal and unitary loss rate for laser-cooled molecular collisions in N=1 rotational manifold (distinguishable particles)
Run on Python 3 or later for no integer division.
"""
import scipy.constants as sc
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import pchip_interpolate
from scipy.optimize import curve_fit
from scipy.special import gamma

#common constants
SrF = 0
CaF = 1

mu = np.array([53.5, 29.5]).reshape((-1,1))*sc.physical_constants["atomic mass constant"][0]#amu convert to kg
Xdip = np.array([3.47, 3.08]).reshape((-1,1))*3.336e-30 #Debye convert to C*m 
Brot = np.array([7.5, 10.15]).reshape((-1,1))*1e9 #GHz convert to Hz
grndC6 = Xdip**4/((4*np.pi*sc.epsilon_0)**2*6*sc.h*Brot) # in J*m^6
rotC6 = np.array([4/25, 17/200, 13/40, (13*np.sqrt(3)+16)/100, -(13*np.sqrt(3)+16)/100, 1/25])*6*grndC6 # in J*m^6
stat_weights = np.array([2, 2, 2, 1, 1, 1])
temp = np.array([40, 80]).reshape((-1,1))*1e-6 #in kelvin


Lmin = 0 #s-wave
Lmax = 5 #h-wave
num_rot_states = 6 #number of eigenstates with distinct C6 in N=1 rotational manifold

"""
calculate unitarity limit (max inelastic cross section)
"""

#return barrier height in kelvin
def bar_height(mu, L, C6):
    if C6 < 0:
        return np.infty
    else:
        return ((sc.hbar**2*L*(L+1)/mu)**(1.5))*np.sqrt(1/(54*C6))/sc.Boltzmann

#return loss rate coefficient in cm^3/s
def beta_th(mu, L, C6, T):
    wavelen_th = np.sqrt(2*np.pi*sc.hbar**2/(mu*sc.Boltzmann*T))
    v_th = np.sqrt(8*sc.Boltzmann*T/(np.pi*mu))
    return 1e6*wavelen_th**2*v_th*(2*L+1)*np.exp(-1*bar_height(mu, L, C6)/T)/4

#array of beta_th for different N=1 collisional eigenstates
beta_rot_SrF = np.empty(num_rot_states) 
beta_rot_CaF = np.empty(num_rot_states)

#compute thermally averaged unitarity loss rate for SrF
for i in range(0, num_rot_states):
    beta_rot_SrF[i] = 0
    for j in range(Lmin, Lmax+1):
        beta_rot_SrF[i] += beta_th(mu[SrF], j, rotC6[SrF][i], temp[SrF])


beta_rot_SrF_avg = np.format_float_scientific(np.average(beta_rot_SrF, weights=stat_weights), 2)
print("Max SrF inelastic loss rate coefficient at T=" + str(round(1e6*temp[SrF][0])) + " uK in cm^3/s:", beta_rot_SrF_avg)

#compute thermally averaged unitarity loss rate for CaF
for i in range(0, num_rot_states):
    beta_rot_CaF[i] = 0
    for j in range(Lmin, Lmax+1):
        beta_rot_CaF[i] += beta_th(mu[CaF], j, rotC6[CaF][i], temp[CaF])

beta_rot_CaF_avg = np.format_float_scientific(np.average(beta_rot_CaF, weights=stat_weights), 2)
print("Max CaF inelastic loss rate coefficient at T=" + str(round(1e6*temp[CaF][0])) + " uK in cm^3/s:", beta_rot_CaF_avg)


"""
calculate universal limit (from J. Hutson paper)
"""

#results of pchip spline interpolation of Fig. 7 (universal loss, thermally averaged)
xvals = np.array([-0.66, -0.024, 0.573, 1.155, 1.746])#used webplotdigitizer to find values
yvals = np.array([0.3, 0.4, 0.5, 0.6, 0.7])#boundary values of logarithmic color bar
x = np.linspace(min(xvals), max(xvals), num=100)
y = pchip_interpolate(xvals, yvals, x)
plt.plot(xvals, yvals, 'o', label="observed")
plt.plot(x, y, label="interpolated")
plt.xlabel('$\log_{10}$(T bar)')
plt.ylabel('$\log_{10}$(K loss) (units of K bar)')

#curve fit a cubic polynomial (4 free parameters for 2 points, 2 slopes) to the points to extract a best-fit function
def cub_fit(x, a, b, c, d):
    return a*x**3+b*x**2+c*x+d

popt, pcov = curve_fit(cub_fit, xvals, yvals)
plt.plot(x, cub_fit(x, popt[0], popt[1], popt[2], popt[3]), label="fitted")
plt.legend()
plt.show()

#dim'less parameters relevant for Hutson paper
r6 = (2*mu*rotC6/sc.hbar**2)**(1/4)
abar = 2*np.pi*r6/(gamma(1/4)**2)
Kbar = abar*sc.h/mu*1e6 # in cm^3/s
Ebar = sc.hbar**2/(2*mu*abar**2)
Tbar = sc.Boltzmann*temp/Ebar

#get results from cubic curve fit to interpolated data
Kloss = 10**(cub_fit(np.log10(Tbar), popt[0], popt[1], popt[2], popt[3]))*Kbar

beta_univ_avg = np.zeros((2,1))
for i in range(0, num_rot_states):
    if ~np.isnan(Kloss[SrF][i]):
        beta_univ_avg[SrF][0] += stat_weights[i]*Kloss[SrF][i]
    if ~np.isnan(Kloss[CaF][i]):
        beta_univ_avg[CaF][0] += stat_weights[i]*Kloss[CaF][i]

beta_univ_avg[SrF][0] /= np.sum(stat_weights)
beta_univ_avg[CaF][0] /= np.sum(stat_weights)


beta_univ_avg[SrF][0] = np.format_float_scientific(beta_univ_avg[SrF][0], 2)
beta_univ_avg[CaF][0] = np.format_float_scientific(beta_univ_avg[CaF][0], 2)
print("Universal SrF loss rate coefficient at T=" + str(round(1e6*temp[SrF][0])) + " uK in cm^3/s:", beta_univ_avg[SrF][0])
print("Universal CaF loss rate coefficient at T=" + str(round(1e6*temp[CaF][0])) + " uK in cm^3/s:", beta_univ_avg[CaF][0])