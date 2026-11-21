# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 13:26:59 2026

@author: PolitoStudents2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from uncertainties import ufloat #for the uncertainty propagation
from scipy import odr #for the linear fit for eta_0
from scipy.constants import c,h #speed of light and planck constant

# Data file & path
file = "./Data_file.xlsx"
# Read data file
df = pd.read_excel(file, header=None)
# Exclude the first column (labels)
df = df.iloc[:, 1:]

var_a = ['N','N_env','P','P_env']

attenuations = df.iloc[0, :].to_numpy()
measurements_P = df.iloc[1:11, :].to_numpy()*10**(-6)# P saved in [uW]
measurements_N = df.iloc[11:21, :].to_numpy()
P_env_vec = df.iloc[22:32, 0].to_numpy()*10**(-9)# P saved in [nW]
N_env_vec = df.iloc[33:43, 0].to_numpy()

#Constants
lambdaa = 1550*10**(-9)
u_lambdaa = 10**(-10)
att_fix =46.5
u_att_fix = 0
u_att_var = 0
t = 1
u_t = 1*10**(-3)
C = 1
u_C = 0.0017
lm = len(P_env_vec)


#calculate the uncertainty of environmental Power
P_env = np.mean(P_env_vec)
u_Penv = np.std(P_env_vec,ddof=1) / np.sqrt(lm)

#calculate the uncertainty of environmental clicks
N_env = np.mean(N_env_vec)
u_Nenv = np.std(N_env_vec,ddof=1) / np.sqrt(lm)

#calculate the mean and u , for each set, of N and P
means_N = []
us_N = []
means_P = []
us_P = []
for i in range(6):
    N_values = measurements_N[:,i]
    P_values = measurements_P[:,i]

    means_N.append(np.mean(N_values))
    us_N.append(np.std(N_values,ddof=1) / np.sqrt(lm))

    means_P.append(np.mean(P_values))
    us_P.append(np.std(P_values,ddof=1) / np.sqrt(lm))

#calculate the efficiency for each set
def efficiency(N,N_env,P,P_env,att_var) -> float:
    costants = (h*c) / (lambdaa*t*C)
    att_tot = att_fix+att_var
    tau = 10**(-att_tot / 10)
    e = costants*((N-N_env) / (tau*(P-P_env)))
    return e

efficiencies = []
for i in range(6):
    e = efficiency(means_N[i],N_env,means_P[i],P_env,attenuations[i])
    efficiencies.append(e)

u_efficiencies = []
u_R = []
u_eta_A = []
u_eta_B = []
u_R_A = []
for i in range(6):
    N = ufloat(means_N[i],us_N[i], tag='N')
    Nenv = ufloat(N_env,u_Nenv, tag='N_env')
    Penv = ufloat(P_env,u_Penv, tag='P_env')
    P = ufloat(means_P[i],us_P[i], tag='P')
    C_formula = ufloat(C,u_C, tag='C')
    t_formula = ufloat(t,u_t, tag='t')
    lambda_formula = ufloat(lambdaa,u_lambdaa, tag='lambda')
    att_fix_formula = ufloat(att_fix,u_att_fix, tag='att_fix')
    att_var = ufloat(attenuations[i],u_att_var, tag='att_var')
    tau = 10**(-(att_fix_formula+att_var) / 10)

    eff = ( h*c * (N - Nenv)) / (lambda_formula*t_formula*tau*C_formula*(P - Penv))
    R = N / t_formula
    R_A = N / t
    u_R_A.append(R_A.std_dev)
    u_R.append(R.std_dev)
    u_efficiencies.append(eff.std_dev)

    sum_a = 0
    sum_b = 0
    for var, contrib in eff.error_components().items():
        #print(f"  {var.tag:8s}: {contrib:.6f}")
        if(var.tag in var_a):
            sum_a += contrib**2
        else:
            sum_b += contrib**2
    u_eta_A.append(np.sqrt(sum_a))
    u_eta_B.append(np.sqrt(sum_b))


#fit 
def line(beta,x):
    m,q = beta
    return m*x + q

model = odr.Model(line)
data = odr.RealData(means_N,efficiencies,sx=u_R_A,sy=u_eta_A)
fit = odr.ODR(data,model,beta0=[0.0,np.mean(efficiencies)])
out = fit.run()

m,eta_0 = out.beta
u_m, u_eta_0_A = out.sd_beta
u_tau = (np.log(10) / 10) * np.sqrt(u_att_fix**2 + u_att_var**2)
u_eta_0_B = eta_0 * np.sqrt((u_lambdaa/lambdaa)**2 + (u_t / t)**2 + (u_tau/tau.nominal_value)**2 + (u_C/C)**2)




#plotting
x_fit = np.linspace(0, max(means_N) * 1.05, 200)
y_fit = line(out.beta, x_fit)

plt.figure(figsize=(8, 5))

plt.errorbar(means_N, efficiencies, xerr=u_R_A, yerr=u_eta_A, fmt='o', capsize=4, label='Experimental Data (Type A)')
plt.plot(x_fit, y_fit, '-', color='red', label=fr'Fit ODR: $\eta_0 = {eta_0:.4f} \pm {u_eta_0_A:.4f}$')

plt.errorbar(0, eta_0, yerr=u_eta_0_A, fmt='s', color='green', capsize=4, label=r'Zero-flux Efficiency $\eta_0$')

plt.xlabel('Count rate $N/t$ [Hz]')
plt.ylabel(r'Detection efficiency $\eta$')
plt.title('SPAD Detection Efficiency vs Count Rate (ODR Fit)')
plt.xlim(left=-500)  # left margin to see the intercept at 0
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig('zero_flux_efficiency_fit.png')
#plt.show()






