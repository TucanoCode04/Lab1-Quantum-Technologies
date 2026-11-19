# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 13:26:59 2026

@author: PolitoStudents2026
"""

import pandas as pd
import numpy as np

# Data file & path
file = "C:\\Users\\PolitoStudents2026\\Desktop\\Group2\\Data_file.xlsx"

# Read data file
df = pd.read_excel(file, header=None)

# Exclude the first column (labels)
df = df.iloc[:, 1:]


attenuations = df.iloc[0, :].to_numpy()
measurements_P = df.iloc[1:11, :].to_numpy()*10**(-6)# P saved in [uW]
measurements_N = df.iloc[12:21, :].to_numpy()
P_env_vec = df.iloc[23:32, 0].to_numpy()*10**(-9)# P saved in [nW]
N_env_vec = df.iloc[34:43, 0].to_numpy()

#Constants
c = 3*10**8
h = 6.626*10**(-34)
lambdaa = 1550*10**(-9)
u_lambdaa = 10**(-10)
att_fix =46.5
t = 1
u_t = 1*10**(-3)
C = 1
u_C = 0.0017
lm = len(P_env_vec)

P_env = np.mean(P_env_vec)
u_Penv = np.std(P_env_vec)/ np.sqrt(lm)

N_env = np.mean(N_env_vec)
u_Nenv = np.std(N_env_vec)/ np.sqrt(lm)

