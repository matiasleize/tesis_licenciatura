"""
Created on Wed Feb  5 16:07:35 2020

@author: matias
"""
import numpy as np
from scipy.optimize import minimize
np.random.seed(42)

import sys
import os
from os.path import join as osjoin
from pc_path import definir_path
path_git, path_datos_global = definir_path()
os.chdir(path_git)
sys.path.append('./Software/Funcionales/')
from funciones_data import leer_data_cronometros
from funciones_cronometros import  params_to_chi2


#ORDEN DE PRESENTACION DE LOS PARAMETROS: Mabs,omega_m,b,H_0,n
#%% Predeterminados:
n = 2
omega_m_true = 0.3
b_true = 2
H0_true =  73.48 #Unidades de (km/seg)/Mpc
#%%

os.chdir(path_git+'/Software/Estadística/Datos/')
z_data, H_data, dH  = leer_data_cronometros('datos_cronometros.txt')


nll = lambda theta: params_to_chi2(theta,n,z_data,H_data,dH)
initial = np.array([omega_m_true,b_true,H0_true])
bnds = ((0.2, 0.4), (0,3),(68,80))
soln = minimize(nll, initial, bounds=bnds)
omega_m_ml, b_ml, H0_ml = soln.x
print(omega_m_ml,b_ml,H0_ml)

os.chdir(path_git + '/Software/Estadística/Resultados_simulaciones')
np.savez('valores_medios_ST_CC+H0_3params', sol=soln.x)
soln.fun/(len(z_data)+1-3) #0.5277728019958039
#%%
os.chdir(path_git+'/Software/Estadística/Resultados_simulaciones/')
with np.load('valores_medios_ST_CC+H0_3params') as data:
    sol = data['sol']

sol
#Tardo 58h 28 min 17 seg!
