"""
Created on Wed Feb  5 16:07:35 2020

@author: matias
"""
import sys
import os
from os.path import join as osjoin
from scipy.interpolate import interp1d
from scipy.constants import c as c_luz #metros/segundos
path_git = '/home/matias/Documents/Tesis/tesis_licenciatura'

path_datos_global = '/home/matias/Documents/Tesis/'

os.chdir(path_git)
sys.path.append('./Software/Funcionales/')
#sys.path.append('/Software/Estadística/')


import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize
import emcee
import corner

from funciones_int import integrador
from funciones_data import leer_data_cronometros
from funciones_cronometros import chi_2_cronometros

import sys
import os
from os.path import join as osjoin
path_git = '/home/matias/Documents/Tesis/tesis_licenciatura'

path_datos_global = '/home/matias/Documents/Tesis/'

os.chdir(path_git)
#sys.path.append('./Software/Funcionales/')
#sys.path.append('/Software/Estadística/')
#%% Predeterminados:
H_0 =  73.48 #Unidades de km/seg/Mpc
n = 1
#Gamma de Lucila (define el modelo)
gamma = lambda r,b,c,d,n: ((1+d*r**n) * (-b*n*r**n + r*(1+d*r**n)**2)) / (b*n*r**n * (1-n+d*(1+n)*r**n))
#Coindiciones iniciales e intervalo
x_0 = -0.339
y_0 = 1.246
v_0 = 1.64
w_0 = 1+x_0+y_0-v_0
r_0 = 41

ci = [x_0, y_0, v_0, w_0, r_0] #Condiciones iniciales
zi = 0
zf = 3 # Es un valor razonable con las SN 1A
#%%


cs_to_r_hs  = lambda c1,c2,omega_m: 6*(1-omega_m)*c2/c1

def params_fisicos_to_modelo(omega_m, b,n=1):
    '''Toma los parametros fisicos (omega_m y el parametro de distorsión b)
    y devuelve los parametros del modelo c1, c2 y R_HS'''
    h0 = H_0/100
    alpha = ((c_luz/100)/8315)**2
    beta = 1 / 0.13
    aux = 6 * (1 - omega_m) / (alpha * (omega_m - beta * h0 ** 2))
    c_2 = (aux * b / 2) ** n
    c_1 = aux * c_2
    r_hs = cs_to_r_hs(c_1,c_2,omega_m)
    return c_1, c_2, r_hs

#%%

os.chdir(path_git+'/Software/Estadística/Datos/')
z_data, H_data, dH  = leer_data_cronometros('datos_cronometros.txt')

def params_to_chi2(theta,params_fijos):
    '''Dados los parámetros del modelo devuelve un chi2 para los datos de los
    cronómetros cósmicos'''

    [omega_m,b] = theta
    [r_0,n] = params_fijos

    ## Transformo los parametros fisicos en los del modelo:
    c1,c2,r_hs = params_fisicos_to_modelo(omega_m,b)

    params_modelo = [c1,r_hs,c2,r_0,n]#de la cruz: [b,c,d,r_0,n]

    def dX_dz(z, variables):
        x = variables[0]
        y = variables[1]
        v = variables[2]
        w = variables[3]
        r = variables[4]

        G = gamma(r,c1,r_hs,c2,n)

        s0 = (-w + x**2 + (1+v)*x - 2*v + 4*y) / (z+1)
        s1 = - (v*x*G - x*y + 4*y - 2*y*v) / (z+1)
        s2 = -v * (x*G + 4 - 2*v) / (z+1)
        s3 = w * (-1 + x+ 2*v) / (z+1)
        s4 = -x*r*G/(1+z)
        return [s0,s1,s2,s3,s4]

    z,E = integrador(dX_dz,ci, params_modelo)
    H_int = interp1d(z,E)
    H_teo = H_0 * H_int(z_data)

    chi = chi_2_cronometros(H_teo,H_data,dH)
    return chi

b_true = 0.22
omega_m_true = 0.26

np.random.seed(42)

log_likelihood = lambda theta: -0.5 * params_to_chi2(theta, params_fijos=[r_0,n])
nll = lambda *args: -log_likelihood(*args)
initial = np.array([omega_m_true,b_true]) + 0.1 * np.random.randn(2)
soln = minimize(nll, initial)#,bounds =([0,1],[-0.5,0.5]))
omega_m_ml, b_ml = soln.x

print(omega_m_ml,b_ml)

# Guardamos los datos

os.chdir(path_git)
np.savez('Software/Estadística/Resultados_simulaciones//MCMC cronometros/mean_cron_om_b'
         , omega_m_ml = omega_m_ml, b_ml = b_ml)
