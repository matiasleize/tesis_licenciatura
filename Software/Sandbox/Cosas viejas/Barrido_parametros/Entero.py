"""
Created on Sun Feb  2 13:02:59 2020

@author: matias
"""
import sys
import os
from os.path import join as osjoin
from pc_path import definir_path
path_git, path_datos_global = definir_path()

os.chdir(path_git)
sys.path.append('./Software/Funcionales/')

import numpy as np
from matplotlib import pyplot as plt

from funciones_int import integrador,magn_aparente_teorica
from funciones_data import leer_data_pantheon
from funciones_estadistica import chi_2

#%% Predeterminados:
H_0 =  73.48

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
#Parametros a ajustar
b = 1
d = 1/19 #(valor del paper)
c = 0.24
n = 1
Mabs=19
#%%
def params_to_chi2(params_modelo,Mabs):
    '''Dados los parámetros del modelo devuelve un chi2'''

    [b,c,d,r_0,n] = params_modelo

    def dX_dz(z, variables):
        x = variables[0]
        y = variables[1]
        v = variables[2]
        w = variables[3]
        r = variables[4]

        G = gamma(r,b,c,d,n)

        s0 = (-w + x**2 + (1+v)*x - 2*v + 4*y) / (z+1)
        s1 = - (v*x*G - x*y + 4*y - 2*y*v) / (z+1)
        s2 = -v * (x*G + 4 - 2*v) / (z+1)
        s3 = w * (-1 + x+ 2*v) / (z+1)
        s4 = -x*r*G/(1+z)
        return [s0,s1,s2,s3,s4]

    z,E = integrador(dX_dz,ci, params_modelo)
    os.chdir(path_git+'/Software/Estadística/')
    zcmb,zhel, Cinv, mb = leer_data_pantheon('lcparam_full_long_zhel.txt')
    muth = magn_aparente_teorica(z,E,zhel,zcmb)

    if isinstance(Mabs,list):
        chis_M = np.zeros(len(Mabs))
        for i,M_0 in enumerate(Mabs):
            chis_M[i] = chi_2(muth,mb,M_0,Cinv)
        return chis_M
    else:
        chi2 = chi_2(muth,mb,Mabs,Cinv)
        return chi2
#%% Con varios M
Ms = list(np.linspace(-19.6,-19,50))
params  = [b,c,d,r_0,n]
#chis = np.zeros(len(Ms))
chis = params_to_chi2(params,Ms)

#%%Con un M fijo
Mabs = -19.4
params  = [b,c,d,r_0,n]
chi = params_to_chi2(params,Mabs)
#%%
bs = list(np.linspace(0.5,1.5,40))
Ms = list(np.linspace(-19.6,-19,50))
chis = np.zeros((len(bs),len(Ms)))
for k,b0 in enumerate (bs):
    params  = [b0,c,d,r_0,n]
    chis[k,:] = params_to_chi2(params,Ms)
#%%
plt.pcolor(chis)
