import numpy as np
import emcee
from matplotlib import pyplot as plt
import corner
import sys
import os
import time

from pc_path import definir_path
path_git, path_datos_global = definir_path()
os.chdir(path_git)
sys.path.append('./Software/Funcionales/')
from funciones_analisis_cadenas import graficar_cadenas,graficar_contornos,graficar_taus_vs_n
#%%
os.chdir(path_git+'/Software/Estadística/Resultados_simulaciones/')
with np.load('valores_medios_HS_CC+SN+BAO_3params_taylor.npz') as data:
    sol = data['sol']
#%%
os.chdir(path_datos_global+'/Resultados_cadenas')
filename = "sample_HS_CC+SN+BAO_3params_taylor.h5"
reader = emcee.backends.HDFBackend(filename)
# Algunos valores
tau = reader.get_autocorr_time()
burnin = int(2 * np.max(tau))
thin = int(0.5 * np.min(tau))
samples = reader.get_chain(discard=burnin, flat=True, thin=thin)
print(tau)
#%%
%matplotlib qt5
graficar_cadenas(reader,
                labels = ['$M_{abs}$','$\Omega_{m}$','b'],title='SN+BAO+CC HS (Taylor)')
#%%
#burnin=30
burnin = int(2 * np.max(tau))
thin = int(0.5 * np.min(tau))
graficar_contornos(reader,params_truths=sol,discard=burnin,#thin=thin
                    labels= ['$M_{abs}$','$\Omega_{m}$','b'],title='SN+BAO+CC HS (Taylor)')
#%%
plt.figure()
graficar_taus_vs_n(reader,num_param=0)
graficar_taus_vs_n(reader,num_param=1)
graficar_taus_vs_n(reader,num_param=2)
