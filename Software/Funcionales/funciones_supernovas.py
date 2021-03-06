"""
Created on Sun Feb  2 13:28:48 2020

@author: matias
"""

import numpy as np

import sys
import os
from os.path import join as osjoin

from pc_path import definir_path
path_git, path_datos_global = definir_path()

os.chdir(path_git)
sys.path.append('./Software/Funcionales/')

from funciones_int import integrador
from funciones_taylor import Taylor_HS, Taylor_ST
from funciones_LambdaCDM import H_LCDM

from scipy.integrate import cumtrapz as cumtrapz
from scipy.interpolate import interp1d
from scipy.constants import c as c_luz #metros/segundos
c_luz_km = c_luz/1000; #km/seg
#ORDEN DE PRESENTACION DE LOS PARAMETROS: Mabs,omega_m,b,H_0,n



def magn_aparente_teorica(zs, Hs, zcmb, zhel):
    '''A partir de un array de redshift y un array de la magnitud E = H_0/H
    que salen de la integración numérica, se calcula el mu teórico que deviene
    del modelo. muth = 25 + 5 * log_{10}(d_L),
    donde d_L = (c/H_0) (1+z) int(dz'/E(z'))'''

    d_c =  c_luz_km * cumtrapz(Hs**(-1), zs, initial=0)
    dc_int = interp1d(zs, d_c) #Interpolamos
    d_L = (1 + zhel) * dc_int(zcmb) #Obs, Caro multiplica por Zhel, con Zcmb da un poquin mejor
    #Magnitud aparente teorica
    muth = 25.0 + 5.0 * np.log10(d_L)
    return muth

def chi_2_supernovas(muth,magn_aparente_obs,M_abs,C_invertida):
    '''Dado el resultado teórico muth y los datos de la
    magnitud aparente y absoluta observada, con su matriz de correlación
    invertida asociada, se realiza el cálculo del estadítico chi cuadrado.'''
    sn = len(muth)
    muobs =  magn_aparente_obs - M_abs
    deltamu = muobs - muth
    transp = np.transpose(deltamu)
    aux = np.dot(C_invertida,deltamu)
    chi2 = np.dot(transp,aux)
    return chi2

def params_to_chi2(theta, params_fijos, zcmb, zhel, Cinv,
                    mb,cantidad_zs=100000, verbose=True,model='HS',
                    taylor=False):
    '''Dados los parámetros del modelo devuelve un chi2 para los datos
    de supernovas. 1 parámetro fijo y 4 variables'''

    if len(theta)==4:
        [Mabs,omega_m,b,H_0] = theta
        n = params_fijos
    elif len(theta)==3:
        [Mabs,omega_m,b] = theta
        [H_0,n] = params_fijos

    if taylor == True:
        zs = np.linspace(0,3,cantidad_zs)
        if model=='HS':
            H_modelo = Taylor_HS(zs, omega_m, b, H_0)
        else:
            H_modelo = Taylor_ST(zs, omega_m, b, H_0)
    else:
        if (0 <= b < 0.2):
            zs = np.linspace(0,3,cantidad_zs)
            if model=='HS':
                H_modelo = Taylor_HS(zs, omega_m, b, H_0)
            else:
                H_modelo = Taylor_ST(zs, omega_m, b, H_0)
        else:
            params_fisicos = [omega_m,b,H_0]
            zs, H_modelo = integrador(params_fisicos, n, model=model)

    muth = magn_aparente_teorica(zs,H_modelo,zcmb,zhel)
    chi = chi_2_supernovas(muth,mb,Mabs,Cinv)
    return chi

def testeo_supernovas(theta, params_fijos, zcmb, zhel, Cinv,
                      mb0, x1, color, hmass, cantidad_zs=int(10**5),
                      model='HS',lcdm=False):
    '''Dados los parámetros del modelo devuelve un chi2 para los datos
    de supernovas. 1 parámetro fijo y 4 variables'''

    if lcdm == True:
        if len(theta)==5:
            [Mabs,omega_m,alpha,beta,gamma] = theta
            [H_0,n] = params_fijos
        if len(theta)==4:
            [Mabs,alpha,beta,gamma] = theta
            [omega_m,H_0,n] = params_fijos

        zs = np.linspace(0,3,cantidad_zs)
        H_modelo = H_LCDM(zs, omega_m, H_0)

    else:
        if len(theta)==7:
            [Mabs,omega_m,b,H_0,alpha,beta,gamma] = theta
            n = params_fijos
        elif len(theta)==6:
            [Mabs,omega_m,b,alpha,beta,gamma] = theta
            [H_0,n] = params_fijos
        elif len(theta)==4:
            [Mabs,alpha,beta,gamma] = theta
            [omega_m,b,H_0,n] = params_fijos

        if (0 <= b < 0.2):
            zs = np.linspace(0,3,cantidad_zs)
            if model=='HS':
                H_modelo = Taylor_HS(zs, omega_m, b, H_0)
            else:
                H_modelo = Taylor_ST(zs, omega_m, b, H_0)

        else:
            params_fisicos = [omega_m,b,H_0]
            zs, H_modelo = integrador(params_fisicos, n, cantidad_zs=cantidad_zs, model=model)

    alpha_0 = 0.154
    beta_0 = 3.02
    gamma_0 = 0.053
    mstep0 = 10.13
    #tau0 = 0.001

    #DeltaM_0=gamma_0*np.power((1.+np.exp((mstep0-hmass)/tau0)),-1)
    #DeltaM=gamma*np.power((1.+np.exp((mstep0-hmass)/tau0)),-1)
    #DeltaM_0 = gamma_0 * np.heaviside(hmass-mstep0, 1)
    #DeltaM = gamma * np.heaviside(hmass-mstep0, 1)

    muobs =  mb0 - Mabs + x1 * (alpha-alpha_0) - color * (beta-beta_0) + np.heaviside(hmass-mstep0, 1) * (gamma-gamma_0)

    muth = magn_aparente_teorica(zs,H_modelo,zcmb,zhel)

    deltamu = muobs - muth
    transp = np.transpose(deltamu)
    aux = np.dot(Cinv,deltamu)
    chi2 = np.dot(transp,aux)

    return chi2

#%%
if __name__ == '__main__':


    import numpy as np
    np.random.seed(42)
    from matplotlib import pyplot as plt

    import sys
    import os
    from os.path import join as osjoin
    from pc_path import definir_path
    path_git, path_datos_global = definir_path()
    os.chdir(path_git)
    sys.path.append('./Software/Funcionales/')
    from funciones_data import leer_data_pantheon_2
    from funciones_data import leer_data_pantheon
    #ORDEN DE PRESENTACION DE LOS PARAMETROS: Mabs,omega_m,b,H_0,n

    #%% Predeterminados:
    M_true = -19.2
    omega_m_true = 0.3
    b_true = 0.5
    H0_true =  73.48 #Unidades de (km/seg)/Mpc
    alpha_true = 0.154
    beta_true = 3.02
    gamma_true = 0.053
    n = 1

    params_fijos = [H0_true,n]

    #%%
    #Datos de SN
    os.chdir(path_git+'/Software/Estadística/Datos/Datos_pantheon/')

    _, zcmb, zhel, Cinv, mb0, x1, cor, hmass = leer_data_pantheon_2(
                'lcparam_full_long_zhel.txt','ancillary_g10.txt')
    zcmb_1,zhel_1, Cinv_1, mb_1 = leer_data_pantheon('lcparam_full_long_zhel.txt')

    #%%
    np.all(zhel_1==zhel)
    np.where(zcmb_1==zcmb)
    zcmb
    zcmb_1
    alpha_0=0.154
    beta_0=3.02
    gamma_0=0.053
    mstep0=10.13
    tau0=0.001
    from matplotlib import pyplot as plt
    plt.figure()
    plt.plot(hmass,gamma_0*np.power((1.+np.exp((mstep0-hmass)/tau0)),-1),'.')
    plt.plot(hmass,gamma_0*np.heaviside(hmass-mstep0, 1),'.')
    plt.grid(True)

    plt.figure()
    plt.plot(zcmb-zcmb_1,'.')
    plt.grid(True)
