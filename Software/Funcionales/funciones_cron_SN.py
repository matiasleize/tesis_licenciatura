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

from funciones_cambio_parametros import params_fisicos_to_modelo
from funciones_int import integrador
from funciones_taylor import Taylor_HS, Taylor_ST

from scipy.integrate import cumtrapz as cumtrapz
from scipy.interpolate import interp1d
from scipy.constants import c as c_luz #metros/segundos
c_luz_km = c_luz/1000 #km/seg
#ORDEN DE PRESENTACION DE LOS PARAMETROS: Mabs,omega_m,b,H_0,n

#Funciones de croómetros
def chi_2_cronometros(H_teo,H_data, dH):
    chi2 = np.sum(((H_data-H_teo)/dH)**2)
    return chi2

#Funciones de supernovas
def magn_aparente_teorica(zs, Hs, zcmb, zhel):
    '''A partir de un array de redshift y un array de la magnitud E = H_0/H
    que salen de la integración numérica, se calcula el mu teórico que deviene
    del modelo. muth = 25 + 5 * log_{10}(d_L),
    donde d_L = (c/H_0) (1+z) int(dz'/E(z'))'''

    d_c =  c_luz_km * cumtrapz(Hs**(-1), zs, initial=0)
    dc_int = interp1d(zs, d_c) #Interpolamos
    d_L = (1 + zhel) * dc_int(zcmb) #Obs, Caro multiplica por Zhel, con Zobs da un poquin mejor
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

#Juntamos todo
def params_to_chi2_taylor(theta, params_fijos, zcmb, zhel, Cinv,
                    mb,z_data_cron, H_data_cron, dH_cron, cantidad_zs=10000,
                     verbose=True,model='HS',M_fijo=False,chi_riess=True):
    '''Dados los parámetros del modelo devuelve un chi2 para los datos
    de supernovas. 1 parámetro fijo y 4 variables'''

    if len(theta)==4:
        [Mabs,omega_m,b,H_0] = theta
        n = params_fijos
        chi2_H0 = ((H_0-73.48)/1.66)**2
    elif len(theta)==3:
        if M_fijo == False:
            [Mabs,omega_m,b] = theta
            [H_0,n] = params_fijos
            chi2_H0 = 0
        else:
            [omega_m,b,H_0] = theta
            [Mabs,n] = params_fijos
            chi2_H0 = ((H_0-73.48)/1.66)**2

    #Cronometros:
    if model=='HS':
        H_modelo_cron = Taylor_HS(z_data_cron, omega_m, b, H_0)
    else:
        H_modelo_cron = Taylor_ST(z_data_cron, omega_m, b, H_0)


    if chi_riess==True:
        chi_cron = chi_2_cronometros(H_modelo_cron,H_data_cron,dH_cron)+chi2_H0
    else:
        chi_cron = chi_2_cronometros(H_modelo_cron,H_data_cron,dH_cron)

    #Supernovas:
    zs_SN = np.linspace(0,3,cantidad_zs)

    if model=='HS':
        H_modelo_SN = Taylor_HS(zs_SN, omega_m, b, H_0)
    else:
        H_modelo_SN = Taylor_ST(zs_SN, omega_m, b, H_0)

    muth = magn_aparente_teorica(zs_SN,H_modelo_SN,zcmb,zhel)
    chi_sn = chi_2_supernovas(muth,mb,Mabs,Cinv)

    return chi_cron+chi_sn

def params_to_chi2(theta, params_fijos, zcmb, zhel, Cinv,
                    mb,z_data_cron, H_data_cron, dH_cron, cantidad_zs=10000,
                     verbose=True,model='HS',M_fijo=False,chi_riess=True):
    '''Dados los parámetros del modelo devuelve un chi2 para los datos
    de supernovas. 1 parámetro fijo y 4 variables'''

    if len(theta)==4:
        [Mabs,omega_m,b,H_0] = theta
        n = params_fijos
        chi2_H0 = ((H_0-73.48)/1.66)**2
    elif len(theta)==3:
        if M_fijo == False:
            [Mabs,omega_m,b] = theta
            [H_0,n] = params_fijos
            chi2_H0 = 0
        else:
            [omega_m,b,H_0] = theta
            [Mabs,n] = params_fijos
            chi2_H0 = ((H_0-73.48)/1.66)**2
    if (0 <= b < 0.1):
        #Cronometros:
        if model=='HS':
            H_modelo_cron = Taylor_HS(z_data_cron, omega_m, b, H_0)
        else:
            H_modelo_cron = Taylor_ST(z_data_cron, omega_m, b, H_0)
    else:
        params_fisicos = [omega_m,b,H_0]
        z, H = integrador(params_fisicos, n)
        H_int = interp1d(z,H)
        H_modelo_cron = H_int(z_data_cron)


    if chi_riess==True:
        chi_cron = chi_2_cronometros(H_modelo_cron,H_data_cron,dH_cron)+chi2_H0
    else:
        chi_cron = chi_2_cronometros(H_modelo_cron,H_data_cron,dH_cron)

    #Supernovas:
    zs_SN = np.linspace(0,3,cantidad_zs)

    if (0 <= b < 0.1):
        if model=='HS':
            H_modelo_SN = Taylor_HS(zs_SN, omega_m, b, H_0)
        else:
            H_modelo_SN = Taylor_ST(zs_SN, omega_m, b, H_0)

    else:
        params_fisicos = [omega_m,b,H_0]
        z, H = integrador(params_fisicos, n)
        H_int = interp1d(z,H)
        H_modelo_SN = H_int(zs_SN)


    muth = magn_aparente_teorica(zs_SN,H_modelo_SN,zcmb,zhel)
    chi_sn = chi_2_supernovas(muth,mb,Mabs,Cinv)

    return chi_cron+chi_sn