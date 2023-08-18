'''
cd /Users/garethjones/Desktop/BEAGLE_FIR-main
conda activate py39
python FullRadioFIR_MAGPHYS_PMN.py 
'''

import numpy as np
from scipy.integrate import trapezoid
import random as rnd
import cosmocalc2 as CC
import matplotlib.pyplot as plt
import os
import pymultinest
import scipy
from uncertainties import unumpy as unp
import sys
from astropy.io import fits

###
#Define constants
h=6.6261e-27 #erg * sec
kbb=1.3807e-16 #erg / K
cspl=2.99792458e+10 #cm/s
TCMB=2.73
cosmoparams=[70,0.3,0.7]

klambda_o = 0.77; lambda_ok = (1E+2)*850E-6

colors=["#ff5da3",
"#c7ce00",
"#014ca9",
"#99d772",
"#ffa181",
"#40dadd",
"#2d4900"]

import matplotlib

fs=12
font = {'family' : 'sans-serif','weight' : 'bold','size' : fs}
matplotlib.rc('font', **font)

#Modified blackbody (optically thin)
def MBB_simple(lambda_o, td, beta):
	lambda_o=np.array(lambda_o)
	MBB = ((2.*h*(cspl**2.))/(lambda_o**5.)) * (1./(np.exp((h*cspl)/(lambda_o*kbb*td))-1.))

	kappa_nu=klambda_o*((lambda_ok/lambda_o)**beta)
	final_MBB = kappa_nu * MBB
	#Normalize
	MBB_sum = np.trapz(final_MBB,lambda_o)
	for mbbi in range(len(final_MBB)):
		final_MBB[mbbi]/=MBB_sum
	return final_MBB

#Priors
def UniDis(ZO,a,b):
	return ZO*(b-a)+a
def LogUniDis(ZO,a,b):
	return 10**(ZO*(np.log10(b)-np.log10(a))+np.log10(a))
def prior(cube, ndim, nparams):
	howmany=0
	for ipar in range(len(Parameter)):
		if Parameter[ipar]['vary']:
			if ('xi_' in Parameter[ipar]['Name']) and ('BC' in Parameter[ipar]['Name']):
				cube[howmany] = UniDis(cube[howmany],0,1)
				howmany=howmany+1
				cube[howmany] = UniDis(cube[howmany],0,1-cube[howmany-1])
				howmany=howmany+1
				ipar+=1
			elif ('xi_' in Parameter[ipar]['Name']) and ('ISM' in Parameter[ipar]['Name']):
				cube[howmany] = UniDis(cube[howmany],0,1)
				howmany=howmany+1
				cube[howmany] = UniDis(cube[howmany],0,1-cube[howmany-1])
				howmany=howmany+1
				cube[howmany] = UniDis(cube[howmany],0,1-cube[howmany-1]-cube[howmany-2])
				howmany=howmany+1
				ipar+=3
			elif Parameter[ipar]['Name']=='tauv':
				cube[howmany] = 1. - np.tanh(1.5*cube[howmany] - 6.7)
				howmany+=1
			elif Parameter[ipar]['Name']=='mu':
				cube[howmany] = 1 - np.tanh(8*cube[howmany] - 6)
				howmany+=1
			else:
				cube[howmany] = UniDis(cube[howmany],Parameter[ipar]['low_hi'][0],Parameter[ipar]['low_hi'][1])
				howmany=howmany+1

#Likelihood function
def loglike(cube,ndim,nparams):
	totdif=0.0
	howmany=0
	Param_vals=[]
	for ipar in range(len(Parameter)):
		if Parameter[ipar]['vary']:
			Param_vals.append(cube[howmany])
			howmany=howmany+1
		else:
			Param_vals.append(Parameter[ipar]['val0'])
	Fvi_full=Full_SED(test_x_cm, Param_vals)
	for i in range(len(test_x_cm)):
		Fvi=Fvi_full[i]
		Di=test_y[i]
		sigi=test_dy[i]
		totdif+=(((Di-Fvi)/sigi)**2+np.log(2*np.pi*(sigi**2)))
	return (-1/2)*totdif

#Parameter values (Change!)
Parameter=[]
Parameter.append({'Name':'T_MIR_1', 'val0':220., 'low_hi':[200,300], 'vary':False})
Parameter.append({'Name':'T_MIR_2', 'val0':130., 'low_hi':[80,180], 'vary':False})
Parameter.append({'Name':'T_W_BC', 'val0':48., 'low_hi':[38,58], 'vary':False})
Parameter.append({'Name':'T_C_ISM', 'val0':22., 'low_hi':[12,32], 'vary':False})
Parameter.append({'Name':'T_W_ISM', 'val0':45., 'low_hi':[35,55], 'vary':False})
Parameter.append({'Name':'xi_MIR_BC', 'val0':0.15, 'low_hi':[0,1], 'vary':False})
Parameter.append({'Name':'xi_PAH_BC', 'val0':0.05, 'low_hi':[0,1], 'vary':False})
Parameter.append({'Name':'xi_W_ISM', 'val0':0.035, 'low_hi':[0,1], 'vary':False})
Parameter.append({'Name':'xi_MIR_ISM', 'val0':0.055, 'low_hi':[0,1], 'vary':False})
Parameter.append({'Name':'xi_PAH_ISM', 'val0':0.11, 'low_hi':[0,1], 'vary':False})
#Parameter.append({'Name':'f_mu', 'val0':0.6, 'low_hi':[0,1], 'vary':False})
#Parameter.append({'Name':'LdTot', 'val0':0.8E-15, 'low_hi':[1E-17,1E-14], 'vary':False})
Parameter.append({'Name':'tauv', 'val0':1.0, 'low_hi':[0,1], 'vary':False})
Parameter.append({'Name':'mu', 'val0':0.4, 'low_hi':[0,1], 'vary':False})


#Get tau values
def tau_BC_lambda(temp_lamb,mu,tauv):
	return (1-mu)*tauv*((temp_lamb/5500E-8)**-1.3)
def tau_ISM_lambda(temp_lamb,mu,tauv):
	return mu*tauv*((temp_lamb/5500E-8)**-0.7)

#Full model
def Full_SED(test_x_cm, Param_vals):

	##Basic values
	#Two MIR temperatures
	T_MIR_1 = Param_vals[0]
	T_MIR_2 = Param_vals[1]
	T_W_BC = Param_vals[2]
	T_C_ISM = Param_vals[3]
	T_W_ISM = Param_vals[4]
	xi_BC=[Param_vals[6], Param_vals[5], 1-Param_vals[5]-Param_vals[6]]
	xi_ISM=[Param_vals[9], Param_vals[8] ,Param_vals[7], 1-Param_vals[9]-Param_vals[8]-Param_vals[7]]
	#f_mu = Param_vals[10]
	#LdTot = Param_vals[11]
	tauv = Param_vals[10]
	mu = Param_vals[11]

	#Get L_d_BC_tot & L_d_ISM_tot
	L_d_BC_tot=0.
	for itot in range(len(test_x_cm)):
		taubc = tau_BC_lambda(test_x_cm[itot],mu,tauv)
		if itot==0:
			tempdel = abs(test_x_cm[1]-test_x_cm[0])
		else:
			tempdel = abs(test_x_cm[itot]-test_x_cm[itot-1])
		L_d_BC_tot+=4.*np.pi*(1.-np.exp(-1*taubc))*young_unatten_y[itot]*tempdel
	L_d_ISM_tot=0.
	for itot in range(len(test_x_cm)):
		taubc = tau_BC_lambda(test_x_cm[itot],mu,tauv)
		tauism = tau_ISM_lambda(test_x_cm[itot],mu,tauv)
		if itot==0:
			tempdel = abs(test_x_cm[1]-test_x_cm[0])
		else:
			tempdel = abs(test_x_cm[itot]-test_x_cm[itot-1])
		L_d_ISM_tot+=4.*np.pi*(1.-np.exp(-1*tauism)) * (old_unatten_y[itot] + np.exp(-1*taubc)*young_unatten_y[itot])*tempdel
	LdTot = L_d_BC_tot+L_d_ISM_tot
	f_mu = L_d_ISM_tot/LdTot

	#Get MBBs
	l_lambda_MIR_1 = MBB_simple(test_x_cm, T_MIR_1, 1.5)
	l_lambda_MIR_2 = MBB_simple(test_x_cm, T_MIR_2, 1.5)
	l_lambda_MIR = np.zeros(len(l_lambda_MIR_2))
	for miri in range(len(test_x_cm)):
		l_lambda_MIR[miri] = l_lambda_MIR_1[miri] + l_lambda_MIR_2[miri]
	MIR_SUM = np.trapz(l_lambda_MIR,test_x_cm)
	for mbbi in range(len(l_lambda_MIR)):
		l_lambda_MIR[mbbi]/=MIR_SUM
	#
	l_lambda_TWBC = MBB_simple(test_x_cm, T_W_BC, 1.5)
	l_lambda_TWISM = MBB_simple(test_x_cm, T_W_ISM, 1.5)
	l_lambda_TCISM = MBB_simple(test_x_cm, T_C_ISM, 2.0)

	BC_PAH=np.zeros(len(test_x_cm)); BC_MIR=np.zeros(len(test_x_cm)); BC_W=np.zeros(len(test_x_cm))
	ISM_PAH=np.zeros(len(test_x_cm)); ISM_MIR=np.zeros(len(test_x_cm)); ISM_W=np.zeros(len(test_x_cm)); ISM_C=np.zeros(len(test_x_cm))
	Young_Stellar=np.zeros(len(test_x_cm)); Old_Stellar=np.zeros(len(test_x_cm))
	full_sed=np.zeros(len(test_x_cm))
	for fsi in range(len(test_x_cm)):
		BC_PAH[fsi] = LdTot*(1-f_mu)*xi_BC[0]*l_lambda_PAH_interp[fsi]
		BC_MIR[fsi] = LdTot*(1-f_mu)*xi_BC[1]*l_lambda_MIR[fsi]
		BC_W[fsi]   = LdTot*(1-f_mu)*xi_BC[2]*l_lambda_TWBC[fsi]
		ISM_PAH[fsi] = LdTot*f_mu*xi_ISM[0]*l_lambda_PAH_interp[fsi]
		ISM_MIR[fsi] = LdTot*f_mu*xi_ISM[1]*l_lambda_MIR[fsi]
		ISM_W[fsi]   = LdTot*f_mu*xi_ISM[2]*l_lambda_TWISM[fsi]
		ISM_C[fsi]   = LdTot*f_mu*xi_ISM[3]*l_lambda_TCISM[fsi]
		ISM_C[fsi]   = LdTot*f_mu*xi_ISM[3]*l_lambda_TCISM[fsi]
		Young_Stellar[fsi] = young_unatten_y[fsi]
		Old_Stellar[fsi] = old_unatten_y[fsi]
		full_sed[fsi] = BC_PAH[fsi] + BC_MIR[fsi] + BC_W[fsi] + ISM_PAH[fsi] + ISM_MIR[fsi] + ISM_W[fsi] + ISM_C[fsi] + Young_Stellar[fsi] + Old_Stellar[fsi]

	return test_x_AA*full_sed

#----------------

#Get basic grid
print('Making grid...')
test_x_log_lims=[-4,6]
test_x = np.arange(test_x_log_lims[0], test_x_log_lims[1], 0.001)
test_x = [10**x for x in test_x]
test_y=np.ones(len(test_x)); test_dy=[0.1 for x in test_y] #Arbitrary
test_x_cm=np.zeros(len(test_x)); test_x_um=np.zeros(len(test_x)); test_x_AA=np.zeros(len(test_x))
for lli in range(len(test_x)):
	test_x_cm[lli]=(1E-4)*test_x[lli]
	test_x_um[lli]=test_x[lli]
	test_x_AA[lli]=(1E+4)*test_x[lli]

#Read the file that contains the PAH template (M17) + hot NIR continuum
print('Getting PAH + NIR template...')
lampah=[]; l_lambda_PAH=[]
f_pahs_nir=open('pahs_nir.dat','r')
ff_pahs_nir=f_pahs_nir.readlines()
for i in range(len(ff_pahs_nir)):
	temp_pahs=ff_pahs_nir[i].split()
	lampah.append((1E-4)*float(temp_pahs[0])) #cm
	l_lambda_PAH.append(float(temp_pahs[1]))
#Interpolate to test_x_cm
l_lambda_PAH_interp = np.interp(test_x_cm, lampah, l_lambda_PAH)	
#Get rid of bad values
for i in range(len(test_x_cm)):
	if test_x_cm[i]<min(lampah) or test_x_cm[i]>max(lampah):
		l_lambda_PAH_interp[i]=0.	
#Convert from fnu to flambda
for mbbi in range(len(test_x_cm)):
	l_lambda_PAH_interp[mbbi]*=(cspl/(test_x_cm[mbbi]**2)) * (1E-10)
#Normalize PAH template
norm_pah=np.trapz(l_lambda_PAH_interp, test_x_cm)
for i in range(len(test_x_cm)):
	l_lambda_PAH_interp[i]/=norm_pah

#Get stellar models
print('Getting young stellar template...')
hdul = fits.open('young_unatten.fits')
data = hdul[1].data	
young_unatten_x=[]; young_unatten_y=[]
for flj in range(len(data)):
	young_unatten_x.append(data[flj][0]*(1E-8)) #cm
	young_unatten_y.append(data[flj][1])
#Interpolate to test_x_cm
young_unatten_y = np.interp(test_x_cm, young_unatten_x, young_unatten_y)	
#Get rid of bad values
for i in range(len(test_x_cm)):
	if test_x_cm[i]<min(young_unatten_x) or test_x_cm[i]>max(young_unatten_x):
		young_unatten_y[i]=0.	
#
print('Getting old stellar template...')
hdul = fits.open('old_unatten.fits')
data = hdul[1].data	
old_unatten_x=[]; old_unatten_y=[]
for flj in range(len(data)):
	old_unatten_x.append(data[flj][0]*(1E-8)) #cm
	old_unatten_y.append(data[flj][1])
#Interpolate to test_x_cm
old_unatten_y = np.interp(test_x_cm, old_unatten_x, old_unatten_y)	
#Get rid of bad values
for i in range(len(test_x_cm)):
	if test_x_cm[i]<min(old_unatten_x) or test_x_cm[i]>max(old_unatten_x):
		old_unatten_y[i]=0.	

#Get array of fit variable names
fit_parameters=[]
for ipar in range(len(Parameter)):
	if Parameter[ipar]['vary']:
		fit_parameters.append(Parameter[ipar]['Name'])
n_params = len(fit_parameters)
datafile='TEST0_SED'

#Run MultiNest
pymultinest.run(loglike, prior, n_params, outputfiles_basename=datafile+'_1_',resume=False,verbose=True,max_iter=0)

#Get best-fit values and uncertainties
skippy=2*(n_params+3)-2
N1=np.genfromtxt(datafile+'_1_stats.dat',skip_header=4,skip_footer=skippy,delimiter='   ')
Best_Values=[]; Best_Errors=[]
try:
	howmany=0
	for ipar in range(len(Parameter)):
		if Parameter[ipar]['vary']:
			Best_Values.append(N1[howmany,1])
			Best_Errors.append(N1[howmany,2])
			howmany+=1
		else:
			Best_Values.append(Parameter[ipar]['val0'])
			Best_Errors.append(-1)
except IndexError: #In case of only one free variable...
	for ipar in range(len(Parameter)):
		if Parameter[ipar]['vary']:
			Best_Values.append(N1[1])
			Best_Errors.append(N1[2])
		else:
			Best_Values.append(Parameter[ipar]['val0'])
			Best_Errors.append(-1)
print(Best_Values);print(Best_Errors)

full_sed = Full_SED(test_x_cm,Best_Values)

fig = plt.figure(figsize=(7,6))
ax = fig.add_subplot(1, 1, 1)
ax.set_xscale('log'); ax.set_yscale('log') 
#
#ax.plot(test_x_um, test_y, color='k', linestyle='dashed')
ax.plot(test_x_um, full_sed, c='k', label='Full Model',lw=2)
#
xl=[10**test_x_log_lims[0], 10**test_x_log_lims[1]]
ax.set_xlim(xl[0],xl[1])
ax.set_xlabel('Observed Wavelength [um]',weight='bold')
ax.set_ylim(10**-11.5,10**-6.9)
ax.set_ylabel(r'$\lambda$ F$_{\lambda}$ [erg cm$^{-2}$ s$^{-1}$]',weight='bold')
plt.legend(fontsize=8)
def conversion_func(x):
	return (1E-9)*cspl*(1E-2)/(x*1E-6)
ticks_x = np.logspace(np.log10(xl[0]), np.log10(xl[1]), int(np.log10(xl[1]) + 1))  # must span limits of first axis with clever spacing
ticks_z = conversion_func(ticks_x)
ax2 = ax.twiny()  # get the twin axis
ax2.semilogx(ticks_z, np.ones_like(ticks_z), alpha=0)  # transparent dummy plot
ax2.set_xlim(ticks_z[0], ticks_z[-1])
ax2.set_xlabel('Observed Frequency [GHz]',weight='bold')
plt.show()



'''
ax.plot(lambda_list_um, lambda_list_AA*ISM_PAH, label='ISM PAH + NIR',c=colors[0])
ax.plot(lambda_list_um, lambda_list_AA*ISM_MIR, label='ISM MIR',c=colors[1])
ax.plot(lambda_list_um, lambda_list_AA*ISM_W, label='ISM Warm',c=colors[2])
ax.plot(lambda_list_um, lambda_list_AA*ISM_C, label='ISM Cold',c=colors[4])
ax.plot(lambda_list_um, lambda_list_AA*BC_PAH, label='BC PAH',c=colors[0],linestyle='dashed')
ax.plot(lambda_list_um, lambda_list_AA*BC_MIR, label='BC MIR',c=colors[1],linestyle='dashed')
ax.plot(lambda_list_um, lambda_list_AA*BC_W, label='BC Warm',c=colors[2],linestyle='dashed')
'''