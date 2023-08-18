

import numpy as np
from scipy.integrate import trapezoid
import random as rnd
import cosmocalc2 as CC
import matplotlib.pyplot as plt

###
#Define constants
h=6.6261e-27 #erg * sec
kbb=1.3807e-16 #erg / K
m_sun=1.989e+33 #g
l_sun=3.826e+33 #erg / s
tbg3=45.
cspl=2.99792458e+10 #cm/s
knu=0.77 #dust emissivity: cm^2 / g (Dunne & al, for 850um)
nu_o=(cspl*1E-2)/(850E-6) #Hz
npoints=6450
nmod=50 #nmod=50000
TCMB=2.73
cosmoparams=[70,0.3,0.7]

cspl_m=cspl*(1E-2)

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

def MBB_simple(lambda_o, td, beta):
	lambda_o=np.array(lambda_o)
	MBB = ((2.*h*(cspl**2.))/(lambda_o**5.)) * (1./(np.exp((h*cspl)/(lambda_o*kbb*td))-1.))
	klambda_o = 0.77; lambda_ok = (1E+2)*850E-6
	kappa_nu=klambda_o*((lambda_ok/lambda_o)**beta)
	final_MBB = kappa_nu * MBB
	#Normalize
	MBB_sum = np.trapz(final_MBB,lambda_o)
	for mbbi in range(len(final_MBB)):
		final_MBB[mbbi]/=MBB_sum
	return final_MBB

def get_firsed2(lambda_list):

	##Basic values
	#Two MIR temperatures
	T_MIR_1=250. #dacu08
	T_MIR_2=130. #dacu08
	T_W_BC=48. #dacu08
	T_C_ISM=22. #dacu08
	T_W_ISM=45. #dacu08
	#Relative contributions to BC luminosity
	xi_BC=[0.05,0.15,0.80] #dacu08
	#np.random.dirichlet(np.ones(3),size=1)[0]
	#xi_BC[2]=rnd.random() #xi_W^BC
	#xi_BC[1]=(1.-xi_BC[2])*rnd.random() #xi_MIR^BC
	#xi_BC[0]=1.-xi_BC[2]-xi_BC[1] #xi_PAH^BC
	#
	#Relative contributions to ISM luminosity
	xi_ISM=[-1,-1,-1,-1]
	xi_ISM[3]=0.8 #dacu08 #xi_C^ISM
	xi_ISM[2]=0.175*(1.-xi_ISM[3]) #dacu08 #xi_W^ISM
	xi_ISM[1]=0.275*(1.-xi_ISM[3]) #dacu08 #xi_MIR^ISM
	xi_ISM[0]=0.550*(1.-xi_ISM[3]) #dacu08 #xi_PAH^ISM
	#
	#Fraction of IR luminosity contributed by ISM
	f_mu=0.6 #dacu08
	#
	LdTot = 0.8E-15 #(Arbitrary!)

	#Read the file that contains the PAH template (M17) + hot NIR continuum
	nupah=[]
	lampah=[]
	l_lambda_PAH=[]
	f_pahs_nir=open('pahs_nir.dat','r')
	ff_pahs_nir=f_pahs_nir.readlines()
	for i in range(npoints):
		temp_pahs=ff_pahs_nir[i].split()
		nupah.append(cspl/((1E-4)*float(temp_pahs[0]))) #Hz
		lampah.append((1E-4)*float(temp_pahs[0])) #cm
		l_lambda_PAH.append(float(temp_pahs[1]))
	#Interpolate to lambda_list
	l_lambda_PAH_interp = np.interp(lambda_list, lampah, l_lambda_PAH)
	#Get rid of bad values
	for i in range(len(lambda_list)):
		if lambda_list[i]<min(lampah) or lambda_list[i]>max(lampah):
			l_lambda_PAH_interp[i]=0.	
	#Convert from fnu to flambda
	for mbbi in range(len(lambda_list)):
		l_lambda_PAH_interp[mbbi]*=(cspl/(lambda_list[mbbi]**2)) * (1E-10)
	#Normalize PAH template
	norm_pah=np.trapz(l_lambda_PAH_interp, lambda_list)
	for i in range(len(lambda_list)):
		l_lambda_PAH_interp[i]/=norm_pah

	#Get MBBs
	l_lambda_MIR_1 = MBB_simple(lambda_list, T_MIR_1, 1.5)
	l_lambda_MIR_2 = MBB_simple(lambda_list, T_MIR_2, 1.5)
	l_lambda_MIR = np.zeros(len(lambda_list))
	for miri in range(len(lambda_list)):
		l_lambda_MIR[miri] = l_lambda_MIR_1[miri] + l_lambda_MIR_2[miri]
	MIR_SUM = np.trapz(l_lambda_MIR,lambda_list)
	for mbbi in range(len(l_lambda_MIR)):
		l_lambda_MIR[mbbi]/=MIR_SUM
	#
	l_lambda_TWBC = MBB_simple(lambda_list, T_W_BC, 1.5)
	l_lambda_TWISM = MBB_simple(lambda_list, T_W_ISM, 1.5)
	l_lambda_TCISM = MBB_simple(lambda_list, T_C_ISM, 2.0)

	BC_PAH=np.zeros(len(lambda_list)); BC_MIR=np.zeros(len(lambda_list)); BC_W=np.zeros(len(lambda_list))
	ISM_PAH=np.zeros(len(lambda_list)); ISM_MIR=np.zeros(len(lambda_list)); ISM_W=np.zeros(len(lambda_list)); ISM_C=np.zeros(len(lambda_list))
	full_sed=np.zeros(len(lambda_list))
	for fsi in range(len(lambda_list)):
		BC_PAH[fsi] = LdTot*(1-f_mu)*xi_BC[0]*l_lambda_PAH_interp[fsi]
		BC_MIR[fsi] = LdTot*(1-f_mu)*xi_BC[1]*l_lambda_MIR[fsi]
		BC_W[fsi]   = LdTot*(1-f_mu)*xi_BC[2]*l_lambda_TWBC[fsi]
		ISM_PAH[fsi] = LdTot*f_mu*xi_ISM[0]*l_lambda_PAH_interp[fsi]
		ISM_MIR[fsi] = LdTot*f_mu*xi_ISM[1]*l_lambda_MIR[fsi]
		ISM_W[fsi]   = LdTot*f_mu*xi_ISM[2]*l_lambda_TWISM[fsi]
		ISM_C[fsi]   = LdTot*f_mu*xi_ISM[3]*l_lambda_TCISM[fsi]
		full_sed[fsi] = BC_PAH[fsi] + BC_MIR[fsi] + BC_W[fsi] + ISM_PAH[fsi] + ISM_MIR[fsi] + ISM_W[fsi] + ISM_C[fsi]

	lambda_list_um=np.zeros(len(lambda_list))
	lambda_list_AA=np.zeros(len(lambda_list))
	for lli in range(len(lambda_list)):
		lambda_list_um[lli]=(1E+4)*lambda_list[lli]
		lambda_list_AA[lli]=(1E+8)*lambda_list[lli]

	fig = plt.figure(figsize=(7,6))
	ax = fig.add_subplot(1, 1, 1)
	ax.set_xscale('log'); ax.set_yscale('log') 
	ax.plot(lambda_list_um, lambda_list_AA*ISM_PAH, label='ISM PAH + NIR',c=colors[0])
	ax.plot(lambda_list_um, lambda_list_AA*ISM_MIR, label='ISM MIR',c=colors[1])
	ax.plot(lambda_list_um, lambda_list_AA*ISM_W, label='ISM Warm',c=colors[2])
	ax.plot(lambda_list_um, lambda_list_AA*ISM_C, label='ISM Cold',c=colors[4])
	ax.plot(lambda_list_um, lambda_list_AA*BC_PAH, label='BC PAH',c=colors[0],linestyle='dashed')
	ax.plot(lambda_list_um, lambda_list_AA*BC_MIR, label='BC MIR',c=colors[1],linestyle='dashed')
	ax.plot(lambda_list_um, lambda_list_AA*BC_W, label='BC Warm',c=colors[2],linestyle='dashed')
	ax.plot(lambda_list_um, lambda_list_AA*full_sed, c='k', label='Full Model',lw=2)
	#plotPoints(flus,lims,exts,ax)
	ax.set_ylim(10**-11.5,10**-6.9)
	ax.set_xlim(xl[0],xl[1])
	ax.set_xlabel('Observed Wavelength [um]',weight='bold')
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

	#Write to text file
	f=open('TEST_SPEC.txt','w');f.close()
	f=open('TEST_SPEC.txt','w')
	for fi in range(len(lambda_list_AA)):
		f.write(str(lambda_list_um[fi])+'\t'+str(lambda_list_AA[fi]*full_sed[fi])+'\t'+str(lambda_list_AA[fi]*full_sed[fi]*0.05)+'\n')
	f.close()

	return full_sed


#Make test lambda list [cm]
xlim_num=[-3.8,-1]
xl=[10**(xlim_num[0]+4),10**(xlim_num[1]+4)]
log_obs_lambda_cm=np.arange(xlim_num[0],xlim_num[1],0.001)
obs_lambda_cm=[]
for i in range(len(log_obs_lambda_cm)):
	obs_lambda_cm.append(10**log_obs_lambda_cm[i])
temp = get_firsed2(obs_lambda_cm)

