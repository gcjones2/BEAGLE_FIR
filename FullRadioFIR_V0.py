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

#Trio of functions that calculate MBB 
def correctTD(td,beta,redshift):
	fb=4.+beta
	ctd=((td**fb)+(TCMB**fb)*(((1.+redshift)**fb)-1.))**(1./fb)
	return ctd
def FCMB(freqs,td_corr,redshift):
	tcmb=TCMB*(1.+redshift)
	exp1=1./(np.exp((h*freqs)/(kbb*td_corr))-1.)
	exp2=1./(np.exp((h*freqs)/(kbb*tcmb))-1.)
	return exp1-exp2
def MBB(lambda_o, redshift, Rgal, td, logMD, beta):
	MD = (10**logMD) * m_sun #[g]
	freqs=np.zeros(len(lambda_o)); freqs_o=np.zeros(len(lambda_o))
	for fib in range(len(lambda_o)):
		freqs[fib] = np.asarray(cspl/lambda_o[fib])*(1.+redshift) #rest-frame [Hz]
		freqs_o[fib] = np.asarray(cspl/lambda_o[fib]) #observer-frame [Hz]
	Rgal *= Rgal * (3.085677E+21) #[cm]
	dl=CC.getcosmos(redshift,cosmoparams[0],cosmoparams[1],cosmoparams[2])[1] * (3.085677E+24) #[cm]
	#
	AGAL = np.pi*(Rgal**2.)
	OMEGA=((1.+redshift)**4.)*AGAL/(dl**2.)
	pt1=OMEGA/((1.+redshift)**3.)
	pt2=((2.*h*(freqs**3.))/(cspl**2.))*FCMB(freqs,correctTD(td,beta,redshift),redshift)
	taunu=(MD*knu*((freqs/nu_o)**beta))/AGAL
	pt3=1.-np.exp(-1.*taunu)
	final_MBB=[]
	#Convert Fnu [erg/cm^2/s/Hz] to Flambda [erg/cm^2/s/A]
	for mbbi in range(len(pt2)):
		temp_MBB = pt1*pt2[mbbi]*pt3[mbbi]
		final_MBB.append(temp_MBB * (cspl/(lambda_o[mbbi]**2)) * (1E-8) )
	#Normalize
	MBB_sum = np.trapz(final_MBB,lambda_o)
	for mbbi in range(len(pt2)):
		final_MBB[mbbi]/=MBB_sum
	return final_MBB

#Free-free and Synchrotron emission
#Equation 4 of Algera+21
#lambda_FFS: List of observer-frame wavelengths [cm]
#Lvo: Flux at 1.4GHz [erg/cm^2/s/Hz]
#fvoth: Fraction of 1.4GHz flux from FF
#alphant: Synchrotron slope (-0.85)
def getFFSynch(lambda_FFS, Lvo, fvoth, alphant, redshift):
	turn_nu = (1.4E+9)/(1.+redshift)
	nu_FFS=np.zeros(len(lambda_FFS))
	for ffsi in range(len(lambda_FFS)):
		nu_FFS[ffsi]=(cspl/lambda_FFS[ffsi])
	ffs1 = (1.-fvoth) * ((nu_FFS/turn_nu)**alphant)
	ffs2 = fvoth * ((nu_FFS/turn_nu)**-0.1)
	dl=CC.getcosmos(redshift,cosmoparams[0],cosmoparams[1],cosmoparams[2])[1] * (3.085677E+24) #[cm]
	ffs = (ffs1 + ffs2) / (dl**2)
	#Convert Fnu [erg/cm^2/s/Hz] to Flambda [erg/cm^2/s/A]
	final_ffs=[]
	for mbbi in range(len(ffs)):
		temp1 = cspl/(lambda_FFS[mbbi]**2)
		temp2 = 1E-8
		final_ffs.append((1.+redshift) * ffs[mbbi] * temp1 * temp2)
	#Normalize to Lvo
	ffs_norm=(1.+redshift) * cspl *(1E-8)
	ffs_norm/=((cspl/(1.4E+9))**2)
	for mbbi in range(len(ffs)):
		final_ffs[mbbi]*=(Lvo/ffs_norm)
	return final_ffs

def GetData():
    name = 'HFLS3'
    redshift = 6.3369
    # Photometric radius
    Rgal = 1.7		
    flus = [[1200.0, 12.0, 2.3],
            [850.0, 32.4, 2.3],
            [600.0, 47.3, 2.8],
            [341.0, 33.0, 2.4],
            [284.161, 20.57, 0.45],
            [270, 21.3, 1.1],
            [260.329, 17.10, 0.37],
            [259.106, 17.12, 0.81],
            [259.106, 13.9, 1.9],
            [251.906, 13.51, 1.22],
            [250.490, 14.07, 0.60],
            [240.364, 15.05, 0.19],
            [240.284, 14.17, 0.38],
            [227.604, 10.13, 0.29],
            [222.167, 11.76, 0.40],
            [207.013, 9.78, 0.45],
            [204.027, 9.79, 0.29],
            [164.598, 6.57, 0.18],
            [157.757, 4.59, 0.39],
            [150, 2.93, 0.37],
            [148.800, 3.35, 0.12],
            [141.328, 3.22, 0.12],
            [134.650, 2.38, 0.11],
            [113.819, 1.25, 0.09],
            [110.128, 1.21, 0.10],
            [110.128, 1.59, 0.12],
            [102.500, 0.705, 0.134],
            [94.246, 0.527, 0.078],
			] #nu_obs [GHz], [mJy]
    lims = []
    lims.append([640000.0, 0.052e-3, -1])
    lims.append([485000.0, 0.083e-3, -1])
    lims.append([400000.0, 0.052e-3, -1])
    lims.append([335000.0, 0.157e-3, -1])
    lims.append([88000.0, 0.08, -1])
    lims.append([65000.0, 0.11, -1])
    lims.append([25000.0, 0.8, -1])
    lims.append([13600.0, 6, -1])
    lims.append([4300.0, 2.0, -1])
    lims.append([2700.0, 2.2, -1])
    lims.append([1900.0, 4.0, -1])
    lims.append([15.7112, 0.015, -1])
    exts = [[135000.0, 1.823e-3, 0.305e-3],
            [83000.0, 2.39e-3, 0.25e-3],
            [67000.0, 3.16e-3, 0.52e-3],
            [47.1311, 0.139, 0.030],
            [31.4217, 0.0469, 0.0093],
            [1.4, 0.059, 0.011]]
    # exts.append([422.76490838, 0.16752175, -1])
    # Initial guesses
    MD = np.log10(1.31E+9)
    beta = 1.92
    td = 55.9
    return [name,redshift,Rgal,MD,beta,td,flus,lims,exts]

def plotPoints(flus,lims,exts,ax):
	for ppi in range(len(flus)):
		tempx = (1E+6)*(cspl_m)/(flus[ppi][0]*(1E+9)) #Observed um
		tempy = (tempx*1E+4) * (1E-26) * flus[ppi][1] * cspl_m * (1E+10) / ((tempx*1E-6)**2)
		ax.plot(tempx, tempy, marker='o', color='k', markersize=6)
	for ppi in range(len(lims)):
		tempx = (1E+6)*(cspl_m)/(lims[ppi][0]*(1E+9)) #Observed um
		tempy = (tempx*1E+4) * (1E-26) * lims[ppi][1] * cspl_m * (1E+10) / ((tempx*1E-6)**2)
		plt.plot(tempx, tempy,marker='v',mec='r',mfc='w')
	for ppi in range(len(exts)):
		tempx = (1E+6)*(cspl_m)/(exts[ppi][0]*(1E+9)) #Observed um
		tempy = (tempx*1E+4) * (1E-26) * exts[ppi][1] * cspl_m * (1E+10) / ((tempx*1E-6)**2)
		ax.plot(tempx, tempy, marker='o', color='k', markersize=6)

def get_firsed2(lambda_list):

	##Basic values
	#Two MIR temperatures
	T_MIR_1=250. #[K]
	T_MIR_2=130. #[K]
	T_W_BC=48.#30.+(60.-30.)*rnd.random()
	T_C_ISM=22.#15.+(25.-15.)*rnd.random()
	T_W_ISM=45.
	#Relative contributions to BC luminosity
	xi_BC=[0.05,0.15,0.80]
	#np.random.dirichlet(np.ones(3),size=1)[0]
	#xi_BC[2]=rnd.random() #xi_W^BC
	#xi_BC[1]=(1.-xi_BC[2])*rnd.random() #xi_MIR^BC
	#xi_BC[0]=1.-xi_BC[2]-xi_BC[1] #xi_PAH^BC
	#
	#Relative contributions to ISM luminosity
	xi_ISM=[-1,-1,-1,0.8]
	xi_ISM[2]=0.07*((1.-xi_ISM[3])/0.4)
	xi_ISM[1]=0.11*((1.-xi_ISM[3])/0.4)
	xi_ISM[0]=0.22*((1.-xi_ISM[3])/0.4)
	#np.random.dirichlet(np.ones(4),size=1)[0]
	#xi_ISM[0]=rnd.random() #xi_PAH^ISM
	#xi_ISM[1]=(1.-xi_ISM[3])*rnd.random() #xi_MIR^ISM
	#xi_ISM[2]=(1.-xi_ISM[3]-xi_ISM[2])*rnd.random() #xi_W^ISM
	#xi_ISM[3]=1.-xi_ISM[3]-xi_ISM[2]-xi_ISM[1] #xi_C^ISM
	#
	#Fraction of IR luminosity contributed by ISM
	f_mu=0.10 #rnd.random() - VARY THIS!
	#
	#FF Synch properties
	alphant = -0.85 #VARY THIS!
	fvoth = 0.1 #VARY THIS!
	Lvo = 1E+51 #[erg/cm^2/s/Hz] VARY THIS!
	#
	LdTot = 0.4 #VARY THIS!

	#Assumed galaxy properties
	beta = 1.8
	logMD = 9.  #[log10(Msol)]
	redshift = 0.5
	RG = 0.7 #[kpc]

	[name, redshift, RG,logMD, beta, td, flus, lims, exts] = GetData()

	#Read the file that contains the PAH template (M17) + hot NIR continuum
	nupah=[]
	lampah=[]
	l_lambda_PAH=[]
	f_pahs_nir=open('pahs_nir.dat','r')
	ff_pahs_nir=f_pahs_nir.readlines()
	for i in range(npoints):
		temp_pahs=ff_pahs_nir[i].split()
		if (1+redshift)*(1E-4)*float(temp_pahs[0])<1.E-2:
			nupah.append(cspl/((1+redshift)*(1E-4)*float(temp_pahs[0]))) #Hz
			lampah.append((1+redshift)*(1E-4)*float(temp_pahs[0])) #cm
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
	l_lambda_MIR_1 = MBB(lambda_list, redshift, RG, T_MIR_1, logMD, beta)
	l_lambda_MIR_2 = MBB(lambda_list, redshift, RG, T_MIR_2, logMD, beta)
	l_lambda_MIR = np.zeros(len(lambda_list))
	for miri in range(len(lambda_list)):
		l_lambda_MIR[miri] = l_lambda_MIR_1[miri] + l_lambda_MIR_2[miri]
	MIR_SUM = np.trapz(l_lambda_MIR,lambda_list)
	for mbbi in range(len(l_lambda_MIR)):
		l_lambda_MIR[mbbi]/=MIR_SUM
	#
	l_lambda_TWBC = MBB(lambda_list, redshift, RG, T_W_BC, logMD, beta)
	l_lambda_TWISM = MBB(lambda_list, redshift, RG, T_W_ISM, logMD, beta)
	l_lambda_TCISM = MBB(lambda_list, redshift, RG, T_C_ISM, logMD, beta)
	#
	l_lambda_FFSynch = getFFSynch(lambda_list, Lvo, fvoth, alphant, redshift)

	BC_PAH=np.zeros(len(lambda_list)); BC_MIR=np.zeros(len(lambda_list)); BC_W=np.zeros(len(lambda_list))
	ISM_PAH=np.zeros(len(lambda_list)); ISM_MIR=np.zeros(len(lambda_list)); ISM_W=np.zeros(len(lambda_list)); ISM_C=np.zeros(len(lambda_list))
	FFSynch=np.zeros(len(lambda_list))
	full_sed=np.zeros(len(lambda_list))
	for fsi in range(len(lambda_list)):
		BC_PAH[fsi] = LdTot*(1-f_mu)*xi_BC[0]*l_lambda_PAH_interp[fsi]
		BC_MIR[fsi] = LdTot*(1-f_mu)*xi_BC[1]*l_lambda_MIR[fsi]
		BC_W[fsi]   = LdTot*(1-f_mu)*xi_BC[2]*l_lambda_TWBC[fsi]
		ISM_PAH[fsi] = LdTot*f_mu*xi_ISM[0]*l_lambda_PAH_interp[fsi]
		ISM_MIR[fsi] = LdTot*f_mu*xi_ISM[1]*l_lambda_MIR[fsi]
		ISM_W[fsi]   = LdTot*f_mu*xi_ISM[2]*l_lambda_TWISM[fsi]
		ISM_C[fsi]   = LdTot*f_mu*xi_ISM[3]*l_lambda_TCISM[fsi]
		FFSynch[fsi] = l_lambda_FFSynch[fsi]
		full_sed[fsi] = BC_PAH[fsi] + BC_MIR[fsi] + BC_W[fsi] + ISM_PAH[fsi] + ISM_MIR[fsi] + ISM_W[fsi] + ISM_C[fsi] + FFSynch[fsi]

	lambda_list_um=np.zeros(len(lambda_list))
	lambda_list_AA=np.zeros(len(lambda_list))
	for lli in range(len(lambda_list)):
		lambda_list_um[lli]=(1E+4)*lambda_list[lli]
		lambda_list_AA[lli]=(1E+8)*lambda_list[lli]

	fig = plt.figure(figsize=(10,6))
	ax = fig.add_subplot(1, 1, 1)
	ax.set_xscale('log'); ax.set_yscale('log') 
	ax.plot(lambda_list_um, lambda_list_AA*ISM_PAH, label='ISM PAH',c=colors[0])
	ax.plot(lambda_list_um, lambda_list_AA*ISM_MIR, label='ISM MIR',c=colors[1])
	ax.plot(lambda_list_um, lambda_list_AA*ISM_W, label='ISM Warm',c=colors[2])
	ax.plot(lambda_list_um, lambda_list_AA*ISM_C, label='ISM Cold',c=colors[4])
	ax.plot(lambda_list_um, lambda_list_AA*BC_PAH, label='BC PAH',c=colors[0],linestyle='dashed')
	ax.plot(lambda_list_um, lambda_list_AA*BC_MIR, label='BC MIR',c=colors[1],linestyle='dashed')
	ax.plot(lambda_list_um, lambda_list_AA*BC_W, label='BC Warm',c=colors[2],linestyle='dashed')
	ax.plot(lambda_list_um, lambda_list_AA*FFSynch, label='FF Synch',c=colors[3],linestyle='solid')
	ax.plot(lambda_list_um, lambda_list_AA*full_sed, c='k', label='Full Model',lw=2)
	plotPoints(flus,lims,exts,ax)
	ax.set_ylim(1E+1,1E+9)
	ax.set_xlim(xl[0],xl[1])
	ax.set_xlabel('Observed Wavelength [um]',weight='bold')
	ax.set_ylabel(r'$\lambda$ F$_{\lambda}$ [erg cm$^{-2}$ s$^{-1}$]',weight='bold')
	plt.legend()
	def conversion_func(x):
		return (1E-9)*cspl*(1E-2)/(x*1E-6)
	ticks_x = np.logspace(np.log10(xl[0]), np.log10(xl[1]), int(np.log10(xl[1]) + 1))  # must span limits of first axis with clever spacing
	ticks_z = conversion_func(ticks_x)
	ax2 = ax.twiny()  # get the twin axis
	ax2.semilogx(ticks_z, np.ones_like(ticks_z), alpha=0)  # transparent dummy plot
	ax2.set_xlim(ticks_z[0], ticks_z[-1])
	ax2.set_xlabel('Observed Frequency [GHz]',weight='bold')
	plt.show()

	return full_sed


#Make test lambda list [cm]
xlim_num=[-5,2]
xl=[10**(xlim_num[0]+4),10**(xlim_num[1]+4)]
log_obs_lambda_cm=np.arange(xlim_num[0],xlim_num[1],0.001)
obs_lambda_cm=[]
for i in range(len(log_obs_lambda_cm)):
	obs_lambda_cm.append(10**log_obs_lambda_cm[i])
temp = get_firsed2(obs_lambda_cm)
#plt.plot(obs_lambda_um,temp)
#plt.semilogx()
#plt.semilogy()
#plt.show()
#

