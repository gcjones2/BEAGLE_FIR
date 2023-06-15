import numpy as np
from scipy.integrate import trapezoid

cspl=2.99792458E+10 #cm/s
h=6.6261E-27 #erg sec
kbb=1.3807E-16 #erg/grad
tbg3=45.
l_sun=3.90E+33 #cgs
m_sun=1.99E+33 #g


#Integrates spectrum (trapz1.f)
def trapz1(x, y):
	return trapezoid(y,x)

#Calculates FIR SED (OK)
def get_firsed(ratio,ratio_ism,t1,t2,tbg1,tbg2,frac,npoints,lmd,fmbb1,fmbb15,fmbb2,fism_mir,fpah):

	#Find the index j corresponding to the MBB temperatures
	k=int(t1)
	if ((t1-k)>0.5):
		k1=k+1
	else:
		k1=k
	k=int(t2)
	if ((t2-k)>0.5):
		k2=k+1
	else:
		k2=k

	#NOTE: the MBB spectra are already normalized
	#Convert spectra from F_lambda to F_nu (to be consistent with PAHs template)
	nu=np.zeros(npoints)
	fvsg=np.zeros(npoints); fvsg1=np.zeros(npoints); fvsg2=np.zeros(npoints)
	fbg1=np.zeros(npoints)
	fism_fir=np.zeros(npoints); fism=np.zeros(npoints)
	
	for i in range(npoints):
		nu[i]=cspl/lmd[i]
		fvsg1[i]=fmbb1[i][k1] * (cspl/(nu[i]**2)) #MBB warm VSGs
		fvsg2[i]=fmbb1[i][k2] * (cspl/(nu[i]**2)) #MBB cold VSGs
		fvsg[i]=fvsg1[i]+fvsg2[i]	
	norm_vsg=abs(trapz1(nu,fvsg))
	for i in range(npoints):
		fvsg[i]/=norm_vsg
	#MBB ISM BGs: FIR ISM with varying temp (15-25K)
	k=int(tbg2)
	if ((tbg2-k)>0.5):
		kbg2=k+1
	else:
		kbg2=k
	for i in range(npoints):
		fism_fir[i]=fmbb2[i][kbg2] * (cspl/(nu[i]**2))
	#Diffuse ISM component
	for i in range(npoints):
		fism[i]=((1.-ratio_ism)*fism_mir[i])+(ratio_ism*fism_fir[i])
		
	#BGs of birth clouds emission spectrum (not fixed)
	#Find index corresponding to BGs temperature
	k=int(tbg1)
	if((tbg1-k)>0.5):
		kbg1=k+1
	else:
		kbg1=k
	#MBB birth cloud BGs
	for i in range(npoints):
		fbg1[i]=fmbb15[i][kbg1] * (cspl/(nu[i]**2))

	#SUM ALL THE SPECTRA - RESULT: Fnu in the wavelengths lambda_final
	#f_mu=ratio=Ld(ISM)/[Ld(BC)+Ld(ISM)]
	#Assume Ld(Total)=1 (normalized) -> Ld(ISM)=ratio; Ld(BC)=1-ratio
	#Final dust emission spectrum
	firsed=np.zeros(npoints)
	for i in range(npoints):
		firsed[i] =(frac[0]*(1.-ratio)*fpah[i])
		firsed[i]+=(frac[1]*(1.-ratio)*fvsg[i])
		firsed[i]+=(frac[2]*(1.-ratio)*fbg1[i])
		firsed[i]+=(ratio*fism[i])

	#Compute black body for tbg1, tbg2 and tbg3 (B_lambda)
	mbb1=np.zeros(npoints); mbb2=np.zeros(npoints); mbb3=np.zeros(npoints)
	for i in range(npoints):
		mbb1[i]=(2*h*nu[i]**3/cspl**2)/(np.exp(h*nu[i]/(kbb*tbg1))-1.) #BG temperatures in BCs: between 30 and 60 K
		mbb2[i]=(2*h*nu[i]**3/cspl**2)/(np.exp(h*nu[i]/(kbb*tbg2))-1.) #BG temperature in the ISM: between 15 and 25 K
		mbb3[i]=(2*h*nu[i]**3/cspl**2)/(np.exp(h*nu[i]/(kbb*tbg3))-1.)
		#!MBB: multiply by nu**beta
		mbb2[i]*=(nu[i]**2)
		mbb1[i]*=(nu[i]**1.5)
		mbb3[i]*=(nu[i]**1.5)

	#Compute dust masses in each component in thermal equilibrium
	#1. Cold dust in the diffuse ISM:
	mass_c_ism=(1.28E+22)*ratio_ism*ratio*l_sun/abs(trapz1(nu,mbb2))
	#2. Warm dust in the diffuse ISM:
	mass_w_ism=(2.165e+16)*(1.-ratio)*0.175*ratio*l_sun/abs(trapz1(nu,mbb3))
	#3. Warm dust in the Birth Clouds:
	mass_w_bc=(2.165e+16)*(1.-ratio)*frac[2]*l_sun/abs(trapz1(nu,mbb1))
	#Correct for stochastically heated small grains:
	md=1.1*(mass_c_ism+mass_w_ism+mass_w_bc)
	#Finally, convert to solar masses:
	print(mass_c_ism,mass_w_ism,mass_w_bc)
	md/=m_sun

	return firsed,md

