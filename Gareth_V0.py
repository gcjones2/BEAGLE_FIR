'''
cd /Users/garethjones/Downloads/mapghys4gareth_V1
conda activate py3
python Gareth_V0.py
'''

from get_firsed import *
from read_library import *
#
from scipy.interpolate import CubicSpline
import numpy as np
import random as rnd

###
#Define constants
knu=0.77 #dust emissivity g^-1 cm^2 (Dunne & al, for 850um)
h=6.6261e-27 #erg sec
kbb=1.3807e-16 #erg/grad
m_sun=1.99e+33 #g
l_sun=3.90e+33 #cgs
lcenter=[12.,25.,60.,100.,6.75,14.3,3.550,4.493,5.731,7.872,23.675,71.44,155.9,443.,863.,15.8]
redshift=0.
tbg3=45.
icall=0
cspl=2.99792458e+10 #cm/s
ntdmbb=300
npoints=6450
inw=6450
nmod=50 #nmod=50000
nmax=21150
nidx=31
ngauss=101
ntbg=100
ntc=150

#Interpolates data (interp.f)
def INTERP(x,y,npts,nterms,xin,yout):
   CS=CubicSpline(x,y)
   yout=np.zeros(len(xin))
   for interp_i in range(len(xin)):
      yout=CS(xin[interp_i])

#(locate.f)
def locate(xx,n,x,j):
  jl=1
  ju=n+1
  if (ju-jl>1):
    jm=(ju+jl)/2
    if ((xx[n]>xx[1])==(x>xx[jm])):
      jl=jm
    else:
      ju=jm
    j=jl
    return xx,n,x,j

#Read templates once
def get_files():
   #Read the file that contains the PAH template (M17) + hot NIR continuum
   f_pahs_nir=open('pahs_nir.dat','r')
   ff_pahs_nir=f_pahs_nir.readlines()
   for i in range(npoints):
      temp_pahs=ff_pahs_nir[i].split()
      nupah.append(cspl/((1E-4)*float(temp_pahs[0])))
      fpah.append(float(temp_pahs[1]))
   #Normalize PAH template
   norm_pah=trapz1(nupah,fpah)
   for i in range(npoints):
      fpah[i]/=norm_pah

   #Read ISM MIR template: in F_nu (includes hot NIR continuum). This is computed from the fit to the Milky Way cirrus emission
   f_ism_mir=open('ism_mir.dat','r')
   ff_ism_mir=f_ism_mir.readlines()
   for i in range(npoints):
      temp_pahs=ff_ism_mir[i].split()
      fism_mir.append(float(temp_pahs[1]))
   #Normalize ISM MIR template
   norm_pah=trapz1(nupah,fism_mir)
   for i in range(npoints):
      fism_mir[i]/=norm_pah

   #Read the files that contains the modified black bodies (T from 1 to 300 K)
   #(three different emissivity indexes)
   #B_lambda in cgs and lambda in cm
   f_mbb_b1=open('mbb_beta1.dat','r')
   ff_mbb_b1=f_mbb_b1.readlines()
   for i in range(len(ff_mbb_b1)):
      temp_mbb=ff_mbb_b1[i].split()
      lmd.append(float(temp_mbb[0])) #[cm]
      temp_mbb1=[]
      for j in range(1,len(temp_mbb)):
         temp_mbb1.append(float(temp_mbb[j]))
      fmbb1.append(temp_mbb1)
   #
   f_mbb_b15=open('mbb_beta1.5.dat','r')
   ff_mbb_b15=f_mbb_b15.readlines()
   for i in range(len(ff_mbb_b15)):
      temp_mbb=ff_mbb_b15[i].split()
      temp_mbb15=[]
      for j in range(1,len(temp_mbb)):
         temp_mbb15.append(float(temp_mbb[j]))
      fmbb15.append(temp_mbb15)
   #
   f_mbb_b2=open('mbb_beta2.dat','r')
   ff_mbb_b2=f_mbb_b2.readlines()
   for i in range(len(ff_mbb_b2)):
      temp_mbb=ff_mbb_b2[i].split()
      temp_mbb2=[]
      for j in range(1,len(temp_mbb)):
         temp_mbb2.append(float(temp_mbb[j]))
      fmbb2.append(temp_mbb2)

###
###
###
#Program ir_generator

#Read wavelength vector [um]
xlam=[]
f_wavel=open('lambda.dat','r')
ff_wavel=f_wavel.readlines()
for i_wavel in range(len(ff_wavel)):
   xlam.append(float(ff_wavel[i_wavel]))

#Get templates
nupah=[]; fpah=[]
fism_mir=[]
lmd=[]
fmbb1=[]; fmbb15=[]; fmbb2=[]
get_files()

### GRID OF MODELS ####
irprop=np.zeros((nmod,8))
sed=np.zeros((nmod,inw))
for igrid in range(nmod):

   #Draw randomly the free parameters of the model:
   #fmu=Ld(ISM)/Ld(Total):
   aux=rnd.random()
   x=aux
   #xi_cold^ISM:
   aux=rnd.random()
   fmu_ism=0.5+(1.-0.5)*aux

   '''
   Contribution of warm dust, hot MIR continuum and PAH to the total luminosity of birth clouds. 
   First xi_W^BC, then xi_MIR^BC, then xi_PAH^BC. 
   The sum of the three must be unity.
   '''
   xi=np.zeros(3)
   #xi_W^BC
   aux=rnd.random()
   xi[2]=aux
   #xi_MIR^BC
   aux=rnd.random()
   xi[1]=(1.-xi[2])*aux
   #xi_PAH^BC
   xi[0]=1.-xi[2]-xi[1]

   ###DUST TEMPERATURES:
   #MIR continuum (fixed temperatures)
   t_vsg1=250.
   t_vsg2=130.
   #t_bg2: BG temperature in the ISM: between 15 and 25 K
   aux=rnd.random()
   t_bg2=15.+(25.-15.)*aux
   #t_bg1: BGs temperatures in BCs: between 30 and 60 K
   aux=rnd.random()
   t_bg1=30.+(60.-30.)*aux

   #Compute infrared spectra (and corresponding dust mass)
   fnu,mdust = get_firsed(x,fmu_ism,t_vsg1,t_vsg2,t_bg1,t_bg2,xi,npoints,lmd,fmbb1,fmbb15,fmbb2,fism_mir,fpah)

   #Infrared properties of each model:
   irprop[igrid,0]=x        #fmu
   irprop[igrid,1]=fmu_ism  #xi_{C}^{ISM}
   irprop[igrid,2]=t_bg1    #T_{W}^{BC}
   irprop[igrid,3]=t_bg2    #T_{C}^{ISM}
   irprop[igrid,4]=xi[0]    #xi_{PAH}^{BC}
   irprop[igrid,5]=xi[1]    #xi_{MIR}^{BC}
   irprop[igrid,6]=xi[2]    #xi_{W}^{BC}
   irprop[igrid,7]=mdust    #Mdust/M_sun

   ###SED
   for sed_i in range(inw):
      sed[igrid,sed_i]=fnu[sed_i]

   print(igrid+1,'/',nmod)

##Open output files 
f_infrared=open('infrared_spectra.lbr','w');f_infrared.close()
f_infrared=open('infrared_spectra.lbr','w')
#Write irprop values
temp_str='X '
for i in range(len(irprop)):
   temp_str+=str(irprop[i]).replace('\n','')
f_infrared.write(temp_str+'\n')
#Write set of parameters and infrared SED
for k in range(inw):
   temp_str=str(xlam[k])
   for i in range(nmod):
      temp_str+=' '+str(sed[i,k])
   f_infrared.write(temp_str+'\n')
f_infrared.close()

###
###

read_library(nmod,inw)


