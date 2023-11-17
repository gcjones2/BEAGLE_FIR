'''
cd /Users/garethjones/Desktop/BEAGLE_FIR-main
conda activate py39
python convert_PAH.py 
'''
#Converts original PAH+NIR template from um, fnu -> cm, normalized flambda

import numpy as np

cspl=2.99792458e+10 #cm/s

#Read the file that contains the PAH template (M17) + hot NIR continuum
print('Getting PAH + NIR template...')
lampah=[]; l_lambda_PAH=[]
f_pahs_nir=open('pahs_nir.dat','r')
ff_pahs_nir=f_pahs_nir.readlines()
for i in range(len(ff_pahs_nir)):
	temp_pahs=ff_pahs_nir[i].split()
	lampah.append((1E-4)*float(temp_pahs[0])) #cm
	l_lambda_PAH.append(float(temp_pahs[1]))
#Convert from fnu to flambda
for mbbi in range(len(lampah)):
	l_lambda_PAH[mbbi]*=(cspl/(lampah[mbbi]**2)) * (1E-10)
#Normalize PAH template
norm_pah=np.trapz(l_lambda_PAH, lampah)
for i in range(len(lampah)):
	l_lambda_PAH[i]/=norm_pah
#Write this out
g = open('pahs_nir_fl_norm.dat','w');g.close()
g = open('pahs_nir_fl_norm.dat','w')
for i in range(len(lampah)):
	g.write(str(round(lampah[i],6))+' '+str(l_lambda_PAH[i])+'\n')
g.close()
