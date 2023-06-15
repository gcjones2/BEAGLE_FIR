import numpy as np

def read_library(nmod,inw):

    #Open file
    f_infrared=open('infrared_spectra.lbr')
    ff_infrared=f_infrared.readlines()

    #Read wavelength scale
    xlam=[]
    for i_wavel in range(1,len(ff_infrared)):
        temp_wavel=ff_infrared[i_wavel].split()
        xlam.append(float(str(temp_wavel[0]).replace('[','')))

    #Read set of parameters 
    irprop=np.zeros((nmod,8))
    irprop_temp=ff_infrared[0].split('][')
    for irpi in range(len(irprop_temp)):
        irprop_temp2=irprop_temp[irpi].replace('X ','').replace(']','').replace('[','').split()
        irprop[irpi]=irprop_temp2
    
    #Read infrared SEDs
    sed=np.zeros((nmod,inw))
    for i_wavel in range(1,len(ff_infrared)):
        temp_sed=ff_infrared[i_wavel].split()
        for k_wavel in range(1,len(temp_sed)):
            sed[k_wavel-1,i_wavel-1]=float(str(temp_sed[k_wavel]).replace(']',''))

    #Print infrared properties
    for i in range(len(irprop)):
        temp_irp=irprop[i]
        temp_str='f_mu= %.3f'%float(temp_irp[0])+' | '
        temp_str+='xi_{C}^{ISM}= %.3f'%float(temp_irp[1])+' | '
        temp_str+='T_{W}^{BC}= %.1f'%float(temp_irp[2])+' K | '
        temp_str+='T_{C}^{ISM}= %.1f'%float(temp_irp[3])+'  K | '
        temp_str+='xi_{PAH}^{BC}= %.3f'%float(temp_irp[4])+' | '
        temp_str+='xi_{MIR}^{BC}= %.3f'%float(temp_irp[5])+' | '
        temp_str+='xi_{W}^{BC}= %.3f'%float(temp_irp[6])+' | '
        temp_str+='M_{dust}= %.3f'%float(temp_irp[7])+' M_{\odot}'
        print(temp_str)
