from aonlm import aonlm
import numpy as np
import nibabel as nib
from mixingsubband import mixingsubband

def test_filters():
    #Load ground truth
    nii_matlab_filtered1=nib.load('data/filtered_naonlm_3_1_1.nii');
    matlab_filtered1=nii_matlab_filtered1.get_data().astype(np.double)
    nii_matlab_filtered2=nib.load('data/filtered_naonlm_3_2_1.nii');
    matlab_filtered2=nii_matlab_filtered2.get_data().astype(np.double)
    nii_matlab_filtered3=nib.load('data/filtered_naonlm_final.nii');
    matlab_filtered3=nii_matlab_filtered3.get_data().astype(np.double)
    #Load data
    name ='data/fibercup-averaged_b-1500.nii';
    nib_image=nib.load(name)
    image=nib_image.get_data().astype(np.double)
    ##################
    fimau=np.zeros_like(image)
    fimao=np.zeros_like(image)
    fima=np.zeros_like(image)
    for i in xrange(image.shape[3]):
        print "Filtering volume",i+1,"/",image.shape[3]
        fimau[:,:,:,i]=aonlm(image[:,:,:,i], 3, 1, 1)
        diffu=abs(fimau[:,:,:,i]-matlab_filtered1[:,:,:,i])
        diff_check=abs(fimau[:,:,:,i]-image[:,:,:,i])
        print "Maximum error [aonlm (block size= 3x3)]: ", diffu.max(),". Check: ",diff_check.max()
        fimao[:,:,:,i]=aonlm(image[:,:,:,i], 3, 2, 1)
        diffo=abs(fimao[:,:,:,i]-matlab_filtered2[:,:,:,i])
        diff_check=abs(fimao[:,:,:,i]-image[:,:,:,i])
        print "Maximum error [aonlm (block size= 5x5)]: ", diffo.max(),". Check: ",diff_check.max()
        fima[:,:,:,i]=mixingsubband(fimau[:,:,:,i],fimao[:,:,:,i])
        diff=abs(fima[:,:,:,i] - matlab_filtered3[:,:,:,i])
        diff_check=abs(fima[:,:,:,i]-image[:,:,:,i])
        print "Maximum error [mixed]: ", diff.max(),". Check: ",diff_check.max()

if __name__=='__main__':
    test_filters()