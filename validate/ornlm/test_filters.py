import ornlm_module as ornlm
import numpy as np
import nibabel as nib
from hsm import hsm
from ascm import ascm
def test_filters():
    nib_image=nib.load("data/fibercup-averaged_b-1500.nii")
    image=nib_image.get_data().astype(np.double)
    #nvox=image.shape[0]*image.shape[1]*image.shape[2]*image.shape[3]
    #image[...]=np.array(range(nvox)).reshape(image.shape, order='F')
    fima1=np.empty_like(image, order='F')
    fima2=np.empty_like(image, order='F')
    fima3=np.empty_like(image, order='F')
    fima4=np.empty_like(image, order='F')
    ###############################################
    for i in range(65):
        mv=image[:,:,:,i].max()
        fima1[:,:,:,i]=ornlm.ornlm_pyx(image[:,:,:,i], 3, 1, 0.05*mv)
        fima2[:,:,:,i]=ornlm.ornlmpy(image[:,:,:,i], 3, 1, 0.05*mv)
        diff1=abs(fima1[...,i]-fima2[...,i])
        print i,": Maximum error [ornlm (block size= 3x3)]: ", diff1.max()
    return
    ###############################################
    for i in xrange(image.shape[3]):
        print "Filtering volume",i+1,"/",image.shape[3]
        mv=image[:,:,:,i].max()
        #fima1[:,:,:,i]=ornlm.ornlmpy(image[:,:,:,i], 3, 1, 0.05*mv)
        #fima2[:,:,:,i]=ornlm.ornlmpy(image[:,:,:,i], 3, 2, 0.05*mv)
        fima1[:,:,:,i]=ornlm.ornlm_pyx(image[:,:,:,i], 3, 1, 0.05*mv)
        fima2[:,:,:,i]=ornlm.ornlm_pyx(image[:,:,:,i], 3, 2, 0.05*mv)
        fima4[:,:,:,i]=np.array(ascm(image[:,:,:,i], fima1[:,:,:,i],fima2[:,:,:,i], 0.05*mv))
        fima3[:,:,:,i]=np.array(hsm(fima1[:,:,:,i],fima2[:,:,:,i]))
    #####ornlm######
    nii_matlab_filtered1=nib.load('data/filtered_3_1_1.nii');
    matlab_filtered1=nii_matlab_filtered1.get_data().astype(np.double)
    diff1=abs(fima1-matlab_filtered1)
    print "Maximum error [ornlm (block size= 3x3)]: ", diff1.max()
    #####ornlm######
    nii_matlab_filtered2=nib.load('data/filtered_3_2_1.nii');
    matlab_filtered2=nii_matlab_filtered2.get_data().astype(np.double)
    diff2=abs(fima2-matlab_filtered2)
    print "Maximum error [ornlm (block size= 5x5)]: ", diff2.max()
    #######hsm########
    nii_matlab_filtered3=nib.load('data/filtered_hsm.nii');
    matlab_filtered3=nii_matlab_filtered3.get_data().astype(np.double)
    diff3=abs(fima3-matlab_filtered3)
    print "Maximum error [hsm]: ", diff3.max()
    #######ascm########
    nii_matlab_filtered4=nib.load('data/filtered_ascm.nii');
    matlab_filtered4=nii_matlab_filtered4.get_data().astype(np.double)
    diff4=abs(fima4-matlab_filtered4)
    print "Maximum error [ascm]: ", diff4.max()


if __name__=='__main__':
    #--test Average_block--
    sz=(5,6,7)
    ll=sz[0]*sz[1]*sz[2]
    X=np.array(range(ll), dtype=np.float64).reshape(sz)
    #X=np.array(np.random.uniform(0,1,ll), dtype=np.float64).reshape(sz)
    error=0
    for mask in range(8):
        px=(mask&1>0)*(sz[1]-1)
        py=(mask&2>0)*(sz[0]-1)
        pz=(mask&4>0)*(sz[2]-1)
        av=np.zeros((3,3,3))
        ornlm.Average_block_pyx(X, px, py, pz, av, 1)
        av_import=np.zeros((3,3,3))
        ornlm.Average_block_cpp(X, px, py, pz, av_import, 1)
        error+=(av-av_import).std()
    print 'Average_block error: ', error
    #--test Value_block--
    sz=(5,6,7)
    ll=sz[0]*sz[1]*sz[2]
    average=np.array(np.random.uniform(0,1,27), dtype=np.float64).reshape((3,3,3))
    error=0
    variation=0
    for mask in range(8):
        px=(mask&1>0)*(sz[1]-1)
        py=(mask&2>0)*(sz[0]-1)
        pz=(mask&4>0)*(sz[2]-1)
        E=np.array(np.random.uniform(0,1,ll), dtype=np.float64).reshape(sz)
        L=2*np.array(range(ll), dtype=np.float64).reshape(sz)
        ornlm.Value_block_cpp(E,L, px, py, pz, average, 1, 1)
        #EE=np.array(range(ll), dtype=np.float64).reshape(sz)
        EE=E.copy()
        LL=2*np.array(range(ll), dtype=np.float64).reshape(sz)
        variation+=(E-EE).std()
        variation+=(L-LL).std()
        ornlm.Value_block_pyx(EE, LL, px, py, pz, average, 1, 1 )
        error+=(E-EE).std()
        error+=(L-LL).std()
    print 'Value_block error: ', error
    #--test distance--
    ima=np.array(np.random.uniform(0,1,125), dtype=np.float64).reshape((5,5,5), order='F')
    err=0.0
    for p in np.ndindex(ima.shape):
        for q in np.ndindex(ima.shape):
            x,y,z=(p[1], p[0], p[2])
            nx,ny,nz=(q[1], q[0], q[2])
            d=ornlm.distance_pyx(ima, x,y,z, nx,ny,nz, 1)
            d_import=ornlm.distance_cpp(ima, x,y,z, nx,ny,nz, 1)
            err+=(d-d_import)**2
    print 'distance error: ', error
    test_filters()
