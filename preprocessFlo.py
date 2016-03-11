import numpy as np;
import scipy.misc;
import matplotlib;
matplotlib.use('Agg')
import matplotlib.pyplot as plt;

def readFlowFile(file_name):
    data2D=None
    with open(file_name,'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print 'Magic number incorrect. Invalid .flo file'
        else:
            w = np.fromfile(f, np.int32, count=1)
            h = np.fromfile(f, np.int32, count=1)
            print 'Reading %d x %d flo file' % (w, h)
            data = np.fromfile(f, np.float32, count=2*w*h)
            # Reshape data into 3D array (columns, rows, bands)
            data2D = np.resize(data, (w, h, 2))
    return data2D


def printInfoNumpy(data_j):
    print data_j.shape
    for idx in range(data_j.shape[2]):
        print idx,np.min(data_j[:,:,idx]),np.max(data_j[:,:,idx]),np.mean(data_j[:,:,idx]);
        # print 
    
def plotFloVals(data_j,out_file_j):
    vals_j=[np.sort(np.ravel(data_j[:,:,0])),np.sort(np.ravel(data_j[:,:,1]))];
    f=plt.figure();
    plt.subplot(1,2,1);
    plt.plot(vals_j[0])
    plt.subplot(1,2,2);
    plt.plot(vals_j[1])
    plt.savefig(out_file_j);

def getMaxMagnitude(data_j):
    mag=np.sqrt(np.power(data_j[:,:,0],2)+np.power(data_j[:,:,1],2));
    return np.min(mag);

def main():
    path_flo='/disk2/marchExperiments/debug_data/flo/flownets-pred-0000000(100).flo'
    flo_fn=readFlowFile(path_flo);
    printInfoNumpy(flo_fn);

    path_flo='/home/maheenrashid/Downloads/debugging_jacob/opticalflow/videos/JuanReneSerranoSemifinal_shoot_bow_u_cm_np1_fr_med_3/JuanReneSerranoSemifinal_shoot_bow_u_cm_np1_fr_med_3.avi_000101.flo'
    flo_df=readFlowFile(path_flo);
    printInfoNumpy(flo_df);

    plotFloVals(flo_fn,'/disk2/marchExperiments/debug_data/flownet_0.png')
    plotFloVals(flo_df,'/disk2/marchExperiments/debug_data/deepflow_0.png')


    return
    file_j='/home/maheenrashid/Downloads/opticalflow/videos/v_PlayingSitar_g01_c01/images/v_PlayingSitar_g01_c01.avi_000101.flo';
    im_j='/home/maheenrashid/Downloads/opticalflow/videos/v_PlayingSitar_g01_c01/images/v_PlayingSitar_g01_c01.avi_000001.tif'

    file_m='/media/maheenrashid/e5507fe3-2bff-4cbe-bc63-400de6deba92/maheen_data/flow_data/UCF-101/PlayingSitar/v_PlayingSitar_g01_c01/flownets-pred-0000000(100).flo';
    im_m='/disk2/februaryExperiments/training_jacob/training_data/v_PlayingSitar_g01_c01_0/image_1.tif';

    data_j=readFlowFile(file_j);
    data_m=readFlowFile(file_m);
    print getMaxMagnitude(data_j);
    print getMaxMagnitude(data_m);

    return
    out_file_j='/disk2/temp/x_j.png';
    plotFloVals(data_j[:,:,0],out_file_j)

    out_file_j='/disk2/temp/y_j.png';
    plotFloVals(data_j[:,:,1],out_file_j)

    out_file_m='/disk2/temp/x_m.png';
    plotFloVals(data_m[:,:,0],out_file_m)

    out_file_m='/disk2/temp/y_m.png';
    plotFloVals(data_m[:,:,1],out_file_m)

    

    
    print 'J';
    printInfoNumpy(data_j)
    print 'M';
    printInfoNumpy(data_m)

    print 'J_im';
    j_im=scipy.misc.imread(im_j);
    printInfoNumpy(j_im);
    print 'M_im';
    m_im=scipy.misc.imread(im_m);
    printInfoNumpy(m_im);

    


if __name__=='__main__':
    main();
    