import os;
import scipy.misc;
import numpy as np;
import util
import multiprocessing;
import h5py

def assignToFlowSoft(data,clusters):
    channelNum = clusters.shape[0];
    data = np.reshape(data, (channelNum,-1));
    
    x_arr=np.zeros((data.shape[1],));
    y_arr=np.zeros((data.shape[1],));

    for i in range(data.shape[1]):
        x_arr[i] = sum(data[:,i]*clusters[:,0]);
        y_arr[i] = sum(data[:,i]*clusters[:,1]);
    
    new_arr=np.zeros((20,20,2));
    new_arr[:,:,0]=np.reshape(x_arr,(20,20))
    new_arr[:,:,1]=np.reshape(y_arr,(20,20))
    
    return new_arr;


def getFlowMat(h5_file,C):

    with h5py.File(h5_file,'r') as hf:
        data = hf.get('Outputs')
        np_data = np.array(data)
    flow=assignToFlowSoft(np_data.ravel(),C);
    return flow;

def resizeSP(flo,im_shape):
    gt_flo_sp=np.zeros((im_shape[0],im_shape[1],2));
    for layer_idx in range(flo.shape[2]):
        min_layer=np.min(flo[:,:,layer_idx]);
        max_layer=np.max(flo[:,:,layer_idx]);
        gt_flo_sp_curr=scipy.misc.imresize(flo[:,:,layer_idx],im_shape);
        # print np.max(gt_flo_sp_scurr)
        gt_flo_sp_curr=gt_flo_sp_curr/float(max(np.max(gt_flo_sp_curr),np.finfo(float).eps));
        gt_flo_sp_curr=gt_flo_sp_curr*abs(max_layer-min_layer);
        gt_flo_sp_curr=gt_flo_sp_curr-abs(min_layer);
        gt_flo_sp[:,:,layer_idx]=gt_flo_sp_curr;
    return gt_flo_sp;


def saveFloAsNpPred((h5_file,dir_flo_im,replace_paths,C,idx)):
    print idx;
    img_file=util.readLinesFromFile(h5_file.replace('.h5','.txt'))[0].strip();
    img_file=img_file.replace(replace_paths[0],replace_paths[1]);
    img_name=img_file[img_file.rindex('/')+1:img_file.rindex('.')];
    out_file_flo=os.path.join(dir_flo_im,img_name+'.npy');
        
    if os.path.exists(out_file_flo):
        return;

    flo=getFlowMat(h5_file,C);
    im=scipy.misc.imread(img_file);
    flo_resize=resizeSP(flo,im.shape);
    np.save(out_file_flo,flo_resize);


def script_saveFloAsNpPred(clusters_file,h5_files,dir_flo_im,replace_paths):

    with h5py.File(clusters_file,'r') as hf:
        C=np.array(hf.get('C'));
    C=C.T    
    print C.shape

    args=[];
    for idx_h5_file,h5_file in enumerate(h5_files):
        # if idx_h5_file%50==0:
        #     print idx_h5_file
        args.append((h5_file,dir_flo_im,replace_paths,C,idx_h5_file));

    print len(args)
    print args[0];
    # print multiprocessing.cpu_count()
    p=multiprocessing.Pool(multiprocessing.cpu_count());
    p.map(saveFloAsNpPred,args);

def bringToImageFrameSP(flo,im_shape):

    flo_shape_org=flo.shape;
    gt_flo_sp=resizeSP(flo,im_shape);
    gt_flo_sp[:,:,0]=gt_flo_sp[:,:,0]*im_shape[1]/float(flo_shape_org[1]);
    gt_flo_sp[:,:,1]=gt_flo_sp[:,:,1]*im_shape[0]/float(flo_shape_org[0]);
    
    return gt_flo_sp


def saveResizeFlo((i,flo_file,img_path,out_file_numpy)):
    print i;
    flo=util.readFlowFile(flo_file,flip=False)
    img=scipy.misc.imread(img_path);
    flo_new=bringToImageFrameSP(flo,img.shape);
    np.save(out_file_numpy,flo_new);

def main():

    
    return
    out_dir='/group/leegrp/maheen_data/flo_all_predictions_finetuned_model/results'
    dir_flo_im='/group/leegrp/maheen_data/flo_all_predictions_finetuned_model/flo_npy';
    util.mkdir(dir_flo_im);
    h5_file_list='/group/leegrp/maheen_data/flo_all_predictions_finetuned_model/h5_file_list.txt';

    util.mkdir(dir_flo_im);
    clusters_file='/group/leegrp/maheen_data/flo_all_predictions_finetuned_model/clusters.mat';
    vision3_path='/disk2/marchExperiments';
    hpc_path='/group/leegrp/maheen_data';


    # replace=['flo_all_predictions','flo_all_predictions_finetuned_model']
    h5_files=util.readLinesFromFile(h5_file_list);
    # h5_files=[file_curr.replace(replace[0],replace[1]) for file_curr in h5_files];
    print h5_files[0];
    # print len(h5_files);
    # print os.path.exists(h5_files[0]);
    # util.writeFile(h5_file_list,h5_files);

    script_saveFloAsNpPred(clusters_file,h5_files,dir_flo_im,[vision3_path,hpc_path])        

    return

    dir_meta_images='/group/leegrp/maheen_data/youtube';
    flo_files_list='/group/leegrp/maheen_data/youtube_list_flo_paths.txt';
    out_dir_numpy='/group/leegrp/maheen_data/youtube_resized_flo_npy_new';
    util.mkdir(out_dir_numpy);

    flo_files=util.readLinesFromFile(flo_files_list);
    img_files=[];
    args=[];
    
    for idx,flo_file in enumerate(flo_files):
        flo_name=flo_file[flo_file.rindex('/')+1:];
        video_name=flo_name[:flo_name.index('.')];
        just_name=flo_name[:flo_name.rindex('.')]
        img_name=just_name+'.jpg';
        
        img_path=os.path.join(dir_meta_images,video_name,'images_transfer',img_name);
        out_file_num=os.path.join(out_dir_numpy,just_name+'.npy');
        args.append((idx,flo_file,img_path,out_file_num));

    print len(args)
    print args[0];
    # print multiprocessing.cpu_count()
    p=multiprocessing.Pool(multiprocessing.cpu_count());
    p.map(saveResizeFlo,args);
  

        # saveResizeFlo(flo_file,img_path,out_file_numpy);






if __name__=='__main__':
    main();