import numpy as np;
import util;
import os;
import processOutput as po;
import subprocess;

def assignToFlowSoftTopN(data,clusters,n):
    channelNum = clusters.shape[0];
    data = np.reshape(data, (channelNum,-1));
    # print data.shape;
    sort_idx=np.argsort(data,axis=0);

    sort_idx=sort_idx[::-1,:];
    for idx in range(data.shape[1]):
        sort_idx=np.argsort(data[:,idx])[::-1];
        sort_idx=sort_idx[:n];
        temp_save=data[sort_idx,idx];
        data[:,idx]=0;
        data[sort_idx,idx]=temp_save;
    sum_data=np.sum(data,axis=0);
    sum_data=np.tile(sum_data,(data.shape[0],1));
    data=data/sum_data;

    x_arr=np.zeros((data.shape[1],));
    y_arr=np.zeros((data.shape[1],));

    for i in range(data.shape[1]):
        x_arr[i] = sum(data[:,i]*clusters[:,0]);
        y_arr[i] = sum(data[:,i]*clusters[:,1]);
    
    new_arr=np.zeros((20,20,2));
    new_arr[:,:,0]=np.reshape(x_arr,(20,20))
    new_arr[:,:,1]=np.reshape(y_arr,(20,20))
    
    return new_arr;

def getMatFromH5TopN(h5_file,img_size,C,n=10):
    if type(C)==type('str'):
        C=po.readClustersFile(C);
    np_data=po.readH5(h5_file);
    # print np_data.shape
    flow=assignToFlowSoftTopN(np_data.ravel(),C,n);
    flow_resize=po.resizeSP(flow,img_size);
    return flow_resize

def main():
    out_dir='/disk2/mayExperiments/eval_nC_zS_youtube';
    model_file='/disk3/maheen_data/ft_youtube_40_images_cluster_suppress_yjConfig/opt_noFix_conv1_conv2_conv3_conv4_conv5_llr__iter_50000.caffemodel';

    clusters_file='/disk3/maheen_data/youtube_train_40/clusters_100000.mat';
    flo_dir_meta=os.path.join(out_dir,'ft_youtube_model')
    flo_dir=os.path.join(flo_dir_meta,'flo');
    match_info_file=os.path.join(flo_dir,'match_info.txt');
    train_val_file='/disk3/maheen_data/ft_youtube_40_images_cluster_suppress_yjConfig/train_val_conv1_conv2_conv3_conv4_conv5.prototxt';
    out_dir_flo_viz_org=os.path.join(flo_dir_meta,'flo_viz');
    gpu=0;

    h5_files,img_files,img_sizes=po.parseInfoFile(match_info_file);
    file_names=util.getFileNames(img_files,ext=False)

    out_dirs_flo_viz=[out_dir_flo_viz_org];
    n_range=[5,10];
    n_range_str='_'.join([str(n) for n in n_range]);
    out_file_html=os.path.join(out_dir,'flo_n_'+n_range_str+'.html');


    for n in n_range:
        out_dir_flo=os.path.join(flo_dir_meta,'flo_n_'+str(n));
        out_dir_flo_viz=os.path.join(flo_dir_meta,'flo_n_'+str(n)+'_viz');
        out_file_sh=out_dir_flo_viz+'.sh'

        util.mkdir(out_dir_flo);
        util.mkdir(out_dir_flo_viz);
        
        out_dirs_flo_viz.append(out_dir_flo_viz)
        
        flo_files=[os.path.join(out_dir_flo,file_name+'.flo') for file_name in file_names];

        for h5_file,img_size,flo_file in zip(h5_files,img_sizes,flo_files):    
            flow_resize=getMatFromH5TopN(h5_file,img_size,clusters_file,n);
            util.writeFlowFile(flow_resize,flo_file);

        out_files_viz=[os.path.join(out_dir_flo_viz,file_name+'.png') for file_name in file_names];
        po.writeScriptToGetFloViz(flo_files,out_files_viz,out_file_sh);

        subprocess.call('sh '+out_file_sh,shell=True);

    img_paths_html=[];
    captions_html=[];
    for img_file,file_name in zip(img_files,file_names):
        row_curr=[img_file];
        caption_curr=[file_name];
        for out_dir_flo_curr in out_dirs_flo_viz:
            file_curr=os.path.join(out_dir_flo_curr,file_name+'.png');
            row_curr.append(file_curr);
            caption_curr.append(util.getFileNames([out_dir_flo_curr])[0]);
        row_curr=[util.getRelPath(f) for f in row_curr];
        img_paths_html.append(row_curr);
        captions_html.append(caption_curr);



    util.writeHTML(out_file_html,img_paths_html,captions_html);
    print out_file_html.replace('/disk2','vision3.cs.ucdavis.edu:1000/');





    print 'hello';


if __name__=='__main__':
    main();
