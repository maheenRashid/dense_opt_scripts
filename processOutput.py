import h5py
import numpy as np;
import util;
import scipy.io;
import os;
import cv2
import multiprocessing;
import visualize;
import random;
import script_resizingFlos as srf;
import shutil;
import math;
import script_testJacob as stj;
import subprocess;
import matplotlib.pyplot as plt;
import math;
import cPickle as pickle;
import shutil;
import time;

NUM_THREADS=3;

def saveFloFileViz(flo_file,out_file_flo_viz,path_to_binary=None):
    if path_to_binary is None:
        path_to_binary='/home/maheenrashid/Downloads/flow-code/color_flow';

    sh_command=path_to_binary+' '+flo_file+' '+out_file_flo_viz;
    subprocess.call(sh_command,shell=True);
    return True;


def saveMatFloViz(flo_mat,out_file_viz,path_to_binary=None):
    if path_to_binary is None:
        path_to_binary='/home/maheenrashid/Downloads/flow-code/color_flow';


    x=random.random();
    out_file_temp=str(x)+'.flo';
    while os.path.exists(out_file_temp):
        x=random.random();
        out_file_temp=str(x)+'.flo';
    # pr
    util.writeFlowFile(flo_mat,out_file_temp);
    sh_command=path_to_binary+' '+out_file_temp+' '+out_file_viz;
    subprocess.call(sh_command,shell=True);
    os.remove(out_file_temp);

    # for input_file,output_file in zip(input_files,output_files):
    #     line=path_to_binary+' '+input_file+' '+output_file;
    #     lines.append(line);
    # util.writeFile(out_file_sh,lines);
    return True;




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

def assignToFlowSoftSize(data,clusters,size):
    channelNum = clusters.shape[0];
    data = np.reshape(data, (channelNum,-1));
    
    x_arr=np.zeros((data.shape[1],));
    y_arr=np.zeros((data.shape[1],));

    for i in range(data.shape[1]):
        x_arr[i] = sum(data[:,i]*clusters[:,0]);
        y_arr[i] = sum(data[:,i]*clusters[:,1]);
    
    new_arr=np.zeros((size[0],size[1],2));
    new_arr[:,:,0]=np.reshape(x_arr,size)
    new_arr[:,:,1]=np.reshape(y_arr,size)
    
    return new_arr;    

def readClustersFile(clusters_file):
    try:
        with h5py.File(clusters_file,'r') as hf:
            # print hf.keys();
            C=np.array(hf.get('C'));
        C=C.T
    except:
        C=scipy.io.loadmat(clusters_file);
        C=C['C'];
    return C;

def readH5(h5_file):
    with h5py.File(h5_file,'r') as hf:
        data = hf.get('Outputs')
        np_data = np.array(data)
        hf.close();
    return np_data;

def getImgFilesFromH5s(list_files):
    img_files=[];
    for list_file in list_files:
        img_file = util.readLinesFromFile(list_file.replace('.h5','.txt'))[0].strip();
        img_files.append(img_file);
    return img_files;

def saveOutputInfoFile(folder,out_file_text):
    if type(folder)!=type('str'):
        list_files=folder;
    else:
        list_files=util.getFilesInFolder(folder,'.h5');

    img_files=getImgFilesFromH5s(list_files);
    lines_to_write=[];
    for idx,img_file in enumerate(img_files):
        im=scipy.misc.imread(img_file);
        if len(im.shape)>2:
            str_size=[im.shape[0],im.shape[1],im.shape[2]];
        else:
            str_size=[im.shape[0],im.shape[1],1];
        str_size=[str(i) for i in str_size]
        line_curr=[list_files[idx],img_file]+str_size;
        line_curr=' '.join(line_curr);
        lines_to_write.append(line_curr)
    util.writeFile(out_file_text,lines_to_write);

def getOutputInfoMP((list_file,out_files_test)):
    img_file = util.readLinesFromFile(list_file.replace('.h5','.txt'))[0].strip();
    if out_files_test is not None and img_file not in out_files_test:
        line_curr=None;
    else: 
        im=scipy.misc.imread(img_file);
        if len(im.shape)>2:
            str_size=[im.shape[0],im.shape[1],im.shape[2]];
        else:
            str_size=[im.shape[0],im.shape[1],1];
        str_size=[str(i) for i in str_size]
        line_curr=[list_file,img_file]+str_size;
        line_curr=' '.join(line_curr);
    
    return line_curr;        

def saveOutputInfoFileMP(folder,out_file_text,out_files_test):
    if type(folder)!=type('str'):
        list_files=folder;
    else:
        list_files=util.getFilesInFolder(folder,'.h5');

    args=[];
    for list_file in list_files:
        args.append((list_file,out_files_test))
    p=multiprocessing.Pool(NUM_THREADS);
    lines_to_write=p.map(getOutputInfoMP,args);
    lines_to_write=[line_curr for line_curr in lines_to_write if line_curr is not None];

    util.writeFile(out_file_text,lines_to_write);    


def parseInfoFile(out_file_text,lim=None):
    lines=util.readLinesFromFile(out_file_text);
    if lim is not None:
        lines=lines[:lim];

    h5_files=[];
    img_files=[];
    img_sizes=[];
    
    for line_curr in lines:
        str_split=line_curr.split(' ');
        h5_files.append(str_split[0]);
        img_files.append(str_split[1]);
        img_sizes.append(tuple([int(i) for i in str_split[2:]]));

    return h5_files,img_files,img_sizes

def resizeSP(flo,im_shape):
    gt_flo_sp=np.zeros((im_shape[0],im_shape[1],2));

    for layer_idx in range(flo.shape[2]):
        min_layer=np.min(flo[:,:,layer_idx]);
        max_layer=np.max(flo[:,:,layer_idx]);
        gt_flo_sp_curr=scipy.misc.imresize(flo[:,:,layer_idx],im_shape);
        gt_flo_sp_curr=gt_flo_sp_curr/float(max(np.max(gt_flo_sp_curr),np.finfo(float).eps));
        gt_flo_sp_curr=gt_flo_sp_curr*abs(max_layer-min_layer);
        gt_flo_sp_curr=gt_flo_sp_curr-abs(min_layer);
        gt_flo_sp[:,:,layer_idx]=gt_flo_sp_curr;

    return gt_flo_sp;

def writeScriptToGetFloViz(input_files,output_files,out_file_sh,path_to_binary=None):
    if path_to_binary is None:
        path_to_binary='/home/maheenrashid/Downloads/flow-code/color_flow';

    lines=[];
    for input_file,output_file in zip(input_files,output_files):
        line=path_to_binary+' '+input_file+' '+output_file;
        lines.append(line);
    util.writeFile(out_file_sh,lines);

def saveH5AsNpy(h5_file,img_size,C,out_file):
    flow_resize=getMatFromH5(h5_file,img_size,C);
    np.save(out_file,flow_resize)

def saveH5AsFlo(h5_file,img_size,C,out_file):
    flow_resize=getMatFromH5(h5_file,img_size,C);
    util.writeFlowFile(flow_resize,out_file);

def saveH5AsFloMP((h5_file,img_size,C,out_file,idx)):
    print idx;
    saveH5AsFlo(h5_file,img_size,C,out_file);

def getMatFromH5(h5_file,img_size,C):
    if type(C)==type('str'):
        C=readClustersFile(C);
    np_data=readH5(h5_file);
    flow=assignToFlowSoft(np_data.ravel(),C);
    flow_resize=resizeSP(flow,img_size);
    return flow_resize

def makeFloHtml(out_file_html,img_files,flo_files,height=200,width=200):
    
    img_paths=[];
    captions=[];
    for img_file,flo_file in zip(img_files,flo_files):
        img_path=[];
        img_path.append(util.getRelPath(img_file,'/disk2'));
        img_path.append(util.getRelPath(flo_file,'/disk2'));
        img_paths.append(img_path);
        captions.append(['img','flo']);
    
    visualize.writeHTML(out_file_html,img_paths,captions,height,width);

def getIdxRange(total,thresh,num_parts):
    step=int(math.floor(total/float(num_parts)));
    
    thresh=min(step/2,thresh);

    idx_range_new=util.getIdxRange(total,step)
    rem = total%step;
    # print 'in getIdxRange',num_parts,idx_range_new,rem
    if 0<rem<thresh and len(idx_range_new)>2:
        idx_range_new=idx_range_new[:-2]+[idx_range_new[-1]]
        # num_parts=num_parts-1;
    num_parts=len(idx_range_new)-1;
    # print 'in getIdxRange',num_parts,idx_range_new,rem

    return idx_range_new,num_parts;

def splitImage((img_path,num_parts,out_dir)):
    thresh=50;
    img_name=img_path[img_path.rindex('/')+1:img_path.rindex('.')];
    ext=img_path[img_path.rindex('.'):];
    im=scipy.misc.imread(img_path);
    
    c_idx,num_parts_c=getIdxRange(im.shape[1],thresh,num_parts);
    r_idx,num_parts_r=getIdxRange(im.shape[0],thresh,num_parts);
    # print im.shape
    # print num_parts_r,num_parts_c,c_idx,r_idx;
    
    out_files=[];
    for r_idx_idx,start_r in enumerate(r_idx[:-1]):
        end_r = r_idx[r_idx_idx+1]
        for c_idx_idx,start_c in enumerate(c_idx[:-1]):
            end_c=c_idx[c_idx_idx+1];
            if len(im.shape)>2:
                im_curr=im[start_r:end_r,start_c:end_c,:];
            else:
                im_curr=im[start_r:end_r,start_c:end_c];
            # print start_r,end_r,start_c,end_c
            out_file_curr=os.path.join(out_dir,img_name+'_'+str(num_parts_r)+'_'+str(num_parts_c)+'_'+str(r_idx_idx)+'_'+str(c_idx_idx)+ext);
            scipy.misc.imsave(out_file_curr,im_curr);
            out_files.append(out_file_curr);

    return out_files

def splitImageOutPre((img_path,num_parts,out_pre)):
    thresh=50;
    img_name=img_path[img_path.rindex('/')+1:img_path.rindex('.')];
    ext=img_path[img_path.rindex('.'):];
    im=scipy.misc.imread(img_path);
    
    c_idx,num_parts_c=getIdxRange(im.shape[1],thresh,num_parts);
    r_idx,num_parts_r=getIdxRange(im.shape[0],thresh,num_parts);
    # print im.shape
    # print num_parts_r,num_parts_c,c_idx,r_idx;
    
    out_files=[];
    for r_idx_idx,start_r in enumerate(r_idx[:-1]):
        end_r = r_idx[r_idx_idx+1]
        for c_idx_idx,start_c in enumerate(c_idx[:-1]):
            end_c=c_idx[c_idx_idx+1];
            if len(im.shape)>2:
                im_curr=im[start_r:end_r,start_c:end_c,:];
            else:
                im_curr=im[start_r:end_r,start_c:end_c];
            # print start_r,end_r,start_c,end_c
            out_file_curr=out_pre+'_'+str(num_parts_r)+'_'+str(num_parts_c)+'_'+str(r_idx_idx)+'_'+str(c_idx_idx)+ext;
            scipy.misc.imsave(out_file_curr,im_curr);
            out_files.append(out_file_curr);

    return out_files

def makeTestFile(img_files,test_file):
    img_files=[file_curr+' 1' for file_curr in img_files];
    # test_file=os.path.join(out_dir,'test.txt');
    util.writeFile(test_file,img_files);
    
def stitchFlos((img_name,img_files,h5_files,img_sizes,C,out_dir,idx_img_name)):
    
    print idx_img_name
    if type(C)==type('str'):
        C=readClustersFile(C);

    file_parts=img_name.split('_');
    
    num_parts_r=int(file_parts[-2]);
    num_parts_c=int(file_parts[-1]);
    
    img_files_names=[file_curr[file_curr.rindex('/')+1:file_curr.rindex('.')] for file_curr in img_files];

    for r_idx_curr in range(num_parts_r):
        row_arr=[];
        for c_idx_curr in range(num_parts_c):
            file_rel_start=img_name+'_'+str(r_idx_curr)+'_'+str(c_idx_curr);
            h5_file=h5_files[img_files_names.index(file_rel_start)];
            img_size=img_sizes[img_files_names.index(file_rel_start)];
    
            im_curr=getMatFromH5(h5_file,img_size,C)
            # im_curr=scipy.misc.imread(img_files[img_files_names.index(file_rel_start)]);
            row_arr.append(im_curr);
        
        row_arr_np=np.hstack(tuple(row_arr));

        if r_idx_curr==0:
            img_yet=row_arr_np;
        else:
            img_yet=np.vstack((img_yet,row_arr_np))

    out_file_name=os.path.join(out_dir,img_name+'.flo');
    util.writeFlowFile(img_yet,out_file_name);
    # print img_yet.shape
    # out_file_name=os.path.join(out_dir,img_name+'.png');
    # scipy.misc.imsave(out_file_name,img_yet)
    
    return out_file_name
    
def script_getGrids(dir_meta,img_paths,grid_size,model_file,gpu,clusters_file):
    # make the dirs
    im_dir=os.path.join(dir_meta,'grid_im_'+str(grid_size));
    flo_restitch_dir=os.path.join(dir_meta,'gird_restitch_'+str(grid_size));
    h5_dir=os.path.join(dir_meta,'h5_'+str(grid_size));
    flo_viz_dir=os.path.join(dir_meta,'grid_flo_viz_'+str(grid_size));
    util.mkdir(im_dir)  
    util.mkdir(flo_restitch_dir)   
    util.mkdir(h5_dir) 
    util.mkdir(flo_viz_dir)

    # split the image
    args=[];
    for img_path in img_paths:
        args.append((img_path,grid_size,im_dir));

    p=multiprocessing.Pool(NUM_THREADS);
    out_files_all=p.map(splitImage,args);
    out_files_all=[os.path.join(im_dir,file_curr) for file_curr in os.listdir(im_dir)];
    
    # make the test file
    test_file=os.path.join(h5_dir,'test.txt');
    makeTestFile(out_files_all,test_file);
    
    # call the network
    command=stj.getCommandForTest(test_file,model_file,gpu);
    subprocess.call(command,shell=True);

    # get the h5 and img file correspondences
    out_file_info=os.path.join(h5_dir,'match_info.txt');
    saveOutputInfoFile(os.path.join(h5_dir,'results'),out_file_info);
    h5_files,img_files,img_sizes=parseInfoFile(out_file_info);

    # get the img_files to restitch
    img_files_to_restitch=[];
    img_names=[];
    for img_path in img_paths:
        img_name=img_path[img_path.rindex('/')+1:img_path.rindex('.')];
        img_name_sub=[file_curr for file_curr in os.listdir(im_dir) if file_curr.startswith(img_name+'_')];
        img_name_sub=img_name_sub[0];
        img_name_sub_split=img_name_sub.split('_');
        assert len(img_name_sub_split)==7;
        img_name_req='_'.join(img_name_sub_split[:-2]);
        img_files_to_restitch.append(img_name_req);
        img_names.append(img_name);

    # restitch the flos from the h5s
    args=[];
    for idx_img_name,img_name in enumerate(img_files_to_restitch):
        args.append((img_name,img_files,h5_files,img_sizes,clusters_file,flo_restitch_dir,idx_img_name));

    p=multiprocessing.Pool(NUM_THREADS);
    flo_files_restitch=p.map(stitchFlos,args);

    # save the flo im for restitched flos
    flo_files_restitch_im=[];
    for file_curr in flo_files_restitch:
        file_name=os.path.join(flo_viz_dir,file_curr[file_curr.rindex('/')+1:file_curr.rindex('.')]+'.png');
        flo_files_restitch_im.append(file_name);

    out_file_sh=flo_viz_dir+'.sh'
    writeScriptToGetFloViz(flo_files_restitch,flo_files_restitch_im,out_file_sh);
    subprocess.call('sh '+out_file_sh,shell=True);

def script_writeHTMLStitchedFlos(out_file_html,out_file,out_dir,grid_sizes=[1,2,4,8],grid_dir_pre='grid_flo_viz_'):
    img_paths=util.readLinesFromFile(out_file);
    
    viz_dirs=[os.path.join(out_dir,grid_dir_pre+str(num)) for num in grid_sizes];
    img_paths_html=[];
    captions=[];

    for img_path in img_paths:
        img_name=img_path[img_path.rindex('/')+1:img_path.rindex('.')];
        img_paths_html_curr=[util.getRelPath(img_path)];
        captions_curr=['im']
        for viz_dir in viz_dirs:
            print viz_dir,img_path
            img_path_curr=[os.path.join(viz_dir,file_curr) for file_curr in os.listdir(viz_dir) if file_curr.startswith(img_name)][0];
            img_paths_html_curr.append(util.getRelPath(img_path_curr));
            captions_curr.append(viz_dir[viz_dir.rindex('/')+1:]);
        img_paths_html.append(img_paths_html_curr);
        captions.append(captions_curr)
    
    visualize.writeHTML(out_file_html,img_paths_html,captions);


def script_writeHTMLStitchedFlos_wDirs(img_paths,out_file_html,viz_dirs):
    img_paths_html=[];
    captions=[];

    for img_path in img_paths:
        img_name=img_path[img_path.rindex('/')+1:img_path.rindex('.')];
        img_paths_html_curr=[util.getRelPath(img_path)];
        captions_curr=['im']
        for viz_dir in viz_dirs:
            print viz_dir,img_path
            # img_path_curr=[os.path.join(viz_dir,file_curr) for file_curr in os.listdir(viz_dir) if file_curr.startswith(img_name)][0];
            img_path_curr=os.path.join(viz_dir,img_name+'.png');
            img_paths_html_curr.append(util.getRelPath(img_path_curr));
            captions_curr.append(viz_dir[viz_dir.rindex('/')+1:]);
        img_paths_html.append(img_paths_html_curr);
        captions.append(captions_curr)
    
    visualize.writeHTML(out_file_html,img_paths_html,captions);


def fuseMagnitudes(flo_1,flo_2,alpha):
    assert 0<=alpha<=1;
    if type(flo_1)==type('str'):
        flo_1=util.readFlowFile(flo_1);
    if type(flo_2)==type('str'):
        flo_2=util.readFlowFile(flo_2);

    flo_mag_1=getFlowMag(flo_1);
    # print np.min(flo_mag_1),np.max(flo_mag_1);

    flo_mag_1=flo_mag_1/np.max(flo_mag_1);
    flo_mag_2=getFlowMag(flo_2);
    
    # print np.min(flo_mag_2),np.max(flo_mag_2);

    
    flo_mag_2=flo_mag_2/np.max(flo_mag_2);
    
    fused_mag=(flo_mag_1*alpha)+(flo_mag_2*(1-alpha));
    return fused_mag,flo_mag_1,flo_mag_2

def getFlowMag(flo_1):
    flo_mag_1=np.power(np.sum(np.power(flo_1,2),axis=2),0.5);
    return flo_mag_1;

def getHeatMap(arr,max_val=255):
    cmap = plt.get_cmap('jet')
    rgba_img = cmap(arr)
    rgb_img = np.delete(rgba_img, 3, 2)
    rgb_img = rgb_img*max_val
    # print rgb_img.shape,rgba_img.shape
    return rgb_img

def fuseAndSave(img,heatmap,alpha,out_file_curr):
    im=(img*alpha)+(heatmap*(1-alpha));
    # print im.shape,np.min(im),np.max(im);
    
    # out_file_curr=os.path.join(out_dir_fusion,img_name+'_fused_overlay.png');
    scipy.misc.imsave(out_file_curr,im);

def getPadBefAft(diff):
    pad_bef_r=diff/2;
    if diff%2==0:
        pad_aft_r=pad_bef_r;
    else:
        pad_aft_r=pad_bef_r+1;
    return pad_bef_r,pad_aft_r

def getPadTuple(width,filter_size,step_size):
    pad_bef_r=0;pad_aft_r=0;
    if (width-filter_size)%step_size!=0:
        div=(width-filter_size)/step_size;
        div=div+1;
        new_w=(div*step_size)+filter_size

        diff=new_w-width;
        # print new_w,diff
        pad_bef_r,pad_aft_r=getPadBefAft(diff);
        # pad_bef_r=diff/2;
        # if diff%2==0:
        #     pad_aft_r=pad_bef_r;
        # else:
        #     pad_aft_r=pad_bef_r+1;

    return (pad_bef_r,pad_aft_r);

def saveSlidingWindows((im_path,filter_size,step_size,out_file_pre,idx)):
    print idx;
    im=scipy.misc.imread(im_path);

    pad_r=getPadTuple(im.shape[0],filter_size[0],step_size);
    pad_c=getPadTuple(im.shape[1],filter_size[1],step_size);
    if len(im.shape)>2:
        im=np.pad(im,(pad_r,pad_c,(0,0)),'edge')
    else:
        im=np.pad(im,(pad_r,pad_c),'edge');
    start_r=0;
    idx_r=0;
    
    out_files=[];
    while start_r<im.shape[0]:
        start_c=0;
        idx_c=0;
        while start_c<im.shape[1]:

            end_r=start_r+filter_size[0];
            end_c=start_c+filter_size[1];
            crop_curr=im[start_r:end_r,start_c:end_c];
            out_file_curr=out_file_pre+'_'+str(idx_r)+'_'+str(idx_c)+'.png';
            scipy.misc.imsave(out_file_curr,crop_curr);
            
            out_files.append(out_file_curr);
            start_c=start_c+step_size;
            idx_c+=1;
    
        start_r=start_r+step_size;
        idx_r+=1;

    return out_files;

def averageMagnitudes((img_name,img_size_org,filter_size,step_size,img_files,h5_files,img_sizes,C,out_dir,idx_img_name)):
    print idx_img_name
    if type(C)==type('str'):
        C=readClustersFile(C);

    img_files_names=util.getFileNames(img_files,ext=False);
    r_pad=getPadTuple(img_size_org[0],filter_size,step_size);
    c_pad=getPadTuple(img_size_org[1],filter_size,step_size);
    
    

    new_shape=(img_size_org[0]+r_pad[0]+r_pad[1],img_size_org[1]+c_pad[0]+c_pad[1])
    assert (new_shape[0]-filter_size)%step_size==0
    num_parts_r = (new_shape[0]-filter_size)/step_size+1
    assert (new_shape[1]-filter_size)%step_size==0;
    num_parts_c = (new_shape[1]-filter_size)/step_size+1

    total_arr=np.zeros(new_shape);
    count_arr=np.zeros(new_shape);
    
    for r_idx_curr in range(num_parts_r):
        for c_idx_curr in range(num_parts_c):

            file_rel_start=img_name+'_'+str(r_idx_curr)+'_'+str(c_idx_curr);
            h5_file=h5_files[img_files_names.index(file_rel_start)];
            img_size=img_sizes[img_files_names.index(file_rel_start)];
    
            im_curr=getMatFromH5(h5_file,img_size,C)
            mag=getFlowMag(im_curr);
            
            start_r=r_idx_curr*step_size;
            start_c=c_idx_curr*step_size;
            
            end_r=start_r+filter_size;
            end_c=start_c+filter_size;
            
            assert end_r-start_r==mag.shape[0];
            assert end_c-start_c==mag.shape[1];

            total_arr[start_r:end_r,start_c:end_c]=total_arr[start_r:end_r,start_c:end_c]+mag;
            count_arr[start_r:end_r,start_c:end_c]=count_arr[start_r:end_r,start_c:end_c]+1;

    avg_arr=total_arr/count_arr;

    out_file_name=os.path.join(out_dir,img_name+'.npy');
    np.save(out_file_name,avg_arr);
    # util.writeFlowFile(total_arr/count_arr,out_file_name);
    return out_file_name
    
def saveHeatMapsAverage((img_file,flo_file,out_file_curr,alpha,idx)):
    print idx;
    flo=np.load(flo_file);
    im=scipy.misc.imread(img_file);
    diff_r=flo.shape[0]-im.shape[0];
    diff_c=flo.shape[1]-im.shape[1];
    for dim in range(2):
        diff=flo.shape[dim]-im.shape[dim];
        pad_bef,pad_aft=getPadBefAft(diff);
        if dim==0:
            flo=flo[pad_bef:flo.shape[dim]-pad_aft,:];
        else:
            flo=flo[:,pad_bef:flo.shape[dim]-pad_aft];

    flo=flo/np.max(flo);
    heatmap=getHeatMap(flo);
    if len(im.shape)==2:
        im=np.dstack((im,im,im));        
    fuseAndSave(im,heatmap,alpha,out_file_curr);
        
    # print flo.shape,im.shape;

    # if diff_r%


def script_getSlidingWindows(dir_meta,img_paths,filter_size,step_size,model_file,gpu,clusters_file,alpha=0.5):
    # make the dirs
    im_dir=os.path.join(dir_meta,'sw_im_'+str(filter_size)+'_'+str(step_size));
    flo_restitch_dir=os.path.join(dir_meta,'sw_restitch_'+str(filter_size)+'_'+str(step_size));
    h5_dir=os.path.join(dir_meta,'sw_h5_'+str(filter_size)+'_'+str(step_size));
    flo_viz_dir=os.path.join(dir_meta,'sw_flo_viz_'+str(filter_size)+'_'+str(step_size));
    util.mkdir(im_dir)  
    util.mkdir(flo_restitch_dir)   
    util.mkdir(h5_dir) 
    util.mkdir(flo_viz_dir)

    img_names=util.getFileNames(img_paths,ext=False);

    # # split the image
    # args=[];
    # out_files_all=[];
    # for idx,img_path in enumerate(img_paths):
    #     out_file_pre=os.path.join(im_dir,img_names[idx]);
    #     # if os.path.exists(out_file_pre):
    #     #     continue

    #     args.append((img_path,[filter_size,filter_size],step_size,out_file_pre,idx));
    
    # p=multiprocessing.Pool(NUM_THREADS);
    # out_files_all=p.map(saveSlidingWindows,args);
    # out_files_all=[out_file_curr for out_files_list in out_files_all for out_file_curr in out_files_list];

    # # make the test file
    # test_file=os.path.join(h5_dir,'test.txt');
    # makeTestFile(out_files_all,test_file);
    
    # # call the network
    # command=stj.getCommandForTest(test_file,model_file,gpu);
    # subprocess.call(command,shell=True);

    # # get the h5 and img file correspondences
    # out_file_info=os.path.join(h5_dir,'match_info.txt');
    # saveOutputInfoFile(os.path.join(h5_dir,'results'),out_file_info);

    # h5_files,img_files,img_sizes=parseInfoFile(out_file_info);

    # # get the img_files to restitch
    # img_files_to_restitch=[];
    # for idx_img_path,img_path in enumerate(img_paths):
    #     img_name=img_names[idx_img_path];
    #     img_name_sub=[file_curr for file_curr in os.listdir(im_dir) if file_curr.startswith(img_name+'_')];
    #     img_name_sub=img_name_sub[0];
    #     img_name_sub_split=img_name_sub.split('_');
    #     assert len(img_name_sub_split)==5;
    #     img_name_req='_'.join(img_name_sub_split[:-2]);
    #     img_files_to_restitch.append(img_name_req);

    # # restitch the flos from the h5s
    # args=[];
    # for idx_img_name,img_name in enumerate(img_files_to_restitch):
    #     img_path=img_paths[idx_img_name];
    #     img_size_org=scipy.misc.imread(img_path).shape;
    #     args.append((img_name,img_size_org,filter_size,step_size,img_files,h5_files,img_sizes,clusters_file,flo_restitch_dir,idx_img_name));

    # p=multiprocessing.Pool(NUM_THREADS);
    # flo_files_restitch=p.map(averageMagnitudes,args)
    
    # flo_files_restitch=[os.path.join(flo_restitch_dir,file_curr) for file_curr in util.getEndingFiles(flo_restitch_dir,'.npy')];
    args=[];
    for idx,img_file in enumerate(img_paths):
        img_name=img_names[idx];
        # flo_file=flo_files_restitch[idx]
        flo_file=os.path.join(flo_restitch_dir,img_name+'.npy');
        out_file_curr=os.path.join(flo_viz_dir,img_name+'.png');
        args.append((img_file,flo_file,out_file_curr,alpha,idx));

    p=multiprocessing.Pool(NUM_THREADS);
    p.map(saveHeatMapsAverage,args)


def getRelevantFilesFromMatchFile(out_file_info,img_name):
    # print 'out_file_info',out_file_info
    h5_files,img_files,img_sizes=parseInfoFile(out_file_info);


    img_names=util.getFileNames(img_files,ext=False);
    idx_rel=[idx for idx,file_curr in enumerate(img_names) if file_curr.startswith(img_name)];
    
    h5_files=[h5_files[idx] for idx in idx_rel];
    img_files=[img_files[idx] for idx in idx_rel];
    img_sizes=[img_sizes[idx] for idx in idx_rel];
    return h5_files,img_files,img_sizes


def script_saveFlos(img_paths,dir_test,gpu,model_file,clusters_file,overwrite=False,train_val_file=None):
    test_file=os.path.join(dir_test,'test.txt');
    out_file_info=os.path.join(dir_test,'match_info.txt');
    out_dir_flo=os.path.join(dir_test,'flo_files');
    util.mkdir(out_dir_flo);

    C=readClustersFile(clusters_file);
    # print overwrite

    if (not os.path.exists(test_file)) or overwrite:
        makeTestFile(img_paths,test_file);

        # call the network
        # print train_val_file
        command=stj.getCommandForTest(test_file,model_file,gpu,train_val_file=train_val_file);
        # print command
        # raw_input();

        # print command;
        # return
        subprocess.call(command,shell=True);


    # print overwrite;
    # raw_input();
    # # get the h5 and img file correspondences
    if (not os.path.exists(out_file_info)) or overwrite:
        # print 'hello'
        saveOutputInfoFileMP(os.path.join(dir_test,'results'),out_file_info,img_paths)
        # saveOutputInfoFile(os.path.join(dir_test,'results'),out_file_info);

    h5_files,img_files,img_sizes=parseInfoFile(out_file_info);
    print len(h5_files)
    out_files_flo=[os.path.join(out_dir_flo,file_curr+'.flo') for file_curr in util.getFileNames(img_files,ext=False)];
    args=[];
    for idx in range(len(h5_files)):
        if not overwrite:
            if os.path.exists(out_files_flo[idx]):
                continue;
        args.append((h5_files[idx],img_sizes[idx],C,out_files_flo[idx],idx))

    print len(args);
    p=multiprocessing.Pool(NUM_THREADS)
    p.map(saveH5AsFloMP,args)


def script_saveFlosAndViz(img_paths,dir_test,flo_viz_dir,gpu,model_file,clusters_file,train_val_file=None,overwrite=False):
    # h5_dir=os.path.join(dir_test,'h5');
    # flo_dir=os.path.join(dir_test,'flo');
    # flo_viz_dir=os.path.join(dir_test,'flo_viz');
    # util.mkdir(h5_dir);
    # print train_val_file,'script_saveFlosAndViz';
    script_saveFlos(img_paths,dir_test,gpu,model_file,clusters_file,train_val_file=train_val_file,overwrite=overwrite)
    
    flo_dir=os.path.join(dir_test,'flo_files');
    
    flo_files=[os.path.join(flo_dir,file_curr) for file_curr in util.getFilesInFolder(flo_dir,'.flo')];
    flo_files_names=util.getFileNames(flo_files,ext=False);
    flo_files_viz=[os.path.join(flo_viz_dir,file_curr+'.png') for file_curr in flo_files_names];

    out_file_sh=flo_viz_dir+'.sh'
    writeScriptToGetFloViz(flo_files,flo_files_viz,out_file_sh);
    subprocess.call('sh '+out_file_sh,shell=True);




def stitchH5s(img_name,h5_files,img_files,C):
    if type(C)==type('str'):
        C=readClustersFile(C);
    num_parts_r=int(img_files[0].split('_')[-4]);
    num_parts_c=int(img_files[0].split('_')[-3]);

    img_files_names=util.getFileNames(img_files);
    file_pre=img_name+'_'+str(num_parts_r)+'_'+str(num_parts_c);

    for r_idx in range(num_parts_r):
        row_curr=[];
        for c_idx in range(num_parts_c):
            file_start=file_pre+'_'+str(r_idx)+'_'+str(c_idx);
            
            idx=[idx for idx,file_curr in enumerate(img_files_names) if file_curr.startswith(file_start)];
            # print len(idx),img_files_names
            assert len(idx)==1;
            idx=idx[0];

            np_data_curr=readH5(h5_files[idx]);
            np_data_curr=np_data_curr[0];
            # print 'np_data_curr.shape',np_data_curr.shape
            np_data_curr=np.transpose(np_data_curr,(1,2,0));
            # print 'np_data_curr.shape',np_data_curr.shape
            row_curr.append(np_data_curr)
        
        row_data=np.hstack(tuple(row_curr));
        # print 'row_data.shape',row_data.shape
        if r_idx==0:
            data_block=row_data;
        else:
            data_block=np.vstack((data_block,row_data));

    return data_block


def script_pyramidFuse((match_files,img_name,im_size,clusters_file,out_dir_flo,out_dir_flo_viz,idx)):
    try:
        print idx;
        flo_file=os.path.join(out_dir_flo,img_name+'.flo')
        # if os.path.exists(flo_file):
        #     return;
        # img_name=util.getFileNames([img_path],ext=False)[0];
        C=readClustersFile(clusters_file);

        pyramid=[];
        for match_file in match_files:
            h5_files,img_files,img_sizes=getRelevantFilesFromMatchFile(match_file,img_name);
            # print img_files,img_name
            h5_block=stitchH5s(img_name,h5_files,img_files,clusters_file);    
            pyramid.append(h5_block);

        size_arr=[pyr_curr.shape[0] for pyr_curr in pyramid]    
        max_size=max(size_arr);
        max_idx=size_arr.index(max_size);
        for idx,pyr_curr in enumerate(pyramid):
            if pyr_curr.shape[0]<max_size:
                pyr_curr=cv2.resize(pyr_curr,(max_size,max_size), interpolation=cv2.INTER_NEAREST);
                pyramid[idx]=pyr_curr;

        total=0;
        for pyr_curr in pyramid:
            if type(total)==type(0):
                total=pyr_curr;
            else:
                total=total+pyr_curr;

        avg=total/float(len(pyramid));
        # print np.sum(avg,axis=2);

        # im=scipy.misc.imread(img_path);
        # im_size=(im.shape[0],im.shape[1]);
        # C=readClustersFile(clusters_file);
        flow=assignToFlowSoftSize(np.transpose(avg,(2,0,1)).ravel(),C,(avg.shape[0],avg.shape[0]));
        flow_resize=resizeSP(flow,im_size);
        # print flow_resize.shape,im_size
        
        util.writeFlowFile(flow_resize,flo_file);
        # pyr_file=os.path.join(out_dir_flo,img_name+'.npy')
        # pyr_npy=np.array(pyramid);
        # print pyr_npy.shape;
        # np.save(pyr_file,pyr_npy);

        if out_dir_flo_viz is not None:
            out_file_viz=os.path.join(out_dir_flo_viz,img_name+'.png');
            command='/home/maheenrashid/Downloads/flow-code/color_flow '+flo_file+' '+out_file_viz;
            subprocess.call(command,shell=True);
    except:
        print 'could not make pyramid for ',img_name
        pass;

def script_saveFloPyramidsAndAverage(dir_meta,img_paths,grid_sizes,model_file,gpu,clusters_file,append_folder=True,overwrite=False):
    
    img_names=util.getFileNames(img_paths,ext=False);
    dirs_to_del=[];
    for grid_size in grid_sizes:
        im_dir=os.path.join(dir_meta,'grid_im_'+str(grid_size));
        h5_dir=os.path.join(dir_meta,'h5_'+str(grid_size));
        
        # split the image
        out_file_info=os.path.join(h5_dir,'match_info.txt');
        if overwrite or not os.path.exists(out_file_info):
            util.mkdir(im_dir)  
            util.mkdir(h5_dir) 
            args=[];
            for img_path,img_name in zip(img_paths,img_names):
                if append_folder:
                    folder_last=img_path[:img_path.rindex('/')];
                    folder_last=folder_last[folder_last.rindex('/')+1:];
                    out_pre=os.path.join(im_dir,folder_last+'_'+img_name);
                else:
                    out_pre=os.path.join(im_dir,img_name);
                
                args.append((img_path,grid_size,out_pre));
            print 'splitting image grid_size',grid_size,len(args);
            p=multiprocessing.Pool(NUM_THREADS);
            out_files_all=p.map(splitImageOutPre,args);
            out_files_all=[file_curr for file_list in out_files_all for file_curr in file_list];
            print len(out_files_all),out_files_all[0];
            print out_files_all
            # out_files_all=[os.path.join(im_dir,file_curr) for file_curr in os.listdir(im_dir)];
        
           # make the test file
            print 'splitting image grid_size',grid_size,len(args);
            test_file=os.path.join(h5_dir,'test.txt');
            makeTestFile(out_files_all,test_file);
            
            # call the network
            command=stj.getCommandForTest(test_file,model_file,gpu);
            subprocess.call(command,shell=True);

            # get the h5 and img file correspondences
            
            saveOutputInfoFile(os.path.join(h5_dir,'results'),out_file_info);
            
            # delete image files
        dirs_to_del.append(im_dir);
        dirs_to_del.append(h5_dir);
            # shutil.rmtree(im_dir);



    match_files=[];
    for grid_size in grid_sizes:
        match_files.append(os.path.join(dir_meta,'h5_'+str(grid_size),'match_info.txt'));
    print match_files
    
    str_grid=[str(grid_size) for grid_size in grid_sizes];
    str_grid='_'.join(str_grid);    
    out_dir_flo=os.path.join(dir_meta,'prob_fuse_'+str_grid);
    util.mkdir(out_dir_flo);

    idx=0;
    args=[]
    for img_path,img_name in zip(img_paths,img_names):
        if append_folder:
            folder_last=img_path[:img_path.rindex('/')];
            folder_last=folder_last[folder_last.rindex('/')+1:];
            img_path_ac=folder_last+'_'+img_name;
        else:
            img_path_ac=img_name
        im=scipy.misc.imread(img_path);
        im_size=im.shape;
        out_dir_flo_viz=None;
        args.append((match_files,img_path_ac,im_size,clusters_file,out_dir_flo,None,idx));
        idx+=1

    p=multiprocessing.Pool(NUM_THREADS);
    p.map(script_pyramidFuse,args);

    # for dir_to_del in dirs_to_del:
    #     if os.path.exists(dir_to_del):
    #         shutil.rmtree(dir_to_del);
    # s#         dirs_to_del.append(h5_dir);
            
    # for arg in args:
    #     print arg
    #     script_pyramidFuse(arg);

    # script_pyramidFuse((match_files,img_path,img_size,clusters_file,out_dir_flo,out_dir_flo_viz,idx))


def script_saveFloPyramidsAndAverageEfficient(dir_meta,img_paths,grid_sizes,model_file,gpu,clusters_file,append_folder=True,overwrite=False):
    
    img_names=util.getFileNames(img_paths,ext=False);

    str_grid=[str(grid_size) for grid_size in grid_sizes];
    str_grid='_'.join(str_grid);    

    out_dir_flo=os.path.join(dir_meta,'prob_fuse_'+str_grid);
    util.mkdir(out_dir_flo);

    # if prob_fuse is already filled return
    if not overwrite:
        # find the images left over;
        img_paths_new=[];
        for idx,img_path in enumerate(img_paths):
            img_name=img_names[idx];
            if append_folder:
                folder_last=img_path[:img_path.rindex('/')];
                folder_last=folder_last[folder_last.rindex('/')+1:];
                img_pre=folder_last+'_'+img_name;
            else:
                img_pre=img_name;

            if not os.path.exists(os.path.join(out_dir_flo,img_pre+'.flo')):
                img_paths_new.append(img_paths[idx]);

        img_paths=img_paths_new[:];
        img_names=util.getFileNames(img_paths,ext=False);
    
    if len(img_paths)==0:
        return


    dirs_to_del=[];
    out_files_all_test=[];
    info_for_split=[];
    for grid_size in grid_sizes:
        im_dir=os.path.join(dir_meta,'grid_im_'+str(grid_size));
        h5_dir=os.path.join(dir_meta,'h5_'+str(grid_size));
        util.mkdir(im_dir)  
        util.mkdir(h5_dir);
        dirs_to_del.append(im_dir);
        out_file_info_curr=os.path.join(h5_dir,'match_info.txt');
        info_for_split.append((out_file_info_curr,im_dir));
        

        args=[];
        for img_path,img_name in zip(img_paths,img_names):
            if append_folder:
                folder_last=img_path[:img_path.rindex('/')];
                folder_last=folder_last[folder_last.rindex('/')+1:];
                img_pre=folder_last+'_'+img_name;
            else:
                img_pre=img_name;
            out_pre=os.path.join(im_dir,img_pre);
            args.append((img_path,grid_size,out_pre));

        print 'splitting image grid_size',grid_size,len(args);
        p=multiprocessing.Pool(NUM_THREADS);
        out_files_all=p.map(splitImageOutPre,args);
        out_files_all=[file_curr for file_list in out_files_all for file_curr in file_list];
        out_files_all_test=out_files_all_test+out_files_all;
        
    print 'len(args);',len(args);
    
    dir_test=os.path.join(dir_meta,'h5_'+str_grid);
    util.mkdir(dir_test);

    test_file=os.path.join(dir_test,'test.txt');
    makeTestFile(out_files_all_test,test_file);

    # call the network
    command=stj.getCommandForTest(test_file,model_file,gpu,min(len(out_files_all_test),100));
    subprocess.call(command,shell=True);

    # get the h5 and img file correspondences
    out_file_info=os.path.join(dir_test,'match_info.txt');
    saveOutputInfoFileMP(os.path.join(dir_test,'results'),out_file_info,out_files_all_test)
    
    lines=util.readLinesFromFile(out_file_info);
    
    dict_match={};
    [out_file_infos,match_files]=zip(*info_for_split);
    for match_file_curr in match_files:
        dict_match[match_file_curr]=[]

    for line in lines:
        line_split=line.split(' ');
        line_rel=line_split[1];
        for match_file_curr in match_files:
            if line_rel.startswith(match_file_curr):
                dict_match[match_file_curr].append(line);

    for match_file_curr in dict_match.keys():
        out_file_curr=out_file_infos[match_files.index(match_file_curr)];
        util.writeFile(out_file_curr,dict_match[match_file_curr]);

    # fuse 
    idx=0;
    args=[]
    [out_file_infos,match_files]=zip(*info_for_split);
    for img_path,img_name in zip(img_paths,img_names):
        if append_folder:
            folder_last=img_path[:img_path.rindex('/')];
            folder_last=folder_last[folder_last.rindex('/')+1:];
            img_path_ac=folder_last+'_'+img_name;
        else:
            img_path_ac=img_name
        im=scipy.misc.imread(img_path);
        im_size=im.shape;
        out_dir_flo_viz=None;
        args.append((out_file_infos,img_path_ac,im_size,clusters_file,out_dir_flo,None,idx));
        idx+=1

    p=multiprocessing.Pool(NUM_THREADS);
    p.map(script_pyramidFuse,args);

    # delete
    for dir_to_del in dirs_to_del:
        if os.path.exists(dir_to_del):
            shutil.rmtree(dir_to_del);

def main():
    h5_file='/disk2/mayExperiments/flow_resolution_scratch/im_viz_padding_ft_nC_sZ_youtube/large_0.707106781187/COCO_val2014_000000000143_pred_flo/results/109.h5';
    data=readH5(h5_file)[0]
    
    print data.shape
    print data[:,10,10];

    return


    # out_dir='/disk2/aprilExperiments/flo_subdivision_actual'
    # out_file=os.path.join(out_dir,'list_of_im.txt');
    # img_paths=util.readLinesFromFile(out_file);
    
    # grid_sizes=[1];
    # out_dir_pre='grid_flo_viz_'
    # im_post='_1_1';

    # # grid_sizes=[1,2,4,5];
    # # out_dir_pre='prob_fuse_flo_viz_';
    # # im_post=''

    # grid_sizes_str=[str(grid_size) for grid_size in grid_sizes];
    # grid_sizes_str='_'.join(grid_sizes_str);

    
    # viz_dir='/disk2/aprilExperiments/flo_subdivision_actual'
    # # out_dir_flo_viz=os.path.join(out_dir,'grid_flo_viz_'+grid_sizes_str);
    # out_dir_flo_viz=os.path.join(out_dir,out_dir_pre+grid_sizes_str);
    

    

    # out_dir_ac='/disk1/maheen_data/mayExperiments/new_model_flo_training_50000';
    
    # prob_folder_only='prob_fuse_viz_'+grid_sizes_str
    # prob_folder=os.path.join(out_dir_ac,prob_folder_only);
    # print prob_folder
    # sym_path='/disk2/temp/'+prob_folder_only;

    # cmd='';
    # if os.path.exists(sym_path):
    #     cmd='rm '+sym_path+';'
    # cmd=cmd+'ln -s '+prob_folder+' '+sym_path;
    # print cmd;
    # subprocess.call(cmd,shell=True)

    # viz_dirs=[out_dir_flo_viz,sym_path];

    # out_file_html=os.path.join(out_dir,'visualizing_fuse_diff_models_'+grid_sizes_str+'.html');
    # print out_file_html
    # print viz_dirs
    # # script_writeHTMLStitchedFlos_wDirs(img_paths,out_file_html,viz_dirs)

    # img_paths_html=[];
    # captions=[];

    # for img_path in img_paths:
    #     img_name=img_path[img_path.rindex('/')+1:img_path.rindex('.')];
    #     img_paths_html_curr=[util.getRelPath(img_path)];
    #     captions_curr=['im']
    #     for idx_viz_dir,viz_dir in enumerate(viz_dirs):
    #         # print viz_dir,img_path
    #         # img_path_curr=[os.path.join(viz_dir,file_curr) for file_curr in os.listdir(viz_dir) if file_curr.startswith(img_name)][0];
    #         if idx_viz_dir==0:
    #             img_path_curr=os.path.join(viz_dir,img_name+im_post+'.png');
    #                 # +'_1_1.png');
    #         else:
    #             img_path_curr=os.path.join(viz_dir,'train2014_'+img_name+'.png');

    #         img_paths_html_curr.append(util.getRelPath(img_path_curr));
    #         captions_curr.append(viz_dir[viz_dir.rindex('/')+1:]);
    #     img_paths_html.append(img_paths_html_curr);
    #     captions.append(captions_curr)
    
    # visualize.writeHTML(out_file_html,img_paths_html,captions);



    # return
    # grid_sizes=[1,2,4,5];
    # # grid_sizes=[1];
    # out_dir_meta='/disk1/maheen_data/mayExperiments/new_model_flo_training_50000';
    # util.mkdir(out_dir_meta);
    # # model_file='/home/maheenrashid/Downloads/debugging_jacob/optical_flow_prediction_test/examples/opticalflow/final.caffemodel'
    # model_file='/disk2/mayExperiments/finetuning_youtube_hmdb_llr/OptFlow_youtube_hmdb_iter_50000.caffemodel';
    # clusters_file='/home/maheenrashid/Downloads/debugging_jacob/optical_flow_prediction_test/examples/opticalflow/clusters.mat';
    # gpu=0;
    # util.mkdir(out_dir_meta);
    # # im_list_file=os.path.join(out_dir_meta,'list_of_im.txt')
    # # util.writeFile(im_list_file,img_paths)
    # # img_paths=util.readLinesFromFile(im_list_file);
    
    # out_file='/disk2/aprilExperiments/flo_subdivision_actual/list_of_im.txt'
    # img_paths=util.readLinesFromFile(out_file);
    # # img_paths=img_paths[:10];

    # print len(img_paths);
    # t=time.time();
    # script_saveFloPyramidsAndAverageEfficient(out_dir_meta,img_paths,grid_sizes,model_file,gpu,clusters_file,append_folder=True,overwrite=False);
    # # script_saveFloPyramidsAndAverage(out_dir_meta,img_paths,grid_sizes,model_file,gpu,clusters_file,append_folder=True,overwrite=True);
    # print time.time()-t;



    # # flo_dir='/disk1/maheen_data/mayExperiments/new_model_flo_training/prob_fuse_1_2_4_5';
    # # flo_viz_dir='/disk1/maheen_data/mayExperiments/new_model_flo_training/prob_fuse_viz_1_2_4_5';
    # str_grid='_'.join([str(val) for val in grid_sizes]);
    # # flo_dir='/disk1/maheen_data/mayExperiments/new_model_flo_training/prob_fuse_'+str_grid;
    # # flo_viz_dir='/disk1/maheen_data/mayExperiments/new_model_flo_training/prob_fuse_viz_'+str_grid;

    # flo_dir=os.path.join(out_dir_meta,'prob_fuse_'+str_grid);
    # flo_viz_dir=os.path.join(out_dir_meta,'prob_fuse_viz_'+str_grid);


    # util.mkdir(flo_viz_dir);
    # flo_paths=util.getFilesInFolder(flo_dir,'.flo');
    # flo_viz_paths=[file_curr.replace('.flo','.png').replace(flo_dir,flo_viz_dir) for file_curr in flo_paths];
    # out_file_sh=flo_viz_dir+'.sh';
    # writeScriptToGetFloViz(flo_paths,flo_viz_paths,out_file_sh);
    # subprocess.call('sh '+out_file_sh,shell=True);
    
    
    
    # return
    out_dir_meta='/disk1/maheen_data/mayExperiments/model_50000_flo'
    batches=[os.path.join(out_dir_meta,dir_curr) for dir_curr in os.listdir(out_dir_meta) if os.path.isdir(os.path.join(out_dir_meta,dir_curr))];
    print len(batches);
    batch_size=300;
    grid_sizes=[1,2,4,5];

    model_file='/disk2/mayExperiments/finetuning_youtube_hmdb_llr/OptFlow_youtube_hmdb_iter_50000.caffemodel';
    clusters_file='/home/maheenrashid/Downloads/debugging_jacob/optical_flow_prediction_test/examples/opticalflow/clusters.mat';
    gpu=1;

    str_grid='_'.join([str(val) for val in grid_sizes]);
    flo_folder='prob_fuse_1_2_4_5'
    done=[];
    not_done=[];

    # flo_folder='h5_1_2_4_5'
    # for batch_curr in batches:
    #     flo_folder_curr=os.path.join(batch_curr,flo_folder)
    #     if os.path.isdir(flo_folder_curr):
    #         done.append(batch_curr);
    #     else:
    #         not_done.append(batch_curr);

    # print len(done);
    # print len(not_done);

    # return
    for batch_curr in batches:
        flo_folder_curr=os.path.join(batch_curr,flo_folder)
        if os.path.isdir(flo_folder_curr):
            files=util.getFilesInFolder(flo_folder_curr,'.flo');
            if len(files)==batch_size:
                done.append(batch_curr);
                continue;
        not_done.append(batch_curr);

    print len(done);
    print len(not_done);

    # h5_dir='h5_1_2_4_5'
    # path_to_mv=util.escapeString('/media/maheenrashid/Seagate Backup Plus Drive/maheen_data/mayExperiments/model_50000_flo');
    # path_to_sh_pre=os.path.join(out_dir_meta,'mv_h5');
    # path_to_sh_meta=os.path.join(out_dir_meta,'mv_h5_meta.sh');

    # commands=[];
    # for done_curr in done:
    #     h5_path=os.path.join(done_curr,h5_dir);
    #     if os.path.exists(h5_path):
    #         batch_dir=done_curr[done_curr.rindex('/')+1:];
    #         new_path=os.path.join(path_to_mv,batch_dir);
    #         str_command='';
    #         if not os.path.exists(new_path):
    #             str_command='mkdir '+new_path+';';
    #         str_command=str_command+'mv '+h5_path+' '+new_path+ '/;';
    #         commands.append(str_command);

    # idx_range=util.getIdxRange(len(commands),len(commands)/12);
    # sh_files=[];
    # for idx_idx,start_idx in enumerate(idx_range[:-1]):
    #     commands_curr=commands[start_idx:idx_range[idx_idx+1]];
    #     sh_curr=path_to_sh_pre+'_'+str(idx_idx)+'.sh'
    #     commands_curr=['#!/bin/sh']+commands_curr;
    #     util.writeFile(sh_curr,commands_curr);
    #     print sh_curr;
    #     sh_files.append(sh_curr);

    # with open(path_to_sh_meta,'wb') as f:
    #     f.write('#!/bin/sh\n');
    #     for file_curr in sh_files:
    #         f.write(file_curr+' &\n');

    # util.writeFile(path_to_sh,commands)






    # return  
    # sort not_done;
    vals=[int(not_done_curr[not_done_curr.rindex('_')+1:]) for not_done_curr in not_done];
    sort_idx=np.argsort(vals);
    not_done=[not_done[idx] for idx in sort_idx];
    print not_done[:10];
    not_done=not_done[:-1];

    for not_done_curr in not_done:
        # [25:]:
        print not_done_curr
        list_file=not_done_curr+'.txt';
        img_paths=util.readLinesFromFile(list_file);
        try:
            script_saveFloPyramidsAndAverageEfficient(not_done_curr,img_paths,grid_sizes,model_file,gpu,clusters_file,append_folder=True,overwrite=False);
        except:
            print not_done_curr,' is problematic';
            continue
        print not_done_curr,' is complete';
        

    return
    dir_training='/disk2/mayExperiments/train_data/rescaled_images/4';
    out_dir_meta='/disk1/maheen_data/mayExperiments/model_50000_flo';
    util.mkdir(out_dir_meta);

    img_paths=util.getFilesInFolder(dir_training,'.jpg');
    batch_size=300;
    grid_sizes=[1,2,4,5];
    model_file='/disk2/mayExperiments/finetuning_youtube_hmdb_llr/OptFlow_youtube_hmdb_iter_50000.caffemodel';
    clusters_file='/home/maheenrashid/Downloads/debugging_jacob/optical_flow_prediction_test/examples/opticalflow/clusters.mat';
    gpu=1;

    idx_range=util.getIdxRange(len(img_paths),batch_size);
    
    print len(idx_range);
    out_file_lists=[];
    for idx_idx,idx_start in enumerate(idx_range[:-1]):
        idx_end=idx_range[idx_idx+1];
        img_paths_rel=img_paths[idx_start:idx_end];
        out_file_list=os.path.join(out_dir_meta,'batch_'+str(idx_idx)+'.txt');
        # util.writeFile(out_file_list,img_paths_rel);
        out_file_lists.append(out_file_list);

    #     print idx_start,idx_end;
    # print len(img_paths),len(idx_range);
    out_file_lists=out_file_lists[100:];
    for list_no,out_file_list in enumerate(out_file_lists):
        dir_curr=out_file_list[:out_file_list.rindex('.')];
        util.mkdir(dir_curr);
        img_paths=util.readLinesFromFile(out_file_list);
        print 'LIST NO',list_no,out_file_list;
        script_saveFloPyramidsAndAverageEfficient(dir_curr,img_paths,grid_sizes,model_file,gpu,clusters_file,append_folder=True,overwrite=False);



   

    
if __name__=='__main__':
    main();