import numpy as np;
import cPickle as pickle;
import scipy.misc;
import scipy.io;
import scipy.spatial.distance
from collections import namedtuple
import cv;
import cv2;
import os;
import sklearn.cluster;
import time;
import random;
import matplotlib;
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import util;
import h5py
import visualize;
import subprocess;
import multiprocessing;

def createParams(type_Experiment):
    if type_Experiment == 'saveTifFiles':
        list_params=['cluster_file',
                    'video_dir_meta',
                    'move_dir_meta',
                    'flo_dir',
                    'tif_dir',
                    'check_dir',
                    'im_shape',
                    'im_file']
        params = namedtuple('Params_saveTifFiles',list_params);
    else:
        params=None;

    return params


def readFlowFile(file_name,flip=False):
    data2D=None
    with open(file_name,'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print 'Magic number incorrect. Invalid .flo file'
        else:
            w = np.fromfile(f, np.int32, count=1)
            h = np.fromfile(f, np.int32, count=1)
            if w.size==0 or h.size==0:
                # print type(w),type(h),w,h
                data2D=None;
            else:               
                data = np.fromfile(f, np.float32, count=2*w*h)
                data2D = np.reshape(data, (h, w, 2))
    return data2D

def makeClustersWithFloFiles(flo_files,clusters_file,num_clusters):
    flo_all=np.zeros((0,2));

    for flo_file_idx,flo_file in enumerate(flo_files):
        flo_file=flo_files[flo_file_idx];
        print flo_file_idx,flo_file,
        flo_org=readFlowFile(flo_file);
        
        if flo_org is None:
            print 'PROBLEM'
            continue;

        print flo_all.shape
        flo=np.reshape(flo_org,(flo_org.size/2,2),order='C');        
        flo_all=np.append(flo_all,flo,0);

    t=time.time();
    kmeans=sklearn.cluster.MiniBatchKMeans(num_clusters);
    kmeans.fit(flo_all);
    print time.time()-t;
    print kmeans.cluster_centers_ 
    np.save(clusters_file,kmeans.cluster_centers_);
    scipy.io.savemat(clusters_file,{'C':kmeans.cluster_centers_})


def getAllFloFiles(out_dir):
    flo_files=[];
    for video_dir in os.listdir(out_dir):
        dir_curr=os.path.join(out_dir,video_dir,'flo');

        if not os.path.isdir(dir_curr):
            continue; 
        
        for file_curr in os.listdir(dir_curr):
            if file_curr.endswith('.flo'):
                flo_files.append(os.path.join(dir_curr,file_curr));

    return flo_files;

def makeTifFilesAndMoveFloFiles((C,im_shape,im_file,flo_dir,tif_dir,move_dir,check_dir)):
    print flo_dir,
    try:
        if not os.path.exists(check_dir):
            im_list=util.readLinesFromFile(os.path.join(flo_dir,im_file));
            flo_files=getSortedFloFiles(flo_dir);
            util.mkdir(tif_dir);
            for idx_flo_file,flo_file in enumerate(flo_files):
                flo_name=flo_file[flo_file.rindex('/')+1:];
                im_path=im_list[idx_flo_file];
                im_name=im_path[im_path.rindex('/')+1:];
                tif_name=im_name[:im_name.rindex('.')]+'.tif';
                tif_file=os.path.join(tif_dir,tif_name);
                # print flo_file,tif_file
                makeTifFile(C,im_shape,flo_file,tif_file);
            command = 'mv '+flo_dir+' '+move_dir;
            # print command;
            subprocess.call(command, shell=True)
            util.mkdir(check_dir);
            print ' done';
        else:
            print ' skipping';
    except:
        print ' error';


def script_saveTifFiles(params):
    cluster_file = params.cluster_file;
    video_dir_meta = params.video_dir_meta;
    move_dir_meta = params.move_dir_meta;
    flo_dir = params.flo_dir;
    tif_dir = params.tif_dir;
    check_dir = params.check_dir;
    im_shape = params.im_shape;
    im_file = params.im_file;

    C=np.load(cluster_file);
    args=[];
    for video_dir in os.listdir(video_dir_meta):
        dir_curr=os.path.join(video_dir_meta,video_dir);
        
        if os.path.isdir(dir_curr):
            check_dir_curr=os.path.join(dir_curr,check_dir);
            flo_dir_curr=os.path.join(dir_curr,flo_dir);
            tif_dir_curr=os.path.join(dir_curr,tif_dir);
            move_dir=os.path.join(move_dir_meta,video_dir);
            

            # print 'flo_dir',flo_dir_curr;
            # print 'tif_dir',tif_dir_curr;
            # print 'move_dir',move_dir;
            # print 'check_dir_curr',check_dir_curr;
            # t=time.time();
            # makeTifFilesAndMoveFloFiles((C,im_shape,im_file,flo_dir_curr,tif_dir_curr,move_dir,check_dir_curr))
            # print time.time()-t;
            # raw_input();
            # break;
            args.append((C,im_shape,im_file,flo_dir_curr,tif_dir_curr,move_dir,check_dir_curr));

    print len(args);
    p=multiprocessing.Pool(multiprocessing.cpu_count());
    p.map(makeTifFilesAndMoveFloFiles,args);

def makeTifFile(C,im_shape,flo_file,tif_file):
    flo=readFlowFile(flo_file);
    R = flo[:,:,0]
    L = flo[:,:,1]
    R = cv2.resize(R, (20,20));
    L = cv2.resize(L, (20,20));
    R = cv2.resize(R, (im_shape[1],im_shape[0]));
    L = cv2.resize(L, (im_shape[1],im_shape[0]));
    M=-1.0*R;
    vals=np.array([np.ravel(R),np.ravel(L)]).T    
    dists=scipy.spatial.distance.cdist(vals,C);
    min_dists=np.argmin(dists,axis=1);
    min_dists=np.reshape(min_dists,im_shape);
    min_dists=min_dists+1;
    vals=np.array([np.ravel(M),np.ravel(L)]).T
    dists_1=scipy.spatial.distance.cdist(vals,C);
    min_dists_1=np.argmin(dists_1,axis=1);
    min_dists_1=np.reshape(min_dists_1,im_shape);
    min_dists_1=min_dists_1+1;
    tif=np.dstack((min_dists,min_dists_1,np.zeros(min_dists.shape)));
    scipy.misc.toimage(tif, cmin=0, cmax=255).save(tif_file)

def getBatchSizeFromDeploy(proto_file):
    with open(proto_file,'rb') as f:
        data=f.read();
    idx=data.index('batch_size')+len('batch_size');
    data=data[idx+2:];
    data=data[:data.index('\n')];
    data=int(data);
    return data; 

def getSortedFloFiles(flo_dir):
    # ,deploy_file):
    flos_all=[os.path.join(flo_dir,file_curr) for file_curr in os.listdir(flo_dir) if file_curr.endswith('.flo')];
    flo_nums=[int(file_curr[file_curr.rindex('(')+1:file_curr.rindex(')')]) for file_curr in flos_all];
    batch_nums=[int(file_curr[file_curr.rindex('-')+1:file_curr.rindex('(')]) for file_curr in flos_all];
    num_zeros=len(str(len(flo_nums)));

    flo_nums=[((batch_nums[idx]+1)*10**num_zeros)+flo_nums[idx] for idx in range(len(flo_nums))];
    # print flo_nums;
    # print set(batch_nums)
    # # print file_curr[file_curr.rindex('-')+1:file_curr.rindex('(')]
    # if len(set(batch_nums))>1:
    #     raw_input();
    flos_all_sorted=[];
    flo_nums_sorted=flo_nums[:];
    flo_nums_sorted.sort();
    # print flo_nums_sorted
    for idx in range(len(flo_nums)):
        flo_curr=flo_nums_sorted[idx];
        idx_curr=flo_nums.index(flo_curr);
        flos_all_sorted.append(flos_all[idx_curr]);
    return flos_all_sorted;
    
def script_visualizeTifAsIm(tif_dir,im_dir,inc,out_tif,out_file_html,rel_path):

    tif_files=[file_curr for file_curr in os.listdir(tif_dir) if file_curr.endswith('.tif')];
    num_files=[int(file_curr[file_curr.rindex('_')+1:file_curr.rindex('.')]) for file_curr in tif_files];
    tif_files_sorted=[];
    num_files_sorted=num_files[:];
    num_files_sorted.sort();
    for idx in range(len(num_files)):
        num_curr=num_files_sorted[idx];
        file_curr=tif_files[num_files.index(num_curr)];
        tif_files_sorted.append(file_curr);

    rows_all=[];
    captions_all=[];
    for tif_file in tif_files_sorted:
        row_curr=[];
        tif_file_full=os.path.join(tif_dir,tif_file);
        file_name_only=tif_file[:tif_file.rindex('.')];
        file_pre=file_name_only[:file_name_only.rindex('_')+1];
        num_file=int(file_name_only[file_name_only.rindex('_')+1:]);
        num_match=num_file+inc;
        im_1=os.path.join(im_dir,file_name_only+'.jpg');
        im_2=os.path.join(im_dir,file_pre+str(num_match)+'.jpg');
        row_curr.append(im_1);
        row_curr.append(im_2);
        tif=scipy.misc.imread(tif_file_full);
        out_x=os.path.join(out_tif,file_name_only+'_x.png');
        out_y=os.path.join(out_tif,file_name_only+'_y.png');
        visualize.visualizeFlo(tif,out_x,out_y);
        row_curr.append(out_x);
        row_curr.append(out_y);
        row_curr=[path.replace(rel_path[0],rel_path[1]) for path in row_curr];
        rows_all.append(row_curr);
        captions_all.append([str(num_file),str(num_match),'x','y']);
    
    visualize.writeHTML(out_file_html,rows_all,captions_all);

def sortTifNames(names):
    nums=[int(name[name.rindex('_')+1:name.rindex('.')]) for name in names];
    nums_sorted=nums[:];
    nums_sorted.sort();
    names_sorted=[];
    for num_no_idx,num_no in enumerate(nums_sorted):
        idx_org=nums.index(num_no);
        names_sorted.append(names[idx_org]);
    return names_sorted;

def writeTrainTxt(out_file_train,video_dirs,im_dir,tif_dir,subsample=5):
    print len(video_dirs);

    # video_dirs=video_dirs[:10];
    pairs=[];
    for idx_vid_dir,vid_dir in enumerate(video_dirs):
        print idx_vid_dir,vid_dir
        tif_dir_curr=os.path.join(vid_dir,tif_dir);
        im_dir_curr=os.path.join(vid_dir,im_dir);
        tif_names=[file_curr for file_curr in os.listdir(tif_dir_curr) if file_curr.endswith('.tif')];
        tif_names=sortTifNames(tif_names);

        for tif_name in tif_names[::subsample]:
            # print tif_name
            jpg_file=os.path.join(im_dir_curr,tif_name.replace('.tif','.jpg'));

            # print jpg_file,os.path.exists(jpg_file)
            if os.path.exists(jpg_file):
                # print jpg_file
                tif_file=os.path.join(tif_dir_curr,tif_name);
                pairs.append(jpg_file+' '+tif_file);

        # raw_input();

    print len(pairs);
    # print pairs[:10];
    # pair_one=[p[:p.index(' ')] for p in pairs];
    # vid_dirs=[p[:p[:p.rindex('/')].rindex('/')] for p in pair_one];
    # print len(set(vid_dirs));
    random.shuffle(pairs);
    util.writeFile(out_file_train,pairs);


def getTrainingCommand(path_train,path_solver,path_weights=None,path_snapshot=None,path_log=None):
    command=['GLOG_logtostderr=1'];
    command.append(path_train);
    command.append('train');
    command.append('-solver='+path_solver);
    if path_weights is not None:
        command.append('-weights');
        command.append(path_weights);
    if path_snapshot is not None:
        command.append('--snapshot');
        command.append(path_snapshot);
    if path_log is not None:
        command.append('2>&1 | tee');
        command.append(path_log);

    command=' '.join(command);
    return command;


def writeTrainTxtDeepFlow(dir_videos,out_file,subsample=5):
    pairs=[];
    for idx_dir_video,dir_video in enumerate(dir_videos):
        if idx_dir_video%100==0:
            print idx_dir_video
        files_jpg=[os.path.join(dir_video,file_curr) for file_curr in os.listdir(dir_video) if file_curr.endswith('.tif')];
        files_jpg.sort();
        files_jpg=files_jpg[::subsample];
        files_tif=[file_curr.replace('.tif','.jpg') for file_curr in files_jpg];
        # for file_curr in files_jpg:
        #     print file_curr
        # raw_input();
        pairs.extend(zip(files_tif,files_jpg));

    print len(pairs);
    # print pairs[:10];
    pairs_txt=[pair[0]+' '+pair[1] for pair in pairs];
    random.shuffle(pairs_txt)
    util.writeFile(out_file,pairs_txt);

def script_getClassFrameMeanCountYoutube(dir_meta):
    dirs_video=[os.path.join(dir_meta,dir_curr) for dir_curr in os.listdir(dir_meta) if os.path.isdir(os.path.join(dir_meta,dir_curr))];
    print len(dirs_video);
    lens=[];
    types=[];
    for idx_dir_curr,dir_curr in enumerate(dirs_video):
        if idx_dir_curr%100==0:
            print idx_dir_curr;

        dir_curr_im=os.path.join(dir_curr,'images_transfer');
        command='ls '+dir_curr_im+'/*.jpg | wc -l';
        # print command
        # commands.append(command);
        response=subprocess.check_output(command,shell=True);
        lens.append(int(response));

        type_curr=dir_curr[dir_curr.rindex('/')+1:];
        type_curr=type_curr[:type_curr.index('_')];
        # print type_curr
        types.append(type_curr);
        # print command;
        # print int(response);
        # raw_input();
    
    lens_np=np.array(lens);
    types_np=np.array(types);

    counts=[];
    for type_curr in set(types):
        print type_curr,
        idx=np.where(types_np==type_curr);
        count_curr=np.sum(lens_np[idx]);
        print count_curr;
        counts.append(count_curr)

    print counts
    print set(types);

    print np.mean(counts);



def main():

    dir_meta='/disk2/marchExperiments/hmdb_try_2/hmdb';
    dir_images='images'
    
    out_txt='/disk2/marchExperiments/finetuning_hmdb/train.txt';
    out_txt_youtube='/disk2/marchExperiments/finetuning_youtube_hmdb/train.txt';
    out_txt_final='/disk2/marchExperiments/finetuning_youtube_hmdb/train_both.txt';

    hmdb_files=util.readLinesFromFile(out_txt);
    youtube_files=util.readLinesFromFile(out_txt_youtube);
    print len(hmdb_files),len(youtube_files);
    len_to_pick=13765
    random.shuffle(hmdb_files);
    hmdb_to_add=hmdb_files[:len_to_pick];
    total_lines=youtube_files+hmdb_to_add;
    print len(total_lines);
    random.shuffle(total_lines);
    print total_lines[13790]
    print total_lines[4]
    util.writeFile(out_txt_final,total_lines);


    return
    dir_meta='/disk2/marchExperiments/youtube'
    dir_images='images_transfer'
    out_txt='/disk2/marchExperiments/finetuning_youtube_hmdb/train.txt';

    videos=[];
    for dir_curr in os.listdir(dir_meta):
        if os.path.isdir(os.path.join(dir_meta,dir_curr)):
            if dir_curr=='Lock':
                print 'FOUND THE LOCK';
            else:
                videos.append(os.path.join(dir_meta,dir_curr,dir_images));

    # videos=[os.path.join(dir_meta,dir_curr) for dir_curr in os.listdir(dir_meta) if os.path.join(dir_meta,dir_curr)
    
    writeTrainTxtDeepFlow(videos,out_txt,subsample=1)
    
    print out_txt

    # print len(videos);
    # for video in videos[:10]:
    #     print video;

    return
    dir_network='/disk2/marchExperiments/network_100_5/results_im'
    out_file_html=os.path.join(dir_network,'visualize.html');
    im_files=[os.path.join(dir_network,file_curr) for file_curr in os.listdir(dir_network)];
    rel_path=['/disk2','../../../..'];
    im_files=[[file_curr.replace(rel_path[0],rel_path[1])] for file_curr in im_files];
    captions=[[' ']]*len(im_files);
    visualize.writeHTML(out_file_html,im_files,captions);

    return

    train_file=os.path.join(dir_network,'train.txt');
    test_file=os.path.join(dir_network,'test.txt');
    lines=util.readLinesFromFile(train_file);
    test=[];
    print train_file
    print len(lines);
    for line in lines:
        test_file_curr=line[:line.index(' ')];
        test_file_curr=test_file_curr+' 1';
        test.append(test_file_curr);
    print len(test)

    util.writeFile(test_file,test);



    return
    dir_network='/disk2/marchExperiments/network_100_5'
    path_train='/home/maheenrashid/Downloads/debugging_jacob/optical_flow_prediction/build/tools/caffe';
    path_solver=os.path.join(dir_network,'train.prototxt');

    # path_weights='/home/maheenrashid/Downloads/debugging_jacob/optical_flow_prediction/models/bvlc_alexnet/bvlc_alexnet.caffemodel';
    path_snapshot='/disk2/marchExperiments/network_100_5/OptFlow_snapshot_iter_900.solverstate';
    path_log=os.path.join(dir_network,'log_stepsizechanged.log');

    command=getTrainingCommand(path_train,path_solver,path_snapshot=path_snapshot,path_log=path_log)
    print command
    return

    dir_meta='/disk2/marchExperiments/ucf-101-new';
    out_dir_network='/disk2/marchExperiments/network_100_5';
    util.mkdir(out_dir_network);

    out_file_train=os.path.join(out_dir_network,'train.txt');
    video_dirs_file=os.path.join(out_dir_network,'video_dirs.p');
    video_dirs=pickle.load(open(video_dirs_file,'rb'));
    for vid in video_dirs:
        print vid;
    # return
    # video_dirs=['/disk2/marchExperiments/ucf-101-new/v_BoxingPunchingBag_g01_c01']
    im_dir='im';
    tif_dir='tif';
    # video_dirs=[os.path.join(dir_meta,vid_dir) for vid_dir in os.listdir(dir_meta) if os.path.isdir(os.path.join(dir_meta,vid_dir))];
    # random.shuffle(video_dirs);
    # video_dirs=video_dirs[:100];
    writeTrainTxt(out_file_train,video_dirs,im_dir,tif_dir)
    # pickle.dump(video_dirs,open(os.path.join(out_dir_network,'video_dirs.p'),'wb'));

    return

    
    # out_dir='/disk2/marchExperiments/ucf-101-new'
    # clusters_file = os.path.join(out_dir,'clusters_100000_full.npy');
    # out_file=os.path.join('/disk2/marchExperiments/ucf-101-new','clusters_100000_full.mat');
    # data=np.load(clusters_file);
    # scipy.io.savemat(out_file,{'C':data});
    # print out_file;

    

    # return    
        




    # tif_dir='/disk2/marchExperiments/ucf-101-new/v_BoxingPunchingBag_g01_c01/tif'
    # im_dir='/disk2/marchExperiments/ucf-101-new/v_BoxingPunchingBag_g01_c01/im'

    # out_tif='/disk2/marchExperiments/v_BoxingPunchingBag_g01_c01_visualize';
    # util.mkdir(out_tif);
    # rel_path=['/disk2','../../..'];
    # out_file_html=os.path.join(out_tif,'visualize.html');
    # inc=10;
    # script_visualizeTifAsIm(tif_dir,im_dir,inc,out_tif,out_file_html,rel_path)


    return
    
    out_dir='/disk2/marchExperiments/ucf-101-new'
    im_shape=[240,320];
    params_dict={};
    params_dict['cluster_file']=os.path.join(out_dir,'clusters_100000_full.npy');
    params_dict['video_dir_meta']=out_dir
    params_dict['move_dir_meta']='/media/maheenrashid/e5507fe3-2bff-4cbe-bc63-400de6deba92/maheen_data/flow_data/UCF-101/'
    params_dict['flo_dir']='flo'
    params_dict['tif_dir']='tif'
    params_dict['check_dir']='done_tif';
    params_dict['im_shape']= [240,320];
    # params_dict['deploy_file']='deploy.prototxt';
    params_dict['im_file']='im_1.txt'

    params=createParams('saveTifFiles');
    params=params(**params_dict);
    script_saveTifFiles(params);

    return

    # out_dir='/disk2/marchExperiments/ucf-101-new'
    # im_shape=[240,320];
    # clusters_file=os.path.join(out_dir,'clusters_1000_full.npy')
    # C=np.load(clusters_file);
    # clusters_file=os.path.join(out_dir,'clusters_1000_full.mat')
    # scipy.io.savemat(clusters_file,{'C':C})

    # return
    # video_name='v_TennisSwing_g01_c03';
    # video_dir=os.path.join(out_dir,video_name);
    # flo_dir=os.path.join(video_dir,'flo');
    # tif_dir=os.path.join(video_dir,'tif');
    # util.mkdir(tif_dir);
    # im_list=util.readLinesFromFile(os.path.join(flo_dir,'im_1.txt'));
    # flo_files=getSortedFloFiles(flo_dir);
    # C=np.load(clusters_file);
    # print C;
    # train_text=os.path.join(out_dir,video_name+'_train.txt');
    # train_data=[];
    # for idx_flo_file,flo_file in enumerate(flo_files):
    #     flo_name=flo_file[flo_file.rindex('/')+1:];
    #     im_path=im_list[idx_flo_file];
    #     im_name=im_path[im_path.rindex('/')+1:];
    #     tif_name=im_name[:im_name.rindex('.')]+'.tif';
    #     tif_file=os.path.join(tif_dir,tif_name);

    #     print flo_name,im_name,tif_file
    #     makeTifFile(C,im_shape,flo_file,tif_file);

    #     # tif=scipy.misc.imread(tif_file);
    #     # print tif.shape
    #     # for dim in range(tif.shape[2]):
    #     #     print np.min(tif[:,:,dim]),np.max(tif[:,:,dim])
    #     # break;
        
    #     # train_data.append(im_path+' '+tif_file);

    # # util.writeFile(train_text,train_data);







    # return
    # out_dir='/disk2/marchExperiments/ucf-101-new'

    # flo_files_file=os.path.join(out_dir,'flo_files.txt');
    # flo_files=util.readLinesFromFile(flo_files_file);
    # n=10;
    # im_file_1='im_1.txt';
    # im_file_2='im_2.txt';
    # out_file_html=os.path.join(out_dir,'check_tifs_full.html');
    # rel_path=['/disk2','../../..'];
    # clusters_file=os.path.join(out_dir,'clusters_1000_full.npy')
    # C=np.load(clusters_file);
    # #read some random flo files
    # random.shuffle(flo_files);
    # flo_files=flo_files[:n];

    # im_files_html=[];
    # captions_html=[];
    # for flo_file in flo_files:
    #     print flo_file
    #     flo_dir=flo_file[:flo_file.rindex('/')];
    #     flo_num=int(flo_file[flo_file.rindex('(')+1:flo_file.rindex(')')]);

    #     #figure out corresponding image pairs
    #     im_list_1=util.readLinesFromFile(os.path.join(flo_dir,im_file_1));
    #     im_list_2=util.readLinesFromFile(os.path.join(flo_dir,im_file_2));
    #     im_1=im_list_1[flo_num];
    #     im_2=im_list_2[flo_num];
    #     # im_pairs.append((im_1,im_2));

    #     #make tif labels for the flo files
    
    #     video_dir=flo_file[:flo_dir.rindex('/')];
    #     out_dir_tif=os.path.join(video_dir,'tif');
    #     # print flo_file
    #     # print video_dir
    #     # print flo_dir
    #     # print out_dir_tif
    #     # print im_1
    #     # print im_2
    #     # raw_input();
    #     util.mkdir(out_dir_tif);

    #     out_tif_name = os.path.join(out_dir_tif,str(flo_num)+'.tif');
    #     out_tif_name_x=os.path.join(out_dir_tif,str(flo_num)+'_x.png');
    #     out_tif_name_y=os.path.join(out_dir_tif,str(flo_num)+'_y.png');

    #     out_flo_name_x=os.path.join(out_dir_tif,'flo_'+str(flo_num)+'_x.png');
    #     out_flo_name_y=os.path.join(out_dir_tif,'flo_'+str(flo_num)+'_y.png');
    #     flo=util.readFlowFile(flo_file);
    #     visualize.visualizeFlo(flo,out_flo_name_x,out_flo_name_y)
    #     im=scipy.misc.imread(im_1);
    #     im_shape=im.shape[:2];

    #     makeTifFile(C,im_shape,flo_file,out_tif_name)
    #     tif=scipy.misc.imread(out_tif_name);
    #     print 'tif',tif.shape
    #     visualize.visualizeFlo(tif,out_tif_name_x,out_tif_name_y);
    #     # makeTifFile(C,im_shape,flo_file,out_tif_name_x,out_tif_name_y)
        
    #     im_files_html_curr=[];
    #     im_files_html_curr.append(im_1.replace(rel_path[0],rel_path[1]));
    #     im_files_html_curr.append(im_2.replace(rel_path[0],rel_path[1]));
    #     im_files_html_curr.append(out_flo_name_x.replace(rel_path[0],rel_path[1]));
    #     im_files_html_curr.append(out_flo_name_y.replace(rel_path[0],rel_path[1]));
    #     im_files_html_curr.append(out_tif_name_x.replace(rel_path[0],rel_path[1]));
    #     im_files_html_curr.append(out_tif_name_y.replace(rel_path[0],rel_path[1]));
        
    #     captions_html_curr=['im1','im2','flo_x','flo_y','tif_r','tif_l'];
    #     im_files_html.append(im_files_html_curr);
    #     captions_html.append(captions_html_curr);

    # #save them

    # #visualize them in an html
    # visualize.writeHTML(out_file_html,im_files_html,captions_html);


    # return
    out_dir='/disk2/marchExperiments/ucf-101-new'
    flo_files_file=os.path.join(out_dir,'flo_files.txt');

    # flo_files=getAllFloFiles(out_dir);
    # print len(flo_files),flo_files[:3];
    # util.writeFile(flo_files_file,flo_files);
    # return
    flo_files=util.readLinesFromFile(flo_files_file);
    print len(flo_files),flo_files[:3];
    downsize=[120,160];
    n=50000;
    num_clusters=40;
    clusters_file=os.path.join(out_dir,'clusters_'+str(n)+'_full.npy');

    #pick n random flo files
    random.shuffle(flo_files);
    flo_files_cluster= flo_files[:n];

    #read and downsize the flo files
    flo_all=np.zeros((n*downsize[0]*downsize[1],2));
    for flo_file_idx,flo_file in enumerate(flo_files_cluster):
        flo_curr=util.readFlowFile(flo_file);
        print flo_curr.shape
        # raw_input();
        x=flo_curr[:,:,0];
        y=flo_curr[:,:,1];
        # x = cv2.resize(flo_curr[:,:,0],(downsize[1],downsize[0]));
        # y = cv2.resize(flo_curr[:,:,1],(downsize[1],downsize[0]));
        quant=x.size
        start_idx=flo_file_idx*quant;
        end_idx=(flo_file_idx*quant)+quant;
        print start_idx,end_idx
        flo_all[start_idx:end_idx,0]=x.ravel();
        flo_all[start_idx:end_idx,1]=y.ravel();
    
    # make clusters
    print 'making clusters',flo_all.shape;
    t=time.time();
    kmeans=sklearn.cluster.MiniBatchKMeans(num_clusters);
    kmeans.fit(flo_all);
    print time.time()-t;
    print kmeans.cluster_centers_ 
    np.save(clusters_file,kmeans.cluster_centers_);
    # scipy.io.savemat(clusters_file,{'C':kmeans.cluster_centers_})



    return
    in_dir='/home/maheenrashid/Downloads/debugging_jacob/opticalflow/videos/temp_vid';
    file_pre='temp_vid.avi_000109';
    clusters_file='/home/maheenrashid/Downloads/debugging_jacob/opticalflow/clusters.mat';
    tif_file=os.path.join(in_dir,file_pre+'.tif');
    flo_file_old=os.path.join(in_dir,file_pre+'.flo');

    in_dir_flo='/disk2/marchExperiments/ucf-101/v_RopeClimbing_g04_c03/flo/'
    in_dir_flo='/media/maheenrashid/e5507fe3-2bff-4cbe-bc63-400de6deba92/maheen_data/flow_data/UCF-101/RopeClimbing/v_RopeClimbing_g04_c03'
    flo_file=os.path.join(in_dir_flo,'flownets-pred-0000000(109).flo');

    flo_file='/home/maheenrashid/Downloads/flownet/flownet-release/models/flownet/flownets-pred-0000000.flo';
    flo_old=readFlowFile(flo_file);

    print flo_file;
    plt.figure();plt.imshow(flo_old[:,:,0]);plt.savefig('/disk2/temp/flo_flownet_x_20.png');
    plt.figure();plt.imshow(flo_old[:,:,1]);plt.savefig('/disk2/temp/flo_flownet_y_20.png');
    return


    # flo_old=readFlowFile(flo_file_old);
    
    # flo_new=readFlowFile(flo_file_new,flip=True);
    # flo_new=np.zeros(flo_new.shape);

    # for i in range(109,114):
    #     flo_file_curr=flo_file_new.replace(str(108),str(i));
    #     flo_curr=readFlowFile(flo_file_curr);
    #     flo_new=flo_new+flo_curr;
    # flo_new=flo_new/5.0;

    # print flo_old.shape,np.min(flo_old[:,:,0]),np.max(flo_old[:,:,1])
    # print flo_new.shape,np.min(flo_new[:,:,0]),np.max(flo_new[:,:,1])
    # plt.figure();plt.imshow(flo_old[:,:,0]);plt.savefig('/disk2/temp/flo_old_x.png');
    # plt.figure();plt.imshow(flo_old[:,:,1]);plt.savefig('/disk2/temp/flo_old_y.png');

    # plt.figure();plt.imshow(flo_new[:,:,0]);plt.savefig('/disk2/temp/flo_new_x.png');
    # plt.figure();plt.imshow(flo_new[:,:,1]);plt.savefig('/disk2/temp/flo_new_y.png');


    # return
    num_file=109;

    clusters=h5py.File(clusters_file)
    C=np.array(clusters['C']).T
    # print C
    # return
    tif=scipy.misc.imread(tif_file);
    flo=readFlowFile(flo_file);
    print flo.shape,tif.shape

    all_flos=np.zeros(flo.shape);
    for new_flo_num in range(110,115):
        new_str=str(new_flo_num);
        new_flo_file=flo_file.replace(str(num_file),new_str);
        new_flo=readFlowFile(new_flo_file);
        all_flos=all_flos+new_flo;
    
    print all_flos.shape;

    for dim in range(all_flos.shape[2]):
        print dim,np.min(all_flos[:,:,dim]),np.max(all_flos[:,:,dim]);
        # print new_flo_file

    R =all_flos[:,:,0]
    # .T;
    L=all_flos[:,:,1]
    # .T ;
    print R.shape,L.shape,tif.shape
    print R.shape,L.shape,tif.shape,np.min(R),np.max(R);
    R = cv2.resize(R, (20,20));
    L = cv2.resize(L, (20,20));

    # R=scipy.ndimage.interpolation.zoom(R,(2,2));

    # R=scipy.misc.imresize(R,(20,20),'bicubic');

    # L=scipy.misc.imresize(L,(20,20),'bicubic');
    print R.shape,L.shape,tif.shape,np.min(R),np.max(R);
    R = cv2.resize(R, (tif.shape[1],tif.shape[0]));
    L = cv2.resize(L, (tif.shape[1],tif.shape[0]));

    # R=scipy.misc.imresize(R,(tif.shape[0],tif.shape[1]),'bicubic');
    # L=scipy.misc.imresize(L,(tif.shape[0],tif.shape[1]),'bicubic');
    print R.shape,L.shape,tif.shape

    org_shape=tif.shape[:2]
    # org_shape=(20,20);  

    M=-1.0*R;
    
    vals=np.array([np.ravel(R),np.ravel(L)]).T
    
    print vals.shape
    
    dists=scipy.spatial.distance.cdist(vals,C);
    
    print dists.shape

    min_dists=np.argmin(dists,axis=1);
    min_dists=np.reshape(min_dists,org_shape);
    min_dists=min_dists+1;
    print min_dists.shape,np.min(min_dists),np.max(min_dists);
    print tif[:,:,0].shape,np.min(tif[:,:,0]),np.max(tif[:,:,0]);

    vals=np.array([np.ravel(M),np.ravel(L)]).T
    dists_1=scipy.spatial.distance.cdist(vals,C);
    min_dists_1=np.argmin(dists_1,axis=1);
    min_dists_1=np.reshape(min_dists_1,org_shape);
    min_dists_1=min_dists_1+1;
    print min_dists_1.shape,np.min(min_dists_1),np.max(min_dists_1);
    print tif[:,:,1].shape,np.min(tif[:,:,1]),np.max(tif[:,:,1]);
    print tif[:,:,2].shape,np.min(tif[:,:,2]),np.max(tif[:,:,2]);

    plt.figure();plt.imshow(min_dists);
    plt.savefig('/disk2/temp/min_dists.png');
    plt.figure();plt.imshow(min_dists_1);
    plt.savefig('/disk2/temp/min_dists_1.png');

    plt.figure();plt.imshow(tif[:,:,0]);
    plt.savefig('/disk2/temp/tif.png');

    plt.figure();plt.imshow(tif[:,:,1]);
    plt.savefig('/disk2/temp/tif_1.png');
    
    
     # L = optFlow(:,:,2);
     # R = imresize(R, [paramBall.labelDim, paramBall.labelDim]);
     # L = imresize(L, [paramBall.labelDim, paramBall.labelDim]);
            
     # R = imresize(R, [size(I,1), size(I,2)]);
     # L = imresize(L, [size(I,1), size(I,2)]);
     # M = -1.0*imresize(R, [size(I,1), size(I,2)]);
            
     # theDists = pdist2([R(:) L(:)], C);
     # [~,theInds] = min(theDists');
            
     # theInds = reshape(theInds, [size(I,1), size(I,2)]);
            
     # theDists = pdist2([M(:) L(:)], C);
     # [~,theIndsM] = min(theDists');
            
     # theIndsM = reshape(theIndsM, [size(I,1), size(I,2)]);
            
     # tmp = uint8(theInds);
     # tmp(:,:,2) = uint8(theIndsM);
     # tmp(:,:,3) = uint8(0*theIndsM);




    # scipy.io.loadmat(clusters_file)
    # squeeze_me=True, struct_as_record=False);
    # for k in clusters['paramBall'].keys():
    #     print k,np.array(clusters['paramBall'][k]);
    # .keys();


    return
    res_dir='/disk2/marchExperiments/train_oneVideo_5/results_viz';
    res_dir='/disk2/marchExperiments/train_oneVideo_5/results_pascal_viz';
    rel_path=['/disk2','../../..'];

    res_dir='/disk2/marchExperiments/sanity_check_deepflow';
    res_dir='/disk2/marchExperiments/sanity_check_deepflow';
    rel_path=['/disk2','../..'];

    out_file_html=os.path.join(res_dir,'results.html')

    im_files=[os.path.join(res_dir,file_curr) for file_curr in os.listdir(res_dir) if file_curr.endswith('.jpg')];
    im_files=[[file_curr.replace(rel_path[0],rel_path[1])] for file_curr in im_files];
    print im_files;
    # captions=im_files[:];
    
    util.writeHTML(out_file_html,im_files,im_files,height=200,width=200)

    return
    num_clusters=40;
    flo_dir='/disk2/marchExperiments/ucf-101/v_RopeClimbing_g04_c03/flo'
    flo_files=[os.path.join(flo_dir,file_curr) for file_curr in os.listdir(flo_dir) if file_curr.endswith('.flo')];
    clusters_file='/home/maheenrashid/Downloads/debugging_jacob/opticalflow/clusters_oneVideo_5.mat'
    
    flo_nums=[int(file_curr[file_curr.rindex('(')+1:file_curr.rindex(')')]) for file_curr in flo_files];
    flo_nums=np.array(flo_nums);
    sort_idx=np.argsort(flo_nums);
    flo_files=np.array(flo_files);
    flo_files=flo_files[sort_idx];
    flo_files=list(flo_files);
    print flo_files

    arr=range(0,len(flo_files),5);
    flos_all=[];

    for idx_idx,idx in enumerate(arr[:-1]):
        start_idx=idx;
        end_idx=arr[idx_idx+1];
        print 'new',
        for idx_idx_idx,idx_curr in enumerate(range(start_idx,end_idx)):
            file_curr=flo_files[idx_curr];
            print file_curr,
            if idx_idx_idx==0:
                flo_curr=readFlowFile(file_curr);
                print flo_curr.shape,
            else:
                flo_curr=flo_curr+readFlowFile(file_curr);
                print flo_curr.shape,
            print '';
        flos_all.append(flo_curr);


    flo_all=np.zeros((0,2));

    for flo_org in flos_all:
        # flo_file=flo_files[flo_file_idx];
        # print flo_file_idx,flo_file,
        # flo_org=readFlowFile(flo_file);
        
        # if flo_org is None:
        #     print 'PROBLEM'
        #     continue;

        print flo_all.shape
#         flo_org_check_x=np.sort(np.ravel(flo_org[:,:,0]))
#         flo_org_check_y=np.sort(np.ravel(flo_org[:,:,1]))
        flo=np.reshape(flo_org,(flo_org.size/2,2),order='C');
        # flo_x=np.sort(flo[:,0]);
        # flo_y=np.sort(flo[:,1]);

        # print np.array_equal(flo_org_check_y,flo_y),np.array_equal(flo_org_check_y,flo_x),np.array_equal(flo_org_check_x,flo_x),np.array_equal(flo_org_check_x,flo_y)
        
        flo_all=np.append(flo_all,flo,0);

    t=time.time();
    kmeans=sklearn.cluster.MiniBatchKMeans(num_clusters);
    kmeans.fit(flo_all);
    print time.time()-t;
    print kmeans.cluster_centers_ 
    np.save(clusters_file,kmeans.cluster_centers_);
    scipy.io.savemat(clusters_file,{'C':kmeans.cluster_centers_})



    return
    # flo_files=flo_files[1::10];
    # flownets-pred-0000000(094).flo'
    clusters_file='/home/maheenrashid/Downloads/debugging_jacob/opticalflow/clusters_oneVidoe.mat';
    num_clusters=40
    # flo_files=[flo_file];

    makeClustersWithFloFiles(flo_files,clusters_file,num_clusters)

    return
    out_dir='/disk2/februaryExperiments/training_jacob';
    file_subset=os.path.join(out_dir,'im_flo_correspondences_hmdb.p');
    out_file_flo=os.path.join(out_dir,'flo_files_random_each_video_hmdb.p');
    clusters_file=os.path.join(out_dir,'clusters_hmdb_100.npy');

    out_dir_visualization='/disk2/marchExperiments/debug_jacob';
    out_file_viz=os.path.join(out_dir_visualization,'hmdb_100_dist_1_diff.png');

    flo_files=pickle.load(open(out_file_flo,'rb'));
    # random.shuffle(flo_files);
    flo_files=flo_files[:100];
    print flo_files[0];

    return

    flo_all=np.zeros((0,2));

    for flo_file in flo_files:
        # print flo_file
        pre_idx=flo_file.rindex('(')+1
        post_idx=flo_file.rindex(')')

        flo_file_pre=flo_file[:pre_idx];
        flo_file_post=flo_file[post_idx:];
        mid_val_str=flo_file[pre_idx:post_idx];
        mid_val=int(mid_val_str);
        mid_val_new=random.randint(0,mid_val);
        mid_val_new_str='0'*(3-len(str(mid_val_new)))+str(mid_val_new);
        flo_file_new=flo_file_pre+mid_val_new_str+flo_file_post;
        # print flo_file_new;
        # raw_input();


        flo_org=readFlowFile(flo_file_new);
        flo=np.reshape(flo_org,(flo_org.size/2,2),order='C');
        flo_all=np.append(flo_all,flo,0);

    mean=np.mean(flo_all,axis=0);
    std=np.std(flo_all,axis=0);
    flo_x=np.sort(flo_all[:,0]);
    flo_y=np.sort(flo_all[:,1]);
    plt.figure();
    plt.suptitle(str(mean)+' '+str(std));

    plt.subplot(1,2,1);
    plt.plot(flo_x);
    plt.subplot(1,2,2);
    plt.plot(flo_y);
    
    plt.savefig(out_file_viz);


    return

    num_clusters=40;

    flo_files=pickle.load(open(out_file_flo,'rb'));
    # random.shuffle(flo_files);
    flo_all=np.zeros((0,2));

    for flo_file_idx in range(100):
        flo_file=flo_files[flo_file_idx];
        print flo_file_idx,flo_file,
        flo_org=readFlowFile(flo_file);
        
        if flo_org is None:
            print 'PROBLEM'
            continue;

        print flo_all.shape
        flo=np.reshape(flo_org,(flo_org.size/2,2),order='C');
        flo_all=np.append(flo_all,flo,0);

    t=time.time();
    kmeans=sklearn.cluster.MiniBatchKMeans(num_clusters);
    kmeans.fit(flo_all);
    print time.time()-t;
    print kmeans.cluster_centers_ 
    np.save(clusters_file,kmeans.cluster_centers_);
    # cc=np.load(clusters_file);
    # print cc;



    return
    
    im_flo_dirs=pickle.load(open(file_subset,'rb'));
    print len(im_flo_dirs);
    [im_dirs,flo_dirs]=zip(*im_flo_dirs);
    flo_files_all=[];

    for flo_dir in flo_dirs:
        flo_files=[os.path.join(flo_dir,file_curr) for file_curr in os.listdir(flo_dir) if file_curr.endswith('.flo')];
        # print (len(flo_files));
        rand_idx=random.randint(0,len(flo_files)-1)
        flo_files_all.append(flo_files[rand_idx]);
        # print len(flo_files),rand_idx,flo_files_all
        # raw_input();
        print len(flo_files_all);

    pickle.dump(flo_files_all,open(out_file_flo,'wb'))

    print 'hello world';

if __name__=='__main__':
    main();