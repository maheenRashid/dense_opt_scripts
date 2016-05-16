import numpy as np;
import os;
import util;
import multiprocessing;
import random;
import math;
import processOutput as po;
import script_testJacob as stj;
import subprocess;
import cv2;
def getEPE(flo_gt,flo_pred):
    sqr=np.power(flo_gt-flo_pred,2);
    # print sqr.shape
    summed=sqr[:,:,0]+sqr[:,:,1];
    # print summed.shape
    sqr_rt=np.power(summed,0.5);
    epe=np.mean(sqr_rt);
    return epe,sqr_rt

def getOrientationSimilarity(flo_gt,flo_pred):
    flo_gt=np.vstack((flo_gt[:,:,0].ravel(),flo_gt[:,:,1].ravel()));
    flo_pred=np.vstack((flo_pred[:,:,0].ravel(),flo_pred[:,:,1].ravel()));

    flo_gt_norms=np.power(np.sum(np.power(flo_gt,2),0),0.5);
    flo_pred_norms=np.power(np.sum(np.power(flo_pred,2),0),0.5);

    dots=[];
    for val_idx in range(flo_gt.shape[1]):
        dots.append(np.dot(flo_gt[:,val_idx],flo_pred[:,val_idx]));
    dots=np.array(dots);
    dots=np.abs(dots);
    
    
    
    printflag=False;
    if np.sum(flo_gt_norms==0)>0 or np.sum(flo_pred_norms==0)>0:
        print 'zero_norms_orientation'
        print np.sum(flo_gt_norms==0),np.sum(flo_pred_norms==0);
        ors_old=dots/(flo_gt_norms*flo_pred_norms);
        flo_gt_norms[flo_gt_norms==0]=np.finfo(float).eps;
        flo_pred_norms[flo_pred_norms==0]=np.finfo(float).eps;
        printflag=True;    

    ors=dots/(flo_gt_norms*flo_pred_norms);
    # ors=dots/flo_pred_norms;
    if printflag:
        print np.mean(ors),np.mean(ors_old);

    return np.mean(ors),ors;

def getDirectionSimilarity(flo_gt,flo_pred):
    flo_gt=np.vstack((flo_gt[:,:,0].ravel(),flo_gt[:,:,1].ravel()));
    flo_pred=np.vstack((flo_pred[:,:,0].ravel(),flo_pred[:,:,1].ravel()));

    flo_gt_norms=np.power(np.sum(np.power(flo_gt,2),0),0.5);
    flo_pred_norms=np.power(np.sum(np.power(flo_pred,2),0),0.5);

    dots=[];
    for val_idx in range(flo_gt.shape[1]):
        dots.append(np.dot(flo_gt[:,val_idx],flo_pred[:,val_idx]));
    dots=np.array(dots);
    

    printflag=False;
    if np.sum(flo_gt_norms==0)>0 or np.sum(flo_pred_norms==0)>0:
        print 'zero_norms_direction'
        print np.sum(flo_gt_norms==0),np.sum(flo_pred_norms==0);
        ors_old=dots/(flo_gt_norms*flo_pred_norms);
        flo_gt_norms[flo_gt_norms==0]=np.finfo(float).eps;
        flo_pred_norms[flo_pred_norms==0]=np.finfo(float).eps;
        printflag=True;    

    ors=dots/(flo_gt_norms*flo_pred_norms);
    # ors=dots/flo_pred_norms;
    if printflag:
        print np.mean(ors),np.mean(ors_old);

    return np.mean(ors),ors;


def getEPEWrapper((path_gt,path_pred,idx,includeBigMat,error_type)):
    print idx;
    flo_gt=np.load(path_gt);
    flo_pred=np.load(path_pred);
    if error_type=='epe':
        epe,epe_pix=getEPE(flo_gt,flo_pred);
    elif error_type=='direction':
        epe,epe_pix=getDirectionSimilarity(flo_gt,flo_pred);
    elif error_type=='orientation':
        epe,epe_pix=getOrientationSimilarity(flo_gt,flo_pred);

    if includeBigMat:
        ret_value=(epe,epe_pix)
    else:
        ret_value=epe;

    return ret_value;


def getAllErrorsWrapper((path_gt,path_pred,idx,isFlo)):
    print idx;
    if isFlo:
        flo_gt=util.readFlowFile(path_gt);
        flo_pred=util.readFlowFile(path_pred);
    else:
        flo_gt=np.load(path_gt);
        flo_pred=np.load(path_pred);
    # print flo_gt.shape,flo_pred.shape,path_gt,path_pred;

    # if flo_gt.shape!=flo_pred.shape:
    # util.writeFlowFile(flo_pred,'/disk2/temp/flo_pred_bef.flo');

    shape_old=flo_pred.shape;
    flo_pred=cv2.resize(flo_pred,(flo_gt.shape[1],flo_gt.shape[0]));
    flo_pred[:,:,0]=flo_pred[:,:,0]*float(flo_gt.shape[1])/shape_old[1];
    flo_pred[:,:,1]=flo_pred[:,:,1]*float(flo_gt.shape[0])/shape_old[0];
    # util.writeFlowFile(flo_pred,'/disk2/temp/flo_pred_aft.flo');
    # util.writeFlowFile(flo_gt,'/disk2/temp/flo_gt.flo');

    epe,_=getEPE(flo_gt,flo_pred);
    dir_sim,_=getDirectionSimilarity(flo_gt,flo_pred);
    or_sim,_=getOrientationSimilarity(flo_gt,flo_pred);
    ret_value=(epe,dir_sim,or_sim)

    return ret_value;


def getErrorMultiProc(dir_gt,dir_pred,np_files,out_file_err,isFlo=False):
    args=[];
    for idx_np_file,np_file in enumerate(np_files):
        args.append((os.path.join(dir_gt,np_file),os.path.join(dir_pred,np_file),idx_np_file,isFlo));

    print len(args);
    print args[0];

    # if error_type=='epe':
    # list_errors=[];
    # for arg_curr in args:
    #     list_errors.append(getAllErrorsWrapper(arg_curr));

    p=multiprocessing.Pool(multiprocessing.cpu_count());

    list_errors=p.map(getAllErrorsWrapper,args);
    # for arg in args:
    #     getAllErrorsWrapper(arg);


    list_errors=np.array(list_errors);

    files=np.array(np_files);
    # print list_errors.shape,files.shape
    # print list_errors[:10],files[:10]
    np.savez(out_file_err, files=files, errors=list_errors);


def splitFilesByClass(np_files):
    class_dict={}
    for file_curr in np_files:
        class_curr=file_curr[:file_curr.index('_')];
        if class_curr in class_dict:
            class_dict[class_curr].append(file_curr);
        else:
            class_dict[class_curr]=[file_curr];

    return class_dict;

def script_compYoutubePerf(dir_org,dir_new):
    # list_files=[file_curr for file_curr in os.listdir(dir_org) if file_curr.endswith('.npz')];

    # for list_file in list_files:
    # arrs=np.load(os.path.join(dir_org,list_file))
    arrs=np.load(dir_org)
    errs_old=arrs['errors'];
    list_files=arrs['files'];
    classes_new=np.array([list_file[:list_file.index('_')] for list_file in list_files]);
    
    classes_uni=np.unique(classes_new);

    print list_files[0];
    # print errs_old.shape;

    # arrs=np.load(os.path.join(dir_new,list_file))
    arrs=np.load(dir_new)
    errs_new=arrs['errors'];
    list_files=arrs['files'];
    classes_old=np.array([list_file[:list_file.index('_')] for list_file in list_files]);
    

    # print errs_new.shape;
    # class_name=list_file[list_file.rindex('/')+1:]
    # class_name=list_file[:list_file.index('_')];
    # print class_name

    classes_uni=np.unique(classes_new);
    print classes_uni;
    for class_curr in classes_uni:
        print class_curr
        print 'old';
        print np.mean(errs_old[classes_old==class_curr,:],axis=0);
        print 'new';
        print np.mean(errs_new[classes_new==class_curr,:],axis=0);
        print '___';

        

        # break;
def getDataSetAndVideoName(line):
    line_split=line.split('/');
    # print line_split
    data_set=line_split[3];
    video=line_split[-1];
    video=video[:video.index('.')];
    return data_set,video


def writeUniqueTrainingDataInfo(training_data_text,out_file_text):

    lines=util.readLinesFromFile(training_data_text);
    img_paths=[line[:line.index(' ')] for line in lines];
    p=multiprocessing.Pool(multiprocessing.cpu_count());
    vals=p.map(getDataSetAndVideoName,img_paths);
    vals_uz=zip(*vals);
    datasets=np.array(vals_uz[0]);
    videos=np.array(vals_uz[1]);
    new_tuples=[];
    for dataset_curr in np.unique(datasets):
        idx_rel=np.where(datasets==dataset_curr)[0];
        videos_rel=videos[idx_rel];
        videos_rel=np.unique(videos_rel);
        for video_curr in videos_rel:
            tuple_curr=(dataset_curr,video_curr)
            new_tuples.append(tuple_curr);

    vals_uni=[' '.join(val_curr) for val_curr in new_tuples];
    util.writeFile(out_file_text,vals_uni);

def writeTrainValTxtExcluded(train_new_text,val_new_text,out_file_text,training_data_text,percent_exclude):

    lines=util.readLinesFromFile(out_file_text);
    info=[tuple(line_curr.split(' ')) for line_curr in lines];
    class_rec={};
    for dataset,video, in info:
        if dataset=='youtube':
            video_split=video.split('_');
            class_curr=video_split[0];
            if class_curr in class_rec:
                class_rec[class_curr].append(video);
            else:
                class_rec[class_curr]=[video];

    list_exclude_all=[];
    for class_curr in class_rec.keys():
        num_exclude=int(math.ceil(len(class_rec[class_curr])*percent_exclude));
        list_shuffle=class_rec[class_curr];
        random.shuffle(list_shuffle);
        list_exclude=list_shuffle[:num_exclude];
        list_exclude_all=list_exclude_all+list_exclude;


    lines=util.readLinesFromFile(training_data_text);
    # print len(lines);
    lines_to_keep=[];
    lines_to_exclude=[];
    for line in lines:
        img=line[:line.index(' ')];
        img_split=img.split('/');
        if img_split[3]=='youtube' and (img_split[4] in list_exclude_all):
            lines_to_exclude.append(line);
            # print img
            continue;
        else:
            lines_to_keep.append(line);
            
    print len(lines_to_keep),len(lines_to_exclude),len(lines),len(lines_to_keep)+len(lines_to_exclude)

    util.writeFile(train_new_text,lines_to_keep);
    util.writeFile(val_new_text,lines_to_exclude);


def saveMinEqualFrames(train_new_text,out_file_idx,out_file_eq,includeHuman=True):
    lines=util.readLinesFromFile(train_new_text);
    img_paths=[line[:line.index(' ')] for line in lines];
    p=multiprocessing.Pool(multiprocessing.cpu_count());
    vals=p.map(getDataSetAndVideoName,img_paths);
    [dataset,video]=zip(*vals)
    dataset=np.array(dataset);
    print np.unique(dataset);

    frame_idx_rec={};
    if includeHuman:
        frame_idx_rec['human']=list(np.where(dataset=='hmdb_try_2')[0]);

    for idx,video_curr in enumerate(video):
        if dataset[idx]=='youtube':
            class_curr=video_curr[:video_curr.index('_')];
            if class_curr in frame_idx_rec:
                frame_idx_rec[class_curr].append(idx);
            else:
                frame_idx_rec[class_curr]=[idx];

    for class_curr in frame_idx_rec.keys():
        print class_curr,len(frame_idx_rec[class_curr]);


    min_frames=min([len(val_curr) for val_curr in frame_idx_rec.values()]);
    print 'min_frames',min_frames

    idx_to_pick=[];

    for class_curr in frame_idx_rec.keys():
        idx_curr=frame_idx_rec[class_curr];
        random.shuffle(idx_curr);
        idx_to_pick.extend(idx_curr[:min_frames]);

        # print class_curr,len(frame_idx_rec[class_curr]);

    idx_all=[idx_curr for idx_curr_all in frame_idx_rec.values() for idx_curr in idx_curr_all];
    print len(idx_all),len(lines);
    assert len(idx_all)==len(lines);

    idx_all.sort();
    print  idx_all==list(range(len(lines)));
    assert idx_all==list(range(len(lines)));
    lines_to_keep=[lines[idx_curr] for idx_curr in idx_to_pick];
    print len(lines_to_keep);

    np.save(out_file_idx,np.array(idx_to_pick))
    util.writeFile(out_file_eq,lines_to_keep);



def main():

    training_data_text='/disk2/marchExperiments/finetuning_youtube_hmdb_llr/train_both.txt';
    dir_meta='/disk2/mayExperiments';
    # dir_meta='/group/leegrp/maheen_data'
    dir_curr='flow_eval';
    dir_train=os.path.join(dir_meta,'finetuning_youtube_hmdb_llr');
    util.mkdir(dir_train);
    train_new_text=os.path.join(dir_train,'train.txt');
    
    out_file_eq=os.path.join(dir_train,'train_eq.txt');
    out_file_idx=os.path.join(dir_train,'train_eq_idx.npy');

    val_file_eq=os.path.join(dir_train,'val_eq.txt');
    val_file_idx=os.path.join(dir_train,'val_eq_idx.npy');

    
    val_new_text=os.path.join(dir_train,'val.txt');

    percent_exclude=0.1;
    out_file_text_org=os.path.join(dir_train,'train_info_org.txt');
    out_file_text=os.path.join(dir_meta,dir_curr,'train_info.txt');


    dir_test_meta=os.path.join(dir_meta,'test_youtube_flow');
    util.mkdir(dir_test_meta);
    
    # dir_test=os.path.join(dir_test_meta,'original_model');
    # util.mkdir(dir_test);
    # model_file='/home/maheenrashid/Downloads/debugging_jacob/optical_flow_prediction_test/examples/opticalflow/final.caffemodel'

    dir_test=os.path.join(dir_test_meta,'50000_model');
    util.mkdir(dir_test);
    model_file='/disk2/mayExperiments/finetuning_youtube_hmdb_llr/OptFlow_youtube_hmdb_iter_50000.caffemodel'

    
    clusters_file='/home/maheenrashid/Downloads/debugging_jacob/optical_flow_prediction_test/examples/opticalflow/clusters.mat';
    gpu=0;


    lines=util.readLinesFromFile(val_new_text);
    img_paths=[line[:line.index(' ')] for line in lines];
    # po.script_saveFlos(img_paths,dir_test,gpu,model_file,clusters_file,overwrite=False)

    dir_pred=os.path.join(dir_test,'flo_files');
    print dir_pred;
    dir_gt=os.path.join(dir_test_meta,'gt_flow');
    flo_files=[img_name+'.flo' for img_name in util.getFileNames(img_paths,False)];
    # flo_files=flo_files[:1];
    out_file_err=dir_test+'_err_all.npz';

    # getErrorMultiProc(dir_gt,dir_pred,flo_files,out_file_err,isFlo=True)


    out_file_err_old=os.path.join(dir_test_meta,'original_model_err_all.npy.npz');

    script_compYoutubePerf(out_file_err_old,out_file_err)

    return
    writeTrainValTxtExcluded(train_new_text,val_new_text,out_file_text_org,training_data_text,percent_exclude)
    writeUniqueTrainingDataInfo(train_new_text,out_file_text)
    
    lines=util.readLinesFromFile(out_file_text_org);
    info=[tuple(line_curr.split(' ')) for line_curr in lines];
    class_rec={};
    for dataset,video, in info:
        if dataset=='youtube':
            video_split=video.split('_');
            class_curr=video_split[0];
            if class_curr in class_rec:
                class_rec[class_curr].append(video);
            else:
                class_rec[class_curr]=[video];
        else:
            if dataset in class_rec:
                class_rec[dataset].append(video);
            else:
                class_rec[dataset]=[video];

    for class_curr in class_rec.keys():
        print class_curr,len(class_rec[class_curr])

    return
    paths_replace=['/disk2/marchExperiments','/group/leegrp/maheen_data'];
    transfer_file=os.path.join(dir_test,'transfer.txt');
    img_paths_replace=[file_curr.replace(paths_replace[0],paths_replace[1]).replace('.jpg','.flo').replace('images_transfer','images') for file_curr in img_paths];
    print len(img_paths_replace);
    print img_paths_replace[0];
    util.writeFile(transfer_file,img_paths_replace);
    print transfer_file;


    return
    
    writeUniqueTrainingDataInfo(training_data_text,out_file_text_org)
    return
    
    return
    # dir_youtube=os.path.join(dir_meta,'youtube');
    # dir_hmdb=os.path.join(dir_meta,'hmdb');

    youtube_data=[dir_curr for dir_curr in os.listdir(dir_youtube)];
    hmdb_data=[dir_curr for dir_curr in os.listdir(dir_hmdb)];
    print len(youtube_data);
    train_info=util.readLinesFromFile(out_file_text);
    youtube_data_train=[];
    hmdb_data_train=[];
    for info_curr in train_info:
        dataset=info_curr[:info_curr.index(' ')];
        video=info_curr[info_curr.index(' ')+1:];
        if dataset=='youtube':
            youtube_data_train.append(video);
        else:
            hmdb_data_train.append(video);

    youtube_data_train=set(youtube_data_train);

    leftover_youtube=set(youtube_data).difference(youtube_data_train);
    leftover_hmdb=set(hmdb_data).difference(hmdb_data_train);

    flo_files_all=[];
    for idx_hmdb,hmdb_curr in enumerate(leftover_hmdb):
        print idx_hmdb,len(leftover_hmdb),hmdb_curr;
        dir_curr=os.path.join(dir_hmdb,hmdb_curr,'images');

        flo_files=[os.path.join(dir_curr,file_curr) for file_curr in os.listdir(dir_curr) if file_curr.endswith('.flo')];
        if len(flo_files)==0:
            print 'PROBLEM';
        flo_files_all=flo_files_all+flo_files;

    print len(flo_files_all);
    print flo_files_all[0];
    print flo_files_all[100];

    util.writeFile(heldout_hmdb,flo_files_all);


    # /group/leegrp/maheen_data/hmdb/AmericanGangster_drink_u_nm_np1_fr_med_40
    # print leftover;
    # print len(leftover_youtube);
    # print len(leftover_hmdb),len(hmdb_data_train);



    # out_dir='/disk2/mayExperiments/flow_eval'
    # out_dir='/disk2/mayExperiments/flow_eval'
    # out_file_text=os.path.join(out_dir,'train_info.txt');



if __name__=='__main__':
    main();