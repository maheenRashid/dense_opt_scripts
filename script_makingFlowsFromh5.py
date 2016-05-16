import h5py
import numpy as np;
import util;
import scipy.io;
import os;
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt;
import multiprocessing;
import visualize;
import random;
import script_resizingFlos as srf;
import shutil;

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

def saveFlowImage((h5_file,img_file,out_file_flow,C,idx)):
    print idx,out_file_flow
    if os.path.exists(out_file_flow):
        print 'SKIP'
        return;

    if not os.path.exists(img_file):
        print 'SKIP NO IMG',img_file
        return;

    theImage =scipy.misc.imread(img_file);
    flow=getFlowMat(h5_file,C);
    mag=np.power(np.power(flow[:,:,0],2)+np.power(flow[:,:,1],2),0.5)
    flow=np.dstack((flow,mag));
    flow = cv2.resize(flow, (theImage.shape[1],theImage.shape[0]))

    min_v=np.min(flow);
    flow=flow+abs(min_v);
    max_v=np.max(flow);
    flow=flow/max_v
    flow=flow*255;
    # print flow.shape
    scipy.misc.imsave(out_file_flow,flow);

def getFlowMat(h5_file,C):

    with h5py.File(h5_file,'r') as hf:
        data = hf.get('Outputs')
        np_data = np.array(data)

    flow=assignToFlowSoft(np_data.ravel(),C);

    return flow;
    
def script_writeNegFile():


    dir_flow='/disk2/aprilExperiments/deep_proposals/flow/results_neg'
    out_text='/disk2/aprilExperiments/deep_proposals/flow/test_neg.txt';
    # util.mkdir(dir_flow);

    neg_text='/disk2/marchExperiments/deep_proposals/negatives.txt';
    lines=util.readLinesFromFile(neg_text);
    neg_images=[line_curr[:line_curr.index(' ')] for line_curr in lines];
    neg_images=neg_images[:100];
    to_write=[neg_image+' 1' for neg_image in neg_images]
    util.writeFile(out_text,to_write);

def getFlowImFiles(dir_meta):
    im_files=[];
    flow_files=[os.path.join(dir_meta,file_curr) for file_curr in os.listdir(dir_meta) if file_curr.endswith('.png')];
    im_files=[util.readLinesFromFile(flow_file.replace('.png','.txt'))[0].strip() for flow_file in flow_files];

    return flow_files,im_files;

def writeNewFileWithFlow(pos_data,flow_files,im_files,out_file_pos):

    pos_data_1=[pos_data_curr[:pos_data_curr.index(' ')] for pos_data_curr in pos_data]

    new_pos_data=[]
    for idx_flow_file,flow_file in enumerate(flow_files):
        img_file_corr=im_files[idx_flow_file];
        pos_data_corr=pos_data[pos_data_1.index(img_file_corr)];
        new_pos=pos_data_corr+' '+flow_file;
        new_pos_data.append(new_pos);

    print new_pos_data[0];

    util.writeFile(out_file_pos,new_pos_data);

def getFlowGT(tif,C):
    tif=tif-1;
    img=np.zeros(tif.shape);
    for r in range(img.shape[0]):
        for c in range(img.shape[1]):
            val_x=tif[r,c,0];
            val_y=tif[r,c,1];
            img[r,c,0]=C[val_x,0];
            img[r,c,1]=C[val_y,1];
    return img;

def getClusters(clusters_file):
    
    with h5py.File(clusters_file,'r') as hf:
        print hf.keys();
        C=np.array(hf.get('C'));
    C=C.T    
    return C;

def makeRelPath(file_curr,replace_str='/disk2'):
    count=file_curr.count('/');
    # end_curr=file_curr[file_curr.rindex('/')+1:];
    str_replace='../'*count
    rel_str=file_curr.replace(replace_str,str_replace);
    return rel_str;

def bringToImageFrame(flo,im_shape):
    flo_shape_org=flo.shape;
    # print flo[0,0,0]
    flo_resize=cv2.resize(flo,(im_shape[1],im_shape[0]));
    # print flo_resize[0,0,0]
    flo_resize[:,:,0]=flo_resize[:,:,0]*im_shape[1]/float(flo_shape_org[1]);
    # print flo_resize[0,0,0]
    flo_resize[:,:,1]=flo_resize[:,:,1]*im_shape[0]/float(flo_shape_org[0]);
    return flo_resize;


def resizeSP(flo,im_shape):
    gt_flo_sp=np.zeros((im_shape[0],im_shape[1],2));
    for layer_idx in range(flo.shape[2]):
        min_layer=np.min(flo[:,:,layer_idx]);
        max_layer=np.max(flo[:,:,layer_idx]);
        gt_flo_sp_curr=scipy.misc.imresize(flo[:,:,layer_idx],im_shape);
        print np.max(gt_flo_sp_curr)
        gt_flo_sp_curr=gt_flo_sp_curr/float(max(np.max(gt_flo_sp_curr),np.finfo(float).eps));
        gt_flo_sp_curr=gt_flo_sp_curr*abs(max_layer-min_layer);
        gt_flo_sp_curr=gt_flo_sp_curr-abs(min_layer);
        gt_flo_sp[:,:,layer_idx]=gt_flo_sp_curr;
    return gt_flo_sp;


def bringToImageFrameSP(flo,im_shape):

    flo_shape_org=flo.shape;
    # print flo[0,0,0]
    gt_flo_sp=resizeSP(flo,im_shape);
    # print gt_flo_sp[0,0,0]
    gt_flo_sp[:,:,0]=gt_flo_sp[:,:,0]*im_shape[1]/float(flo_shape_org[1]);
    # print gt_flo_sp[0,0,0]
    gt_flo_sp[:,:,1]=gt_flo_sp[:,:,1]*im_shape[0]/float(flo_shape_org[0]);
    
    return gt_flo_sp

def script_saveFloAsNpPred(clusters_file,h5_files,img_files,dir_flo_im):

    with h5py.File(clusters_file,'r') as hf:
        # print hf.keys();
        C=np.array(hf.get('C'));
    C=C.T    


    for idx_h5_file,h5_file in enumerate(h5_files):
        # print idx_h5_file
        img_file=img_files[idx_h5_file];
        # print img_file
        img_name=img_file[img_file.rindex('/')+1:img_file.rindex('.')];
        
        flo=getFlowMat(h5_file,C);
        # print flo.shape
        # print np.min(flo),np.max(flo);
        im=scipy.misc.imread(img_file);
        flo_resize=resizeSP(flo,im.shape);
        # print flo_resize.shape
        # print np.min(flo_resize),np.max(flo_resize);
        out_file_flo=os.path.join(dir_flo_im,img_name+'.npy');
        np.save(out_file_flo,flo_resize);
        break;

    # return seg;

def writeh5ImgFile(dir_neg,out_file_match):

    lines=[];
    h5_files=[os.path.join(dir_neg,file_curr) for file_curr in os.listdir(dir_neg) if file_curr.endswith('.h5')];
    print len(h5_files)
    for idx_file_curr,file_curr in enumerate(h5_files):
        if idx_file_curr%100==0:
            print idx_file_curr
        img_file=util.readLinesFromFile(file_curr.replace('.h5','.txt'))[0].strip();
        # print file_curr,img_file
        lines.append(file_curr+' '+img_file);

    util.writeFile(out_file_match,lines);


def main():

    dir_meta='/disk2/aprilExperiments/deep_proposals/flow_all_humans/';
    dir_other_images='/disk2/aprilExperiments/deep_proposals/flow_neg/flo_subset_for_pos_cropped';
    dir_pos=os.path.join(dir_meta,'results');
    out_file_match=os.path.join(dir_meta,'match.txt');

    dir_curr='/disk2/aprilExperiments/deep_proposals/flow_neg_subset_debug/results';
    dir_small_im=dir_curr+'_images'
    dir_full_crops='/disk2/aprilExperiments/deep_proposals/flow_neg/flo_subset_for_pos_cropped'

    list_files=[os.path.join(dir_curr,file_curr) for file_curr in os.listdir(dir_curr) if file_curr.endswith('.h5')]
    img_files=[];
    h5_names=[];
    for list_file in list_files:
        img_file=util.readLinesFromFile(list_file.replace('.h5','.txt'))[0].strip();

        h5_names.append(list_file[list_file.rindex('/')+1:list_file.rindex('.')]);
        
        img_file=img_file[img_file.rindex('/')+1:];
        img_files.append(img_file);


    # h5_names_sorted=[int(name) for name in h5_names];
    # h5_names_sorted.sort();



    out_file_html=os.path.join(dir_full_crops,'comparison.html');
    imgs_full_crops=[file_curr for file_curr in os.listdir(dir_full_crops) if file_curr.endswith('.png')];

    img_paths=[];
    captions=[];
    rel_path=['/disk2','../../../..'];
    for img_file in imgs_full_crops:
        img_file_full=os.path.join(dir_full_crops,img_file);
        print img_files[0],img_file
        
        # h5_name_rel=int(h5_names[img_files.index(img_file)])
        img_file_small=os.path.join(dir_small_im,h5_names[img_files.index(img_file)]+'.jpg');



        img_paths.append([img_file_full.replace(rel_path[0],rel_path[1]),img_file_small.replace(rel_path[0],rel_path[1])]);
        captions.append(['Crop full flow','Flow on Cropped']);
    util.writeHTML(out_file_html,img_paths,captions,240,240);





    # # util.mkdir(out_dir_matches);
    # # writeh5ImgFile(dir_pos,out_file_match);
    # img_files_to_find=[file_curr for file_curr in os.listdir(dir_other_images) if file_curr.endswith('.png')];
    # lines=util.readLinesFromFile(out_file_match);
    # just_imgs=[line[line.rindex('/')+1:] for line in lines];
    # idx_matches=[];
    # # idx_matches=[just_imgs.index(file_curr) for file_curr in img_files_to_find];
    # # h5_files=[lines[idx_match][:lines[idx_match].index(' ')] for idx_match in idx_matches];
    # # print h5_files[0],len(h5_files);
    # # for file_curr in h5_files:
    # for img_file_to_find in img_files_to_find:
    #     if img_file_to_find in just_imgs:
    #         idx_match=just_imgs.index(img_file_to_find)
    #     else:
    #         continue;
    #     file_curr=lines[idx_match][:lines[idx_match].index(' ')]
    #     out_file_curr=os.path.join(out_dir_matches,file_curr[file_curr.rindex('/')+1:]);
    #     shutil.copyfile(file_curr,out_file_curr);

    #     file_curr=file_curr.replace('.h5','.txt');
    #     out_file_curr=out_file_curr.replace('.h5','.txt');
        
    #     shutil.copyfile(file_curr,out_file_curr);

    # # for file_curr in img_files_to_find:
    # #     idx_match=lines.index(file_curr);
    # #     idx_matches.append(idx_match)

    # return
    # dir_curr='/disk2/aprilExperiments/deep_proposals/flow_neg/flo_subset_for_pos_cropped';
    # visualize.writeHTMLForFolder(dir_curr,'.jpg');

    return    
    dir_meta='/disk2/aprilExperiments/deep_proposals/flow_neg/';
    dir_neg=os.path.join(dir_meta,'results');
    out_file_rec=os.path.join(dir_meta,'ims_to_analyze.npz')
    out_file_match=os.path.join(dir_meta,'match.txt');

    lines=util.readLinesFromFile('/disk2/aprilExperiments/deep_proposals/positives_person.txt');

    clusters_file='/home/maheenrashid/Downloads/debugging_jacob/optical_flow_prediction_test/examples/opticalflow/clusters.mat';

    arrs=np.load(out_file_rec)
    negs=arrs['negs'];
    h5_files=[line[:line.index(' ')] for line in negs];
    dir_flo_im=os.path.join(dir_meta,'flo_subset_for_pos');
    util.mkdir(dir_flo_im);
    replace_paths=['','']; 

    srf.script_saveFloAsNpPred(clusters_file,h5_files,dir_flo_im,replace_paths)

    return
    print lines[0];
    imgs=[line[:line.index(' ')] for line in lines];

    num_to_keep=100;
    random.shuffle(imgs);
    imgs=imgs[:num_to_keep];
    imgs_neg=[img[img.rindex('/')+1:img.rindex('_')] for img in imgs];
    print imgs_neg[0]

    lines_neg=util.readLinesFromFile(out_file_match);
    print 'lines_neg',len(lines_neg);
    print lines_neg[0];
    rel_imgs=[];
    for idx_lines,line in enumerate(lines_neg):
        # if idx_lines%100==0:
        #     print idx_lines;
        rel_img_part=line[line.rindex(' ')+1:];
        rel_img_part=rel_img_part[rel_img_part.rindex('/')+1:rel_img_part.rindex('.')];
        rel_imgs.append(rel_img_part);
    print len(rel_imgs);
    print rel_imgs[0];
    
    idx_rel=[];
    paths_rel=[];
    for idx_curr,img_curr in enumerate(imgs_neg):
        print idx_curr
        idx=rel_imgs.index(img_curr);
        idx_rel.append(idx);
        paths_rel.append(lines_neg[idx]);

    imgs=np.array(imgs);
    paths_rel=np.array(paths_rel);
    np.savez(out_file_rec,pos=imgs,negs=paths_rel);


        # lines=[];
    # h5_files=[os.path.join(dir_neg,file_curr) for file_curr in os.listdir(dir_neg) if file_curr.endswith('.h5')];
    # print len(h5_files)
    # for idx_file_curr,file_curr in enumerate(h5_files):
    #     if idx_file_curr%100==0:
    #         print idx_file_curr
    #     img_file=util.readLinesFromFile(file_curr.replace('.h5','.txt'))[0].strip();
    #     # print file_curr,img_file
    #     lines.append(file_curr+' '+img_file);

    # util.writeFile(out_file_match,lines);


    return

    dir_meta='/disk2/aprilExperiments/flo_debug/results'
    h5_file=os.path.join(dir_meta,'0.h5');
    clusters_file='/home/maheenrashid/Downloads/debugging_jacob/optical_flow_prediction_test/examples/opticalflow/clusters.mat';
    img_file = util.readLinesFromFile(h5_file.replace('.h5','.txt'))[0].strip();
    
    with h5py.File(clusters_file,'r') as hf:
        # print hf.keys();
        C=np.array(hf.get('C'));
    C=C.T    

    flow_mat = getFlowMat(h5_file,C);
    im=scipy.misc.imread(img_file);

    flow_mat=cv2.resize(flow_mat,(im.shape[1],im.shape[0]));
    # print flow_mat.shape,im.shape
    out_file_mat=h5_file.replace('.h5','.mat');
    scipy.io.savemat(out_file_mat,{'N':flow_mat});

    out_file_im=os.path.join(dir_meta,'from_mat.png');

    string="img_file='"+img_file+"';flo_mat='"+out_file_mat+"';out_file='"+out_file_im+"';"
    print string
    
    
    return
    out_dir='/disk2/aprilExperiments/flo_all_predictions/results'
    dir_flo_im='/disk2/aprilExperiments/flo_all_predictions/flo_npy';
    util.mkdir(dir_flo_im);
    clusters_file='/home/maheenrashid/Downloads/debugging_jacob/optical_flow_prediction_test/examples/opticalflow/clusters.mat';
    vision3_path='/disk2/marchExperiments';
    hpc_path='/group/leegrp/maheen_data';
    h5_files=[os.path.join(out_dir,file_curr) for file_curr in os.listdir(out_dir) if file_curr.endswith('.h5')];
    
    img_files=[];
    for h5_file in h5_files[:100]:
        img_file=util.readLinesFromFile(h5_file.replace('.h5','.txt'))[0].strip();
        img_files.append(img_file);

    script_saveFloAsNpPred(clusters_file,h5_files,img_files,dir_flo_im)        
    

    return

    dir_pred='/disk2/aprilExperiments/flo_all_predictions'
    dir_im_meta='/disk2/marchExperiments/youtube'
    file_name='youtube_list_flo_paths.txt';
    test_file=os.path.join(dir_pred,'test.txt');

    flo_files=util.readLinesFromFile(os.path.join(dir_pred,file_name));
    print len(flo_files);
    random.shuffle(flo_files);

    im_paths=[];

    for flo_file in flo_files:
        flo_name=flo_file[flo_file.rindex('/')+1:];
        video_name=flo_name[:flo_name.index('.')];
        im_path=os.path.join(dir_im_meta,video_name,'images_transfer',flo_name[:flo_name.rindex('.')]+'.jpg');
        im_paths.append(im_path+' 1');
        # assert os.path.exists(im_path);
    
    print len(im_paths);
    print im_paths[0];

    util.writeFile(test_file,im_paths);
    print test_file;

    



    return
    dir_flo_pred='/disk2/aprilExperiments/flo_subset_predictions/pred_flo_im';
    dir_flo_gt='/disk2/aprilExperiments/flo_im';
    dir_im_meta='/disk2/marchExperiments/youtube'
    out_dir_debug='/disk2/aprilExperiments/flo_debug';


    flo_names=[file_curr for file_curr in os.listdir(dir_flo_pred) if file_curr.endswith('.npy')];

    errors=[];

    for flo_name in flo_names[:100]:
    
        video_name=flo_name[:flo_name.index('.')];
        im_path=os.path.join(dir_im_meta,video_name,'images_transfer',flo_name[:flo_name.rindex('.')]+'.jpg');        

        gt_flo_file=os.path.join(dir_flo_gt,flo_name)
        # print gt_flo_file
        gt_flo=np.load(gt_flo_file);
        # print im_path.replace('/disk2','vision3.cs.ucdavis.edu:1000');
        im=scipy.misc.imread(im_path);
        
        gt_flo_cv2=bringToImageFrame(gt_flo,im.shape);
        gt_flo_sp=bringToImageFrameSP(gt_flo,im.shape);
        tol=1.0
        error=np.sum(abs(gt_flo_sp-gt_flo_cv2)<tol)/float(gt_flo_sp.size);
        errors.append(error);

    print min(errors),max(errors),np.mean(errors);

    


    return

    out_file=os.path.join(out_dir_debug,flo_name);
    out_file_mat=os.path.join(out_dir_debug,flo_name[:flo_name.rindex('.')]+'.mat');
    scipy.io.savemat(out_file_mat,{'img':gt_flo});
    print out_file
    np.save(out_file,gt_flo);
    out_file_gt_flo=os.path.join(out_dir_debug,'flo_gt_no_rescale.png');
    print out_file_gt_flo


    return
    for flo_name in flo_names[:1]:
    
        video_name=flo_name[:flo_name.index('.')];
        im_path=os.path.join(dir_im_meta,video_name,'images_transfer',flo_name[:flo_name.rindex('.')]+'.jpg');
        im=scipy.misc.imread(im_path);
        # print im.shape
        gt_flo=np.load(os.path.join(dir_flo_gt,flo_name));
        pred_flo=np.load(os.path.join(dir_flo_pred,flo_name));
        # print gt_flo.shape
        # print pred_flo.shape

        # pred_flo=bringToImageFrame(pred_flo,im.shape);
        mag_pred_flo_bef=np.power(np.power(pred_flo[:,:,0],2)+np.power(pred_flo[:,:,1],2),0.5)
        pred_flo=cv2.resize(pred_flo,(im.shape[1],im.shape[0]));
        mag_pred_flo_aft=np.power(np.power(pred_flo[:,:,0],2)+np.power(pred_flo[:,:,1],2),0.5)

        mag_gt_flo_bef=np.power(np.power(gt_flo[:,:,0],2)+np.power(gt_flo[:,:,1],2),0.5)
        gt_flo=bringToImageFrame(gt_flo,im.shape);
        mag_gt_flo_aft=np.power(np.power(gt_flo[:,:,0],2)+np.power(gt_flo[:,:,1],2),0.5)

        print pred_flo.shape,gt_flo.shape
        # pred_flo = cv2.resize(pred_flo, (im.shape[1],im.shape[0]))
        # gt_flo = cv2.resize(gt_flo, (im.shape[1],im.shape[0]));

        # pred_flo=makeUnitMag(pred_flo);
        # gt_flo=makeUnitMag(gt_flo);

        pred_values=[np.sort(pred_flo[:,:,0].ravel()),np.sort(pred_flo[:,:,1].ravel())]
        
        gt_values=[np.sort(gt_flo[:,:,0].ravel()),np.sort(gt_flo[:,:,1].ravel())]

        # print np.max(np.power(np.power(pred_values[0],2)+np.power(pred_values[1],2),0.5))
        # print np.max(np.power(np.power(gt_values[0],2)+np.power(gt_values[1],2),0.5))

        # print np.min(pred_values[0]),np.min(gt_values[0]);
        # print np.min(pred_values[1]),np.min(gt_values[1]);

        # print gt_values[0].shape,gt_values[1].shape

        
        util.mkdir(out_dir_debug);

        fig=plt.figure();
        ax1 = fig.add_subplot(221)
        ax1.plot(np.sort(mag_pred_flo_bef.ravel()));

        ax2 = fig.add_subplot(222)
        ax2.plot(np.sort(mag_pred_flo_aft.ravel()))

        ax3 = fig.add_subplot(223)
        ax3.plot(np.sort(mag_gt_flo_bef.ravel()))

        ax4 = fig.add_subplot(224)
        ax4.plot(np.sort(mag_gt_flo_aft.ravel()))


        plt.tight_layout()
        plt.savefig(os.path.join(out_dir_debug,'ranges.png'));




    return

    dir_flo_org='/disk2/aprilExperiments/flo_im';
    dir_flo_pred='/disk2/aprilExperiments/flo_subset_predictions/pred_flo_im';
    dir_im_meta='/disk2/marchExperiments/youtube'
    im_names=[file_curr[:file_curr.rindex('_')] for file_curr in os.listdir(dir_flo_org) if file_curr.endswith('_x.png')];
    
    out_file_html='/disk2/aprilExperiments/flo_subset_predictions/visualizeFlosComparison.html';
    img_paths_all=[];
    captions_all=[];
    for im_name in im_names:
        video_name=im_name[:im_name.index('.')];
        jpg_file=os.path.join(dir_im_meta,video_name,'images_transfer',im_name+'.jpg');

        x_flo_org=os.path.join(dir_flo_org,im_name+'_x.png');
        y_flo_org=os.path.join(dir_flo_org,im_name+'_y.png');
        
        x_flo_pred=os.path.join(dir_flo_pred,im_name+'_x.png');
        y_flo_pred=os.path.join(dir_flo_pred,im_name+'_y.png');

        row=[makeRelPath(jpg_file),makeRelPath(x_flo_org),makeRelPath(y_flo_org),makeRelPath(x_flo_pred),makeRelPath(y_flo_pred)];
        captions=['im','org_x','org_y','pred_x','pred_y'];
        img_paths_all.append(row);
        captions_all.append(captions)

    visualize.writeHTML(out_file_html,img_paths_all,captions_all,200,200);

    return
    results_dir='/disk2/aprilExperiments/flo_subset_predictions/results';
    dir_flo_im='/disk2/aprilExperiments/flo_subset_predictions/pred_flo_im';
    util.mkdir(dir_flo_im);

    clusters_file='/home/maheenrashid/Downloads/debugging_jacob/optical_flow_prediction_test/examples/opticalflow/clusters.mat';

    with h5py.File(clusters_file,'r') as hf:
        print hf.keys();
        C=np.array(hf.get('C'));
    C=C.T    

    # img_files=[];
    h5_files=[os.path.join(results_dir,file_curr) for file_curr in os.listdir(results_dir) if file_curr.endswith('.h5')];
    for idx_h5_file,h5_file in enumerate(h5_files):
        print idx_h5_file
        img_file=util.readLinesFromFile(h5_file.replace('.h5','.txt'))[0];
        img_name=img_file[img_file.rindex('/')+1:img_file.rindex('.')];
        # img_files.append(img_file);

        flo=getFlowMat(h5_file,C);
        out_file_flo=os.path.join(dir_flo_im,img_name+'.npy');
        np.save(out_file_flo,flo);

        out_file_flo_x=os.path.join(dir_flo_im,img_name+'_x.png');
        out_file_flo_y=os.path.join(dir_flo_im,img_name+'_y.png');
        visualize.saveMatAsImage(flo[:,:,0],out_file_flo_x);
        visualize.saveMatAsImage(flo[:,:,1],out_file_flo_y);

    visualize.writeHTMLForFolder(dir_flo_im,ext='_x.png',height=300,width=300)


    return

    # pos_file_org='/disk2/aprilExperiments/deep_proposals/positives_person.txt';
    # flow_dir='/disk2/aprilExperiments/deep_proposals/flow_all_humans/results_flow';
    # out_dir='/disk2/aprilExperiments/dual_flow/onlyHuman_all_xavier';

    # pos_file_new=os.path.join(out_dir,'positives.txt');

    pos_file_org='/disk2/marchExperiments/deep_proposals/negatives.txt';
    flow_dir='/disk2/aprilExperiments/deep_proposals/flow_neg/results_flow';
    out_dir='/disk2/aprilExperiments/dual_flow/onlyHuman_all_xavier';

    pos_file_new=os.path.join(out_dir,'negatives.txt');
    

    pos_data=util.readLinesFromFile(pos_file_org);
    pos_data_new=[];
    for idx_pos_data_curr,pos_data_curr in enumerate(pos_data):
        if idx_pos_data_curr%100==0:
            print idx_pos_data_curr

        img_name=pos_data_curr[:pos_data_curr.index(' ')];
        img_name=img_name[img_name.rindex('/')+1:];
        flow_name=img_name[:img_name.rindex('.')]+'_flow.png';
        flow_path=os.path.join(flow_dir,flow_name);
        if os.path.exists(flow_path):
            pos_data_new_curr=pos_data_curr+' '+flow_path;
            pos_data_new.append(pos_data_new_curr);

    print len(pos_data_new)
    util.writeFile(pos_file_new,pos_data_new);


    return
    dir_flow='/disk2/aprilExperiments/deep_proposals/flow_all_humans/results';
    out_dir_flow_im='/disk2/aprilExperiments/deep_proposals/flow_all_humans/results_flow';
    util.mkdir(out_dir_flow_im);

    clusters_file='/home/maheenrashid/Downloads/debugging_jacob/optical_flow_prediction_test/examples/opticalflow/clusters.mat';

    with h5py.File(clusters_file,'r') as hf:
        print hf.keys();
        C=np.array(hf.get('C'));
    C=C.T


    h5_files=[os.path.join(dir_flow,file_curr) for file_curr in os.listdir(dir_flow) if file_curr.endswith('.h5')];
    img_files=[];
    for idx,h5_file in enumerate(h5_files):
        if idx%100==0:
            print idx;
        img_file=util.readLinesFromFile(h5_file.replace('.h5','.txt'))[0].strip();
        img_files.append(img_file);

    args=[];
    count_missing=0;
    for idx,(h5_file,img_file) in enumerate(zip(h5_files,img_files)):
        if not os.path.exists(img_file):
            count_missing=count_missing+1;
            continue;
        # print idx,h5_file,img_file
        out_file_flow=img_file[:img_file.rindex('.')]+'_flow.png';
        out_file_flow=out_file_flow[out_file_flow.rindex('/')+1:]
        out_file_flow=os.path.join(out_dir_flow_im,out_file_flow);

        arg_curr=(h5_file,img_file,out_file_flow,C,idx);
        args.append(arg_curr);

    print len(args);
    print count_missing;
    print len(img_files);
    # for arg in args[:10]:
    #     arg=list(arg);
        # print arg[:-2];

    print 'starting Pool';
    p=multiprocessing.Pool(8);
    p.map(saveFlowImage,args);


        # out_file_flow=h5_file.replace('.h5','.png');
        # saveFlowImage(h5_file,img_file,out_file_flow,C);
        # print h5_files.index(h5_file),out_file_flow
    

    return

     # /disk2/marchExperiments/youtube/dog_11_1/images_transfer/dog_11_1.avi_000326.tif

    out_dir_results='/disk2/aprilExperiments/testing_flow/debug/results'

    out_dir='/disk2/aprilExperiments/testing_flow/debug';
    img_path='/disk2/marchExperiments/youtube/dog_11_1/images_transfer/dog_11_1.avi_000326.jpg'
    tif_path='/disk2/marchExperiments/youtube/dog_11_1/images_transfer/dog_11_1.avi_000326.tif';
    
    pred_path=os.path.join(out_dir_results,'0.h5');

    clusters_file='/home/maheenrashid/Downloads/debugging_jacob/optical_flow_prediction_test/examples/opticalflow/clusters.mat';

    C=getClusters(clusters_file);
    

    img=scipy.misc.imread(img_path)
    tif=scipy.misc.imread(tif_path);
    tif=tif[:,:,:2];
    for idx in range(tif.shape[2]):
        print np.min(tif[:,:,idx]),np.max(tif[:,:,idx]);
    flo=getFlowGT(tif,C);

    with h5py.File(pred_path,'r') as hf:
        data = hf.get('Outputs')
        data = np.array(data)

    flo_pred=assignToFlowSoft(data,C)
    flo_pred=cv2.resize(flo_pred, (flo.shape[1],flo.shape[0]))

    print np.min(flo),np.max(flo);
    print np.min(flo_pred),np.max(flo_pred);

    print flo_pred.shape
    print flo.shape
    print img.shape
    print tif.shape

    out_img=os.path.join(out_dir,'img.png');
    tif_x=os.path.join(out_dir,'tif_x.png');
    tif_y=os.path.join(out_dir,'tif_y.png');
    flo_x=os.path.join(out_dir,'flo_x.png');
    flo_y=os.path.join(out_dir,'flo_y.png');

    flo_pred_x=os.path.join(out_dir,'flo_pred_x.png');
    flo_pred_y=os.path.join(out_dir,'flo_pred_y.png');


    plt.figure;plt.imshow(img);plt.savefig(out_img);plt.close();
    plt.figure;plt.imshow(tif[:,:,0]);plt.savefig(tif_x);plt.close();
    plt.figure;plt.imshow(tif[:,:,1]);plt.savefig(tif_y);plt.close();
    plt.figure;plt.imshow(flo[:,:,0]);plt.savefig(flo_x);plt.close();
    plt.figure;plt.imshow(flo[:,:,1]);plt.savefig(flo_y);plt.close();

    plt.figure;plt.imshow(flo_pred[:,:,0]);plt.savefig(flo_pred_x);plt.close();
    plt.figure;plt.imshow(flo_pred[:,:,1]);plt.savefig(flo_pred_y);plt.close();



    return

    dir_prop='/disk2/aprilExperiments/deep_proposals/addingFlow/';
    util.mkdir(dir_prop);    
    out_file_pos=os.path.join(dir_prop,'positives.txt');
    out_file_neg=os.path.join(dir_prop,'negatives.txt');

    pos_file_org='/disk2/aprilExperiments/deep_proposals/positives_person.txt';
    neg_file_org='/disk2/marchExperiments/deep_proposals/negatives.txt';
    dir_flow_pos='/disk2/aprilExperiments/deep_proposals/flow/results';
    dir_flow_neg='/disk2/aprilExperiments/deep_proposals/flow/results_neg';

    num_to_keep=100;

    flow_files,im_files=getFlowImFiles(dir_flow_pos);
    pos_data=util.readLinesFromFile(pos_file_org);
    pos_data=pos_data[:100];
    writeNewFileWithFlow(pos_data,flow_files,im_files,out_file_pos)

    flow_files,im_files=getFlowImFiles(dir_flow_neg);
    pos_data=util.readLinesFromFile(neg_file_org);
    pos_data=pos_data[:100];
    writeNewFileWithFlow(pos_data,flow_files,im_files,out_file_neg)

    return
    dir_flow='/disk2/aprilExperiments/deep_proposals/flow/results_neg';
    clusters_file='/home/maheenrashid/Downloads/debugging_jacob/optical_flow_prediction_test/examples/opticalflow/clusters.mat';

    with h5py.File(clusters_file,'r') as hf:
        print hf.keys();
        C=np.array(hf.get('C'));
    C=C.T


    h5_files=[os.path.join(dir_flow,file_curr) for file_curr in os.listdir(dir_flow) if file_curr.endswith('.h5')];
    img_files=[];
    for h5_file in h5_files:
        img_file=util.readLinesFromFile(h5_file.replace('.h5','.txt'))[0].strip();
        img_files.append(img_file);


    for h5_file,img_file in zip(h5_files,img_files):
        out_file_flow=h5_file.replace('.h5','.png');
        saveFlowImage(h5_file,img_file,out_file_flow,C);
        print h5_files.index(h5_file),out_file_flow
        
        





    return
    clusters_file='/home/maheenrashid/Downloads/debugging_jacob/optical_flow_prediction_test/examples/opticalflow/clusters.mat';
    with h5py.File(clusters_file,'r') as hf:
        print hf.keys();
        C=np.array(hf.get('C'));
    C=C.T
    print C.shape;



    # return
    theImage='/disk2/februaryExperiments/deep_proposals/positives/COCO_train2014_000000024215_61400.png';
    theImage =scipy.misc.imread(theImage);

    

    file_name='/disk2/aprilExperiments/deep_proposals/flow/results/44.h5';        

    with h5py.File(file_name,'r') as hf:
        print('List of arrays in this file: \n', hf.keys())
        data = hf.get('Outputs')
        np_data = np.array(data)
        print('Shape of the array dataset_1: \n', np_data.shape)

    flow=assignToFlowSoft(np_data.ravel(),C);

    # print np.min(flow),np.max(flow);

    # flow_new_x=scipy.misc.imresize(flow[:,:,0],(theImage.shape[0],theImage.shape[1]))

    # flow_new_y=scipy.misc.imresize(flow[:,:,1],(theImage.shape[0],theImage.shape[1]))
    # flow=np.dstack((flow_new_x,flow_new_y));
    # print np.min(flow),np.max(flow);
    flow = cv2.resize(flow, (240,240))
    print flow.shape
    flow_old=flow;
    # f=h5py.File(file_name,'r')
    # output=f.get('/Outputs')
    # f.close();

    # output=np.array(output);

    # return
    
    # h5disp(file_name, '/Outputs');
 #        theTemp = h5read(file_name, '/Outputs');
    # theOutput = [];
    #     for j = 1:size(theTemp,2)
    #   for k = 1:size(theTemp,1)
    #        for l = 1:size(theTemp,3)
    #       theOutput = [theOutput theTemp(k,j,l)];
    #        end
    #   end
 #            end
        
 #        N = assignToFlowSoft(theOutput(:), C);

    # N = imresize(N, [size(theImage,1) size(theImage,2)]);


    # return
    path_to_mat='/disk2/temp/checking_h5.mat';
    mat_data=scipy.io.loadmat(path_to_mat)
    flow_new=mat_data['N'];
    # print flow_new.shape

    # print flow[:3,:3,0];
    # print flow_new[:3,:3,0];
    # ans=np.isclose(flow_new,flow);
    # print ans
    # print ans
    # scipy.misc.imsave('/disk2/temp/a.png',255*np.dstack((np.dstack((ans[:,:,0],ans[:,:,0])),ans[:,:,0])))
    # scipy.misc.imsave('/disk2/temp/b.png',255*np.dstack((np.dstack((ans[:,:,1],ans[:,:,1])),ans[:,:,1])))


    # return

    out_flow_range_old='/disk2/temp/checking_h5_flow_old.png';
    out_flow_range_new='/disk2/temp/checking_h5_flow_new.png';
    out_flow_range_img='/disk2/temp/checking_h5_flow_img.png';

    plt.ion();

    plt.figure();
    plt.plot(np.sort(np.ravel(flow_old)));
    plt.savefig(out_flow_range_old);
    
    

    plt.figure();
    plt.plot(np.sort(np.ravel(flow_new)));
    plt.savefig(out_flow_range_new);

    # return
    out_flow_x='/disk2/temp/checking_h5_x.png';
    out_flow_y='/disk2/temp/checking_h5_y.png';
    out_mag='/disk2/temp/checking_h5_mag.png';
    
    mag=np.power(np.power(flow[:,:,0],2)+np.power(flow[:,:,1],2),0.5)
    flow=np.dstack((flow,mag));
    print flow.shape

    min_v=np.min(flow);
    flow=flow+abs(min_v);
    max_v=np.max(flow);
    flow=flow/max_v
    flow=flow*255;

    flow=scipy.misc.imread(out_mag);
    plt.figure();
    plt.plot(np.sort(np.ravel(flow[:,:,2])));
    plt.savefig(out_flow_range_img);
    
    return
    
    # flow=flow/np.dstack((mag,mag));
    # flow=flow
    # flow=flow+abs(np.min(flow))
    # flow=255*flow;


    # print np.min(flow[:,:,0]),np.max(flow[:,:,0]);
    # print np.min(flow[:,:,1]),np.max(flow[:,:,1]);
    for idx in range(flow.shape[2]):
        mag=flow[:,:,idx];
        print np.min(mag),np.max(mag);
    
    im_x=np.dstack((np.dstack((flow[:,:,0],flow[:,:,0])),flow[:,:,0]));
    im_y=np.dstack((np.dstack((flow[:,:,1],flow[:,:,1])),flow[:,:,1]));
    # mag=np.dstack((np.dstack((mag,mag)),mag));
    # mag=np.dstack((np.dstack((flow[:,:,0],flow[:,:,1])),mag))
    scipy.misc.imsave(out_flow_x,im_x);
    scipy.misc.imsave(out_flow_y,im_y);
    scipy.misc.imsave(out_mag,flow);



    # print mat_data;

if __name__=='__main__':
    main();