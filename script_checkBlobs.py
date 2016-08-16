import visualize;

import sys;
sys.path.append('/home/maheenrashid/Downloads/debugging_jacob/optical_flow_prediction/python/caffe/');
import caffe
# from caffe._caffe import Solver
import numpy as np;
import util;
import scipy.misc;
import caffe.io;
import random;
import cPickle as pickle;
import os;
import shutil;
import re;
import script_visualizeLoss as svl;
# import visualize;

def setUpData(img_paths,tif_paths,start_idx,batch_size,mean,crop_size):
    im_data=np.zeros((batch_size,3,200,200));
    tif_data=np.zeros((batch_size,400,1,1));
    img_paths_curr=img_paths[start_idx:start_idx+batch_size];

    for idx_im in range(im_data.shape[0]):
        # print idx_im
        if idx_im>len(img_paths):
            idx_curr=idx_im%len(img_paths);
        else:
            idx_curr=idx_im;

        im_curr=scipy.misc.imread(img_paths_curr[idx_curr]);
        tif_curr = scipy.misc.imread(tif_paths[idx_curr]);
        # tif_curr=tif_curr[:,:,:2];

        if len(im_curr.shape)<3:
            im_curr=np.dstack((im_curr,im_curr,im_curr));

        start_r=random.randint(0,im_curr.shape[0]-crop_size[0]);
        start_c=random.randint(0,im_curr.shape[1]-crop_size[1]);

        im_curr=im_curr[start_r:start_r+crop_size[0],start_c:start_c+crop_size[1],:]
        tif_curr=tif_curr[start_r:start_r+crop_size[0],start_c:start_c+crop_size[1],:];
        
        im_curr=np.transpose(im_curr,(2,0,1));
        tif_curr=scipy.misc.imresize(tif_curr,0.1,'nearest')[:,:,random.randint(0,1)];
        tif_curr=np.expand_dims(np.expand_dims(tif_curr.ravel(),axis=1),axis=2)
        tif_data[idx_im]=tif_curr;
        mean_curr=mean[:,start_r:start_r+crop_size[0],start_c:start_c+crop_size[1]]
        # print im_curr.shape;
        im_curr=im_curr-mean_curr;
        im_data[idx_im]=im_curr;
        tif_data[idx_im]=tif_curr;
        start_idx_new=idx_curr+1;

    return im_data,tif_data,start_idx_new;


def script_saveStandardMean():
    mean_proto_file='/home/maheenrashid/Downloads/debugging_jacob/opticalflow/opt_train_db.binaryproto';
    mean_standard_proto_file='/home/maheenrashid/Downloads/debugging_jacob/opticalflow/standard.binaryproto';
    mean_standard_file_npy='/disk2/temp/mean_standard_jacob.npy';

    # caffe.io.saveProtoAsNpy(mean_standard_proto_file,mean_standard_file_npy)
    arr=np.load(mean_standard_file_npy);
    print arr.shape
    print arr;

    return
    mean_file_npy='/disk2/temp/mean_jacob.npy';
    data = open(mean_proto_file , 'rb' ).read()
    print len(data);
    print type(data);
    # print data[:100]
    # return

    mean=np.load(mean_file_npy);
    # print mean.shape;

    

    mean_std=[122,117,104]
    for dim in range(mean.shape[0]):
        mean[dim,:,:]=mean_std[dim];

    
    mean=np.expand_dims(mean,axis=0);
    print mean.shape;


    blob=caffe.io.array_to_blobproto(mean);
    # print blob;
    string=blob.SerializeToString();
    print len(string);

    f=open(mean_standard_proto_file,'wb');
    f.write(string);
    f.close();

def script_oldRatioCheck():

    # mean_standard_proto_file='/home/maheenrashid/Downloads/debugging_jacob/opticalflow/standard.binaryproto';
    model_file='/home/maheenrashid/Downloads/debugging_jacob/optical_flow_prediction/examples/opticalflow/final.caffemodel';
    layers_to_copy=['conv1','conv2','conv3','conv4','conv5']

    # model_file='/disk2/mayExperiments/ft_youtube_hmdb_newClusters_layerMagic/OptFlow_youtube_hmdb__iter_5000.caffemodel';
    # layers_to_copy=['conv1','conv2','conv3','conv4','conv5','fc6_fix','fc7_fix','fc8_fix']

    # model_file='/disk2/mayExperiments/ft_youtube_hmdb_newClusters_layerMagic_llr/OptFlow_youtube_hmdb__iter_65000.caffemodel';
    # layers_to_copy=['conv1','conv2','conv3','conv4','conv5','fc6_fix','fc7_fix','fc8_fix']


    # deploy_file='/disk2/mayExperiments/ft_youtube_hmdb_newClusters_layerMagic/opt_train_coarse_xavier.prototxt';
    # solver_file='/disk2/mayExperiments/ft_youtube_hmdb_newClusters_layerMagic/train.prototxt';
    deploy_file='/disk2/mayExperiments/ft_youtube_hmdb_newClusters_layerMagic/deploy_debug.prototxt';
    solver_file='/disk2/mayExperiments/ft_youtube_hmdb_newClusters_layerMagic/solver_debug.prototxt';

    # layers_to_copy=['conv1','conv2','conv3','conv4','conv5','fc6_fix','fc7_fix','fc8_fix']
    # layers_to_explore=['conv1','conv2','conv3','conv4','conv5','fc6_fix','fc7_fix','fc8_fix']

    
    # ,'fc6','fc7','fc8']
    layers_to_explore=['conv1','conv2','conv3','conv4','conv5','fc6_fix','fc7_fix','fc8_fix']
    blobs_lr=[(0,0),(0,0),(0,0),
            # (10,20),(10,20),
            (0.1,0.2),(0.1,0.2),
            (1,2),(1,2),(1,2)]

    iterations=400;
    momentum=0.9;
    lr=0.000001;

    caffe.set_device(1)
    caffe.set_mode_gpu()



    solver=caffe.SGDSolver(solver_file);

    
    net_org=caffe.Net(deploy_file,model_file);
        
    # copy weights
    for layer_name in layers_to_copy:
        solver.net.params[layer_name][0].data[...]=net_org.params[layer_name][0].data;
        solver.net.params[layer_name][1].data[...]=net_org.params[layer_name][1].data;

    layer_names=list(solver.net._layer_names);

    ratios={};
    for key in layers_to_explore:
        ratios[key]=[];


    dict_layers={};
    for idx_curr,layer_name in enumerate(layer_names):
        print idx_curr,layer_name,
        if layer_name in solver.net.params.keys():
            print len(solver.net.params[layer_name])
            update_prev=[np.zeros(solver.net.layers[idx_curr].blobs[0].diff.shape),
                        np.zeros(solver.net.layers[idx_curr].blobs[1].diff.shape)];
            blob_lr=list(blobs_lr[layers_to_explore.index(layer_name)]);
            dict_layers[idx_curr]=[layer_name,update_prev,blob_lr];
        else:
            print 0;

    for idx_curr in dict_layers.keys():
        print idx_curr,len(dict_layers[idx_curr]),dict_layers[idx_curr][0],dict_layers[idx_curr][1][0].shape,dict_layers[idx_curr][1][1].shape,dict_layers[idx_curr][2]

    
    for iteration in range(iterations):
        print iteration
    


        solver.net.forward();
        solver.net.backward();
        
        for idx_curr in dict_layers.keys():

            rel_row=dict_layers[idx_curr]
            layer_name=rel_row[0];
            update_prev=rel_row[1][0];
            print rel_row[2][0]
            lr_curr=rel_row[2][0]*lr;
            
            diffs_curr=solver.net.params[layer_name][0].diff;
            weights_curr=solver.net.params[layer_name][0].data;

            param_scale = np.linalg.norm(weights_curr.ravel())

            update = update_prev*momentum-lr_curr*diffs_curr;
            
            update_scale = np.linalg.norm(update.ravel())
            ratio= update_scale / param_scale # want ~1e-3
            print layer_name,ratio,update_scale,param_scale
            ratios[layer_name].append(ratio);
        
        for idx_curr,layer in enumerate(solver.net.layers):
            for idx_blob,blob in enumerate(layer.blobs):
                rel_row=dict_layers[idx_curr]
                layer_name=rel_row[0];
                update_prev=rel_row[1][idx_blob];
                lr_curr=rel_row[2][idx_blob]*lr;
                
                diffs_curr=blob.diff;
                update_curr=momentum*update_prev-(lr_curr*diffs_curr);
                blob.data[...] -= update_curr
                blob.diff[...] = np.zeros(blob.diff.shape);
                
                dict_layers[idx_curr][1][idx_blob]=update_curr;

    
    # print ratios
    # pickle.dump(ratios,open('/disk2/mayExperiments/ft_youtube_hmdb_newClusters_layerMagic/ratios_greaterConv.p','wb'));


def replaceSolverFile(out_file,template_file,deploy_file,base_lr,snapshot,snapshot_prefix,gpu=0):
    f=open(template_file,'rb');
    text=f.read()[:];
    f.close();
    

    text=text.replace('$DEPLOY_FILE','"'+deploy_file+'"');
    text=text.replace('$BASE_LR',str(base_lr));
    text=text.replace('$SNAPSHOT_PREFIX','"'+snapshot_prefix+'"');
    text=text.replace('$SNAPSHOT',str(snapshot));
    text=text.replace('$GPU',str(gpu));
    

    f=open(out_file,'wb')
    f.write(text);
    f.close();

def replaceDeployFile(out_file,template_file,train_file,fix_layers=None):
    f=open(template_file,'rb');
    text=f.read()[:];
    f.close();
    

    text=text.replace('$TRAIN_TXT','"'+train_file+'"');
    if fix_layers is not None:
        start_excludes=[];
        for fix_layer_curr in fix_layers:
            starts = [match.start() for match in re.finditer(re.escape('name: "'+fix_layer_curr), text)]
            assert len(starts)==1;
            # start_excludes=starts[:];
            start_excludes.append(starts[0]);
        starts=[match.start() for match in re.finditer(re.escape('name: '), text)]
        starts=[idx for idx in starts if idx not in start_excludes];
        starts.sort();
        starts=starts[::-1];
        # starts=starts[1:];
        for start in starts:
            string_orig=text[start:];   
            string_orig=string_orig[:string_orig.index('\n')]
            # [:string_orig.rindex('"')+1]
            # print string_orig
            string_new=string_orig[:string_orig.rindex('"')]+'_fix"';
            # print string_new,string_orig
            text=text.replace(string_orig,string_new);


    f=open(out_file,'wb')
    f.write(text);
    f.close();    

def printTrainingCommand(solver_file,log_file,initialize_model=None):
    command_pre='GLOG_logtostderr=1 /home/maheenrashid/Downloads/debugging_jacob/optical_flow_prediction/build/tools/caffe train'
    solver_path='-solver='+solver_file;
    if initialize_model is not None:
        model_file='-weights '+initialize_model;
    else:
        model_file='';
    log_part='2>&1 | tee '+log_file
    command_total=command_pre+' '+solver_path+' '+model_file+' '+log_part;
    print command_total;
    return command_total;


def getRatios(net_org,layers_to_explore):
    ratios_dict={}
    for layer_name in layers_to_explore:
        ratios_dict[layer_name]=[];
        diffs_curr=net_org.params[layer_name][0].diff;
        weights_curr=net_org.params[layer_name][0].data;
        param_scale = np.linalg.norm(weights_curr.ravel())
        update_scale = np.linalg.norm(diffs_curr.ravel())
        ratio= update_scale / param_scale # want ~1e-3
        print layer_name,ratio,update_scale,param_scale
        ratios_dict[layer_name]=[ratio,update_scale,param_scale]
    return ratios_dict;


def script_writeCommandsForExperiment():
    # out_dir='/disk3/maheen_data/debug_networks/noFixCopyByLayer';
    # model_file='/home/maheenrashid/Downloads/debugging_jacob/optical_flow_prediction/examples/opticalflow/final.caffemodel';

    out_dir='/disk3/maheen_data/debug_networks/noFixCopyByLayerAlexNet';
    model_file='/home/maheenrashid/Downloads/debugging_jacob/optical_flow_prediction/models/bvlc_alexnet/bvlc_alexnet.caffemodel';

    util.mkdir(out_dir);
    train_txt_orig_path='/disk3/maheen_data/debug_networks/noFix/train.txt';

    template_deploy_file='deploy_debug_noFix.prototxt';
    template_solver_file='solver_debug.prototxt';

    train_file=os.path.join(out_dir,'train.txt');
    
    shutil.copyfile(train_txt_orig_path,train_file);

    base_lr=0.0001;
    snapshot=100;
    layers=['conv1','conv2','conv3','conv4','conv5','fc6','fc7'];

    command_pre=os.path.join(out_dir,'debug_');
    commands=[];

    for idx in range(len(layers)):
        # if idx==0:
        #     fix_layers=layers[0];
        #     layer_str=str(fix_layers);
        #     model_file_curr=None;
        # else:
        fix_layers=layers[:idx+1];
    
        layer_str='_'.join(fix_layers);
        model_file_curr=model_file
        # print fix_layers

        if idx<len(layers)/2:
            gpu=0;
        else:
            gpu=1;


        snapshot_prefix=os.path.join(out_dir,'opt_noFix_'+layer_str+'_');
        out_deploy_file=os.path.join(out_dir,'deploy_'+layer_str+'.prototxt');
        out_solver_file=os.path.join(out_dir,'solver_'+layer_str+'.prototxt');
        log_file=os.path.join(out_dir,'log_'+layer_str+'.log');
        replaceSolverFile(out_solver_file,template_solver_file,out_deploy_file,base_lr,snapshot,snapshot_prefix,gpu);
        replaceDeployFile(out_deploy_file,template_deploy_file,train_file,fix_layers);
        command=printTrainingCommand(out_solver_file,log_file,model_file_curr);
        commands.append(command);
    
    command_file_1=command_pre+'0.sh';
    util.writeFile(command_file_1,commands[:len(commands)/2]);
    command_file_2=command_pre+'1.sh';
    util.writeFile(command_file_2,commands[len(commands)/2:]);

def script_visualizeLossesFromExperiment():

    # out_dir='/disk3/maheen_data/ft_youtube_40_noFix_alexnet';
    out_dir='/disk3/maheen_data/debug_networks/noFixCopyByLayerAlexNet';
    model_file='/home/maheenrashid/Downloads/debugging_jacob/optical_flow_prediction/models/bvlc_alexnet/bvlc_alexnet.caffemodel';
    layers=['conv1','conv2','conv3','conv4','conv5','fc6'];
    # ,'fc7'];

    layers_str=[];
    for idx in range(len(layers)):
        # if idx==0:
        #     fix_layers=layers[0];
        #     layer_str=str(fix_layers);
        # else:
        fix_layers=layers[:idx+1];
        layer_str='_'.join(fix_layers);
        layers_str.append(layer_str);

    log_files=[os.path.join(out_dir,'log_'+layer_str+'.log') for layer_str in layers_str];
    str_match=' solver.cpp:209] Iteration ';
    xAndYs=[svl.getIterationsAndLosses(log_file,str_match) for log_file in log_files];

    out_files=[];
    for layer_str,log_file in zip(layers_str,log_files):
        xAndY=svl.getIterationsAndLosses(log_file,str_match);
        print xAndY
        out_file=os.path.join(out_dir,'loss_'+layer_str+'.png');
        visualize.plotSimple([xAndY],out_file,title=layer_str);
        out_files.append(out_file);

    out_file_html=os.path.join(out_dir,'losses_all.html');
    img_paths=[[util.getRelPath(out_file,'/disk3')] for out_file in out_files];
    captions=[['']]*len(out_files);
    print img_paths
    print captions
    visualize.writeHTML(out_file_html,img_paths,captions,height=300,width=300);


def justCheckGradients(solver_file,deploy_file,model_file):
    model=model_file
    print model
    print os.path.exists(model);

    caffe.set_device(1)
    caffe.set_mode_gpu()

    solver=caffe.SGDSolver(solver_file);
    solver.net.forward();

    net=caffe.Net(deploy_file,model);

    print list(net._layer_names);
    print net.blobs.keys();
    # return
    net.blobs['data'].data[...]=solver.net.blobs['data'].data;
    net.blobs['thelabelscoarse'].data[...]=solver.net.blobs['thelabelscoarse'].data;

    net.forward();
    # print net.blobs['thelabelscoarse'].data[:10,0,0,0,0];
    # print net.blobs['reshapefc8'].data[0,39,0,:]
    net.backward();
    # print net.blobs.keys();

    layers_to_explore=['conv1','conv2','conv3','conv4','conv5','fc6_fix','fc7_fix','fc8_fix']
    ratios=getRatios(net,layers_to_explore);

    for layer_name in ratios.keys():
        print layer_name,ratios[layer_name];


def main():
    # out_dir='/disk3/maheen_data/ft_youtube_40_noFix_diffLR_sZclusters';
    # model_file='/home/maheenrashid/Downloads/debugging_jacob/optical_flow_prediction/examples/opticalflow/final.caffemodel';

    # util.mkdir(out_dir);

    # train_file=os.path.join(out_dir,'train.txt');

    # template_deploy_file='trainval_noFix_withRandom_diffForConv.prototxt';
    # template_solver_file='solver_debug.prototxt';

    # base_lr=0.000001;
    # snapshot=1000;
    # layers=[None,'conv1','conv2','conv3','conv4','conv5','fc6','fc7','fc8'];
    # gpu=1;
    # commands=[];
    # idx=len(layers)-4;
    # fix_layers=layers[1:idx+1];

    # layer_str='_'.join(fix_layers);
    # print layer_str;
    # # return
    # model_file_curr=model_file
    # snapshot_prefix=os.path.join(out_dir,'opt_noFix_'+layer_str+'_');
    # out_deploy_file=os.path.join(out_dir,'deploy_'+layer_str+'.prototxt');
    # out_solver_file=os.path.join(out_dir,'solver_'+layer_str+'.prototxt');
    # log_file=os.path.join(out_dir,'log_'+layer_str+'.log');
    # replaceSolverFile(out_solver_file,template_solver_file,out_deploy_file,base_lr,snapshot,snapshot_prefix,gpu=gpu);
    # replaceDeployFile(out_deploy_file,template_deploy_file,train_file,fix_layers);
    # command=printTrainingCommand(out_solver_file,log_file,model_file_curr);
    # util.writeFile(os.path.join(out_dir,'train.sh'),[command]);


    # return
    model_file='/disk3/maheen_data/ft_youtube_40_images_cluster_suppress_yjConfig/opt_noFix_conv1_conv2_conv3_conv4_conv5_llr__iter_50000.caffemodel'
    # model_file='/home/maheenrashid/Downloads/debugging_jacob/optical_flow_prediction/examples/opticalflow/final.caffemodel';
    solver_file='/disk3/maheen_data/ft_youtube_40_images_cluster_suppress_yjConfig/solver_conv1_conv2_conv3_conv4_conv5.prototxt';
    deploy_file='/disk3/maheen_data/ft_youtube_40_images_cluster_suppress_yjConfig/deploy_conv1_conv2_conv3_conv4_conv5.prototxt';

    model_file='/home/maheenrashid/Downloads/debugging_jacob/optical_flow_prediction/examples/opticalflow/final.caffemodel';
    solver_file='/disk3/maheen_data/ft_youtube_40_noFix_diffLR_sZclusters/solver_conv1_conv2_conv3_conv4_conv5.prototxt'
    deploy_file='/disk3/maheen_data/ft_youtube_40_noFix_diffLR_sZclusters/deploy_conv1_conv2_conv3_conv4_conv5.prototxt';    
    justCheckGradients(solver_file,deploy_file,model_file);


    return
    out_dir='/disk3/maheen_data/debug_networks/sanityCheckDebug';
    model_file='/home/maheenrashid/Downloads/debugging_jacob/optical_flow_prediction/examples/opticalflow/final.caffemodel';

    util.mkdir(out_dir);

    train_file=os.path.join(out_dir,'train.txt');

    template_deploy_file='deploy_withRandom.prototxt';
    template_solver_file='solver_debug.prototxt';

    base_lr=0.000001;
    snapshot=1000;
    layers=[None,'conv1','conv2','conv3','conv4','conv5','fc6','fc7','fc8'];
    gpu=1;
    commands=[];
    idx=len(layers)-1;
    fix_layers=layers[1:idx+1];

    layer_str='_'.join(fix_layers);
    print layer_str;
    
    model_file_curr=model_file
    snapshot_prefix=os.path.join(out_dir,'opt_noFix_'+layer_str+'_');
    out_deploy_file=os.path.join(out_dir,'deploy_'+layer_str+'.prototxt');
    out_solver_file=os.path.join(out_dir,'solver_'+layer_str+'.prototxt');
    log_file=os.path.join(out_dir,'log_'+layer_str+'.log');
    replaceSolverFile(out_solver_file,template_solver_file,out_deploy_file,base_lr,snapshot,snapshot_prefix,gpu=gpu);
    replaceDeployFile(out_deploy_file,template_deploy_file,train_file,fix_layers);
    command=printTrainingCommand(out_solver_file,log_file,model_file_curr);
    util.writeFile(os.path.join(out_dir,'train.sh'),[command]);


    return

    out_dir='/disk3/maheen_data/ft_youtube_40_images_cluster_suppress_yjConfig/';
    out_dir='/disk3/maheen_data/ft_youtube_40_images_cluster_suppress_yjConfig_llr_diff/';
    model_file='/home/maheenrashid/Downloads/debugging_jacob/optical_flow_prediction/examples/opticalflow/final.caffemodel';

    util.mkdir(out_dir);

    train_file=os.path.join(out_dir,'train.txt');

    template_deploy_file='deploy_withRandom_yjConfig.prototxt';
    template_solver_file='solver_debug.prototxt';

    base_lr=0.00001;
    snapshot=500;
    layers=[None,'conv1','conv2','conv3','conv4','conv5','fc6','fc7','fc8'];
    gpu=1;
    commands=[];
    idx=len(layers)-4;
    fix_layers=layers[1:idx+1];

    layer_str='_'.join(fix_layers);
    print layer_str;
    # return
    model_file_curr=model_file
    snapshot_prefix=os.path.join(out_dir,'opt_noFix_'+layer_str+'_');
    out_deploy_file=os.path.join(out_dir,'deploy_'+layer_str+'.prototxt');
    out_solver_file=os.path.join(out_dir,'solver_'+layer_str+'.prototxt');
    log_file=os.path.join(out_dir,'log_'+layer_str+'.log');
    replaceSolverFile(out_solver_file,template_solver_file,out_deploy_file,base_lr,snapshot,snapshot_prefix,gpu=gpu);
    replaceDeployFile(out_deploy_file,template_deploy_file,train_file,fix_layers);
    command=printTrainingCommand(out_solver_file,log_file,model_file_curr);
    util.writeFile(os.path.join(out_dir,'train.sh'),[command]);



    return
    out_dir='/disk3/maheen_data/ft_youtube_40_noFix_noCopyFC8_FC7';
    model_file='/home/maheenrashid/Downloads/debugging_jacob/optical_flow_prediction/examples/opticalflow/final.caffemodel';

    # out_dir='/disk3/maheen_data/ft_youtube_40_noFix_alexnet';
    # model_file='/home/maheenrashid/Downloads/debugging_jacob/optical_flow_prediction/models/bvlc_alexnet/bvlc_alexnet.caffemodel';

    util.mkdir(out_dir);
    train_txt_orig_path='/disk3/maheen_data/ft_youtube_40/train.txt';

    
    template_deploy_file='deploy_withRandom.prototxt';
    template_solver_file='solver_debug.prototxt';

    train_file=os.path.join(out_dir,'train.txt');
    
    data=util.readLinesFromFile(train_txt_orig_path);
    random.shuffle(data);
    # data[:100];
    util.writeFile(train_file,data);


    # shutil.copyfile(train_txt_orig_path,train_file);



    # out_dir='/disk3/maheen_data/ft_youtube_40_ucf_permute';
    # train_file=os.path.join(out_dir,'train_permute.txt');

    

    base_lr=0.0001;
    snapshot=2000;
    layers=[None,'conv1','conv2','conv3','conv4','conv5','fc6','fc7','fc8'];
    gpu=0;
    # command_file=os.path.join(out_dir,'debug_0.sh');
    commands=[];

    # for idx in range(4,len(layers)):
    #     if idx==0:
    #         fix_layers=layers[0];
    #         layer_str=str(fix_layers);
    #         model_file_curr=None;
    #     else:

    idx=len(layers)-3;
    fix_layers=layers[1:idx+1];

    layer_str='_'.join(fix_layers);
    print layer_str;

    return
    model_file_curr=model_file
    # print fix_layers
    snapshot_prefix=os.path.join(out_dir,'opt_noFix_'+layer_str+'_');
    out_deploy_file=os.path.join(out_dir,'deploy_'+layer_str+'.prototxt');
    out_solver_file=os.path.join(out_dir,'solver_'+layer_str+'.prototxt');
    log_file=os.path.join(out_dir,'log_'+layer_str+'.log');
    replaceSolverFile(out_solver_file,template_solver_file,out_deploy_file,base_lr,snapshot,snapshot_prefix,gpu=gpu);
    replaceDeployFile(out_deploy_file,template_deploy_file,train_file,fix_layers);
    command=printTrainingCommand(out_solver_file,log_file,model_file_curr);
    util.writeFile(os.path.join(out_dir,'train.sh'),[command]);

    # commands.append(command);

    
    # util.writeFile(command_file,commands);





    return
    # out_dir='/disk3/maheen_data/debug_networks/noFix';
    # model_file='/home/maheenrashid/Downloads/debugging_jacob/optical_flow_prediction/examples/opticalflow/final.caffemodel';
    # '/disk2/mayExperiments/ft_youtube_hmdb_newClusters_layerMagic/train.txt'
    # data=util.readLinesFromFile(train_txt_orig_path);
    # random.shuffle(data);
    # # data[:100];
    # util.writeFile(train_file,data[:100]);

    # out_dir='/disk3/maheen_data/debug_networks/noFixNoCopy';
    # model_file=None;


    out_dir='/disk3/maheen_data/debug_networks/noFixCopyByLayer';
    model_file='/home/maheenrashid/Downloads/debugging_jacob/optical_flow_prediction/examples/opticalflow/final.caffemodel';


    util.mkdir(out_dir);
    train_txt_orig_path='/disk3/maheen_data/debug_networks/noFix/train.txt';

    deploy_file='/disk3/maheen_data/debug_networks/noFix/deploy.prototxt';
    solver_file='/disk3/maheen_data/debug_networks/noFix/solver.prototxt';

    # template_deploy_file='deploy_debug_noFix.prototxt';
    template_deploy_file='deploy_fc8NoCopy.prototxt';
    template_solver_file='solver_debug.prototxt';

    train_file=os.path.join(out_dir,'train.txt');
    

    # shutil.copyfile(train_txt_orig_path,train_file);


    

    base_lr=0.0001;
    snapshot=100;
    layers=[None,'conv1','conv2','conv3','conv4','conv5','fc6','fc7','fc8'];









    return
    out_dir='/disk3/maheen_data/ft_youtube_40_noFix_alexnet';
    model_file='/home/maheenrashid/Downloads/debugging_jacob/optical_flow_prediction/models/bvlc_alexnet/bvlc_alexnet.caffemodel';
    layers=['conv1','conv2','conv3','conv4','conv5','fc6','fc7'];

    layers_str=[];
    for idx in range(len(layers)):
        if idx==0:
            fix_layers=layers[0];
            layer_str=str(fix_layers);
        else:
            fix_layers=layers[1:idx+1];
            layer_str='_'.join(fix_layers);
        layers_str.append(layer_str);

    log_files=[os.path.join(out_dir,'log_'+layer_str+'.log') for layer_str in layers_str];
    str_match=' solver.cpp:209] Iteration ';
    xAndYs=[svl.getIterationsAndLosses(log_file,str_match) for log_file in log_files];

    out_files=[];
    for layer_str,log_file in zip(layers_str,log_files):
        xAndY=svl.getIterationsAndLosses(log_file,str_match);
        print xAndY
        out_file=os.path.join(out_dir,'loss_'+layer_str+'.png');
        visualize.plotSimple([xAndY],out_file,title=layer_str);
        out_files.append(out_file);

    out_file_html=os.path.join(out_dir,'losses_all.html');
    img_paths=[[util.getRelPath(out_file,'/disk3')] for out_file in out_files];
    captions=[['']]*len(out_files);
    print img_paths
    print captions
    visualize.writeHTML(out_file_html,img_paths,captions,height=300,width=300);

    # out_file=os.path.join(out_dir,'losses_all.png');

    # print len(xAndYs);
    # print xAndYs[-2][1]

    # visualize.plotSimple(xAndYs,out_file,legend_entries=layers_str,loc=0,outside=True)

    


        

        







    


    return
    # mean_standard_proto_file='/home/maheenrashid/Downloads/debugging_jacob/opticalflow/standard.binaryproto';
    model_file='/home/maheenrashid/Downloads/debugging_jacob/optical_flow_prediction/examples/opticalflow/final.caffemodel';
    layers_to_copy=['conv1','conv2','conv3','conv4','conv5']

    # model_file='/disk2/mayExperiments/ft_youtube_hmdb_newClusters_layerMagic/OptFlow_youtube_hmdb__iter_5000.caffemodel';
    # layers_to_copy=['conv1','conv2','conv3','conv4','conv5','fc6_fix','fc7_fix','fc8_fix']

    # model_file='/disk2/mayExperiments/ft_youtube_hmdb_newClusters_layerMagic_llr/OptFlow_youtube_hmdb__iter_65000.caffemodel';
    # layers_to_copy=['conv1','conv2','conv3','conv4','conv5','fc6_fix','fc7_fix','fc8_fix']


    # deploy_file='/disk2/mayExperiments/ft_youtube_hmdb_newClusters_layerMagic/opt_train_coarse_xavier.prototxt';
    # solver_file='/disk2/mayExperiments/ft_youtube_hmdb_newClusters_layerMagic/train.prototxt';
    deploy_file='/disk2/mayExperiments/ft_youtube_hmdb_newClusters_layerMagic/deploy_debug.prototxt';
    solver_file='/disk2/mayExperiments/ft_youtube_hmdb_newClusters_layerMagic/solver_debug.prototxt';

    # layers_to_copy=['conv1','conv2','conv3','conv4','conv5','fc6_fix','fc7_fix','fc8_fix']
    # layers_to_explore=['conv1','conv2','conv3','conv4','conv5','fc6_fix','fc7_fix','fc8_fix']

    
    # ,'fc6','fc7','fc8']
    layers_to_explore=['conv1','conv2','conv3','conv4','conv5','fc6_fix','fc7_fix','fc8_fix']
    blobs_lr=[(0,0),(0,0),(0,0),
            # (10,20),(10,20),
            (0.1,0.2),(0.1,0.2),
            (1,2),(1,2),(1,2)]

    iterations=400;
    momentum=0.9;
    lr=0.000001;

    caffe.set_device(1)
    caffe.set_mode_gpu()



    solver=caffe.SGDSolver(solver_file);

    
    net_org=caffe.Net(deploy_file,model_file);
        
    # copy weights
    for layer_name in layers_to_copy:
        solver.net.params[layer_name][0].data[...]=net_org.params[layer_name][0].data;
        solver.net.params[layer_name][1].data[...]=net_org.params[layer_name][1].data;

    layer_names=list(solver.net._layer_names);

    ratios={};
    for key in layers_to_explore:
        ratios[key]=[];


    dict_layers={};
    for idx_curr,layer_name in enumerate(layer_names):
        print idx_curr,layer_name,
        if layer_name in solver.net.params.keys():
            print len(solver.net.params[layer_name])
            update_prev=[np.zeros(solver.net.layers[idx_curr].blobs[0].diff.shape),
                        np.zeros(solver.net.layers[idx_curr].blobs[1].diff.shape)];
            blob_lr=list(blobs_lr[layers_to_explore.index(layer_name)]);
            dict_layers[idx_curr]=[layer_name,update_prev,blob_lr];
        else:
            print 0;

    for idx_curr in dict_layers.keys():
        print idx_curr,len(dict_layers[idx_curr]),dict_layers[idx_curr][0],dict_layers[idx_curr][1][0].shape,dict_layers[idx_curr][1][1].shape,dict_layers[idx_curr][2]

    
    for iteration in range(iterations):
        print iteration
    


        solver.net.forward();
        solver.net.backward();
        
        for idx_curr in dict_layers.keys():

            rel_row=dict_layers[idx_curr]
            layer_name=rel_row[0];
            update_prev=rel_row[1][0];
            print rel_row[2][0]
            lr_curr=rel_row[2][0]*lr;
            
            diffs_curr=solver.net.params[layer_name][0].diff;
            weights_curr=solver.net.params[layer_name][0].data;

            param_scale = np.linalg.norm(weights_curr.ravel())

            update = update_prev*momentum-lr_curr*diffs_curr;
            
            update_scale = np.linalg.norm(update.ravel())
            ratio= update_scale / param_scale # want ~1e-3
            print layer_name,ratio,update_scale,param_scale
            ratios[layer_name].append(ratio);
        
        for idx_curr,layer in enumerate(solver.net.layers):
            for idx_blob,blob in enumerate(layer.blobs):
                rel_row=dict_layers[idx_curr]
                layer_name=rel_row[0];
                update_prev=rel_row[1][idx_blob];
                lr_curr=rel_row[2][idx_blob]*lr;
                
                diffs_curr=blob.diff;
                update_curr=momentum*update_prev-(lr_curr*diffs_curr);
                blob.data[...] -= update_curr
                blob.diff[...] = np.zeros(blob.diff.shape);
                
                dict_layers[idx_curr][1][idx_blob]=update_curr;

    
    # print ratios
    # pickle.dump(ratios,open('/disk2/mayExperiments/ft_youtube_hmdb_newClusters_layerMagic/ratios_greaterConv.p','wb'));




if __name__=='__main__':
    main();