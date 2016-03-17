FOR getTrainingCommand
    dir_network='/disk2/marchExperiments/network_box'
    path_train='/home/maheenrashid/Downloads/debugging_jacob/optical_flow_prediction/build/tools/caffe';
    path_solver=os.path.join(dir_network,'train.prototxt');
    path_weights='/home/maheenrashid/Downloads/debugging_jacob/optical_flow_prediction/models/bvlc_alexnet/bvlc_alexnet.caffemodel';
    path_log=os.path.join(dir_network,'log.log');

    command=getTrainingCommand(path_train,path_solver,path_weights=path_weights,path_log=path_log)
    print command


FOR writeTrainTxt
	dir_meta='/disk2/marchExperiments/ucf-101-new';
    out_dir_network='/disk2/marchExperiments/network_ucf';
    util.mkdir(out_dir_network);
    out_file_train=os.path.join(out_dir_network,'train.txt');
    
    video_dirs=[os.path.join(dir_meta,vid_dir) for vid_dir in os.listdir(dir_meta) if os.path.isdir(os.path.join(dir_meta,vid_dir))];
    writeTrainTxt(out_file_train,video_dirs,im_dir,tif_dir)