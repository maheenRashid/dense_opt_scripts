import util;
import os;
import visualize;
# import matplotlib.pyplot as plt;

def getIterationsAndLosses(log_file,str_match,iter_str='Iteration ',score_str='loss = '):

    lines=util.readLinesFromFile(log_file);
    iterations=[];
    losses=[];

    for line in lines:
        if str_match in line:
            idx=line.index(iter_str)
            iter_no=line[idx+len(iter_str):line.rindex(',')];
            iter_no=int(iter_no);
            loss=line[line.index(score_str)+len(score_str):];
            loss=float(loss);
            iterations.append(iter_no);
            losses.append(loss);
    return iterations,losses;

def main():
    # file_curr='log.log';
    dir_curr='/disk2/marchExperiments/network_100_5'
    # /log_2.log
    file_curr=os.path.join(dir_curr,'log_stepsizechanged.log');
    file_old=os.path.join(dir_curr,'log.log');
    out_file=os.path.join(dir_curr,'loss_graph.png');

    str_match=' solver.cpp:209] Iteration ';
    
    # lines=util.readLinesFromFile(file_curr);
    iterations,losses=getIterationsAndLosses(file_curr,str_match);    
    iterations_old,losses_old=getIterationsAndLosses(file_old,str_match);    

    idx=iterations_old.index(880)
    print idx;
    print iterations[0];
    iterations_old=iterations_old[:idx];
    losses_old=losses_old[:idx];
    iterations_old.extend(iterations);
    losses_old.extend(losses);
    iterations=iterations_old;
    losses=losses_old;

    title='Iterations vs Loss at '+str(iterations[-1]);
    # file_curr='log_1.log';
    # # lines=util.readLinesFromFile(file_curr);
    # iters,loss=getIterationsAndLosses(file_curr,str_match);

    # # iters=[a+iterations[-1] for a in iters];
    # iterations.extend(iters);
    # losses.extend(loss);
    # plotSimple(xAndYs,out_file,title='',xlabel='',ylabel='',legend_entries=None,loc=0,outside=False)
    visualize.plotSimple([(iterations,losses)],out_file,xlabel='Iterations',ylabel='Loss',title=title);
    # plt.ion();
    # plt.figure();
    # plt.plot(iterations,losses);
    # # plt.title(title);
    # plt.xlabel('Iterations');
    # plt.ylabel('Loss');
    # plt.show();
    # raw_input();


if __name__=='__main__':
    main();