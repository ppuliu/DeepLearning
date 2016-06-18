import numpy as np

def read_spike_train(file_name):

    spike_times=[]
    max_time=0
    with open(file_name) as f:
        for line in f:
            entries=map(int, line.split(','))
            print len(entries)
            print entries[-1]
            if entries[-1]>max_time:
                max_time=entries[-1]
            spike_times.append(entries)
    n=len(spike_times)
    spike_train=np.zeros([max_time+1,n])

    for i in xrange(n):
        for j in spike_times[i]:
            spike_train[j,i]=1

    return spike_train


