import numpy as np
import sys

def read_spike_train(file_name, time_bin = 1, multiplier = 1):

    spike_times=[]
    max_time=0
    min_time=sys.maxint / time_bin
    with open(file_name) as f:
        for line in f:
            if len(line.strip()) == 0:
                continue
            entries=map(int,map(lambda x: multiplier * x, map(float, line.split(','))))
            # print len(entries)
            # print entries[-1]
            if entries[0]<min_time * time_bin:
                min_time=entries[0] / time_bin
            if entries[-1]>max_time * time_bin:
                max_time=entries[-1] / time_bin
            spike_times.append(entries)
    n=len(spike_times)
    spike_train=np.zeros([max_time - min_time +1,n])

    for i in xrange(n):
        for j in spike_times[i]:
            spike_train[j / time_bin - min_time,i]=1

    return spike_train


