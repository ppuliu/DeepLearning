import matplotlib.pylab as plt
import numpy as np

def visualSeries(X):

    if len(X.shape)==1:
        X=np.expand_dims(X, axis=1)
    num_steps, num_ch = X.shape
    Y=np.zeros([num_steps,num_ch])

    # figure
    # plot raw time series
    for i in xrange(num_ch):
        Y[:,i] = X[:,i]*1+(1*i)

    plt.plot(Y)
    plt.show()

def overlayPredicts(targetX, predictX):

    if len(targetX.shape) == 1:
        targetX = np.expand_dims(targetX, axis=1)
    num_steps, num_ch = targetX.shape

    predictY = np.zeros(predictX.shape)

    for i in xrange(num_ch):
        predictY[:, i] = predictX[:, i] * 1 + (1 * i)

    # plot target spikes
    (txPoints, tyPoint) = np.where(targetX == 1)

    xys = zip(txPoints, tyPoint)
    for i in xrange(len(xys)):
        xy = xys[i]
        plt.plot([xy[0], xy[0]], [xy[1], xy[1] + 1], 'r-')

    # plot predicted time series

    plt.plot(predictY, color='k');
    plt.show()
