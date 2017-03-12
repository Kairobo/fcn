from numpy import *
from matplotlib.pyplot import *

result = load('IU.npy')
IU = result[..., 0] / result[..., 1]
#IU[IU == 0] = nan

mean_IU = nanmean(IU, axis=0)


fig = figure(1)
clf()
ax = fig.add_subplot(1, 1, 1)
bar(arange(result.shape[1]), mean_IU)
ylim(0, 1)
