from numpy import *
from matplotlib.pyplot import *
from scipy.signal import filtfilt, butter

result = load('IU.npy')
intersection = result[..., 0]
union = result[..., 1]
union[union == 0] = nan
IU = intersection / union
#IU[IU == 0] = nan

mean_IU = nanmean(IU, axis=0)

fig = figure(1, figsize=(4, 3))
clf()
ax = fig.add_subplot(1, 1, 1)
bar(arange(result.shape[1]), mean_IU)
ylim(0, 1)
xlabel('Class Labels')
ylabel('Mean IU')
tight_layout()
savefig('bar_IU.png', dpi=300)

# find good and bad results
rank = nanmean(IU, axis=1).argsort()
N = 5
for k in range(N):
    img = imread('./results/%04d.png' % rank[k])
    imsave('bad%d.png' % k, img)

for k in range(N):
    img = imread('./results/%04d.png' % rank[-(k + 1)])
    imsave('good%d.png' % k, img)

##
figure(2, figsize=(4, 3))
clf()
b, a = butter(1, 0.01)

data = loadtxt('./run_.,tag_loss.csv', delimiter=',', skiprows=1, usecols=[1, 2])
step = data[:, 0]
loss = data[:, 1]
loss_lpf = filtfilt(b, a, loss)
plot(step, loss, 'b', alpha=0.2)
plot(step, loss_lpf, 'b', label='training loss')

data = loadtxt('./run_.,tag_loss_val.csv', delimiter=',', skiprows=1, usecols=[1, 2])
step = data[:, 0]
loss = data[:, 1]
loss_lpf = filtfilt(b, a, loss)
plot(step, loss, 'r', alpha=0.2)
plot(step, loss_lpf, 'r', label='validation loss')

xlabel('iterations')
legend()
tight_layout()
savefig('loss.png', dpi=300)
