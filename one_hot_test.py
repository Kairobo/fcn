from glob import glob
from matplotlib.pyplot import *
from numpy import *

fn = glob('./data/train/labels/*.npy')
random.shuffle(fn)

print(fn[0])
img = load(fn[0])

fig = figure(1, figsize=(16, 12))
clf()
for k in range(20):
    ax = fig.add_subplot(4, 5, k + 1)
    ax.imshow(img[..., k], clim=[0, 1])
    ax.set_xticks([])
    ax.set_yticks([])

tight_layout()
show()
