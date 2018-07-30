import numpy as np
from skimage.io import imread
from matplotlib import pyplot as plt
from generator import train_generator
from pympler import summary, muppy


size = (512, 512)

im = plt.imshow(np.zeros((size[0] * 2, size[1] * 2)))
plt.ion()
plt.show()

train_gen = train_generator(2, (['input'], ['output']), size, 2, dump_mem=True)

ctr = 0

for i, o in train_gen:
    o = o[:, :, :, :1]
    i = i[:, :, :, :3]
    o = np.concatenate((o, o, o), axis=3)
    i1 = np.hstack((i[0], o[0]))
    i2 = np.hstack((i[1], o[1]))
    im.set_data(np.vstack((i1, i2)))
    plt.draw()
    plt.pause(2)
    ctr += 1
    #if ctr > 100:
    #    ctr = 0
    #    sum1 = summary.summarize(muppy.get_objects())
    #    summary.print_(sum1)
