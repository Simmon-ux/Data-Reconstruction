import argparse
import numpy as np
import random
from PIL import Image
import matplotlib.pyplot as plt
import h5py
#--------------------------------------------------
def random_walk(img_size):
    canvas = np.zeros([img_size, img_size])
    for i in range(img_size):
        for j in range(img_size):
            a = random.randint(0,99)
            if (i >=17) & (i <= 50) & (j >=17) & (j <=50):
                if a >= 98:
                    canvas[i,j] = 1
            else:
                if a >= 86:
                    canvas[i, j] = 1
    return canvas
f1 = h5py.File('D:\py\AI\mask\\train_random_mask_68x68_30000.h5',mode='w')
tas = np.zeros([30000, 68, 68])
for i in range(30000):
    a = random_walk(68)
    tas[i,:,:] = a
f1.create_dataset('tas',data = tas,dtype='f4')
f1.close()

'''y = 0
for k in range(100):
    a = random_walk(68)
    a_0 = 0
    for i in range(68):
        for j in range(68):
            if a[i,j] == 0:
                a_0 += 1
    y += a_0/(68*68) * 100
print(y/100)'''

'''data = h5py.File('D:\py\AI\data\\test_large\ease_daily-t2m-Arctic-np-buoy-ghcn-40N-2011-2020-anomaly-6868.h5','r').get('tas')
y = 0
for k in range(0,365,1):
    data1 = data[k,:,:]
    a_0 = 0
    for i in range(17,50):
        for j in range(17,50):
            if data1[i, j] == 0:
                a_0 += 1
    y += a_0 / (33 * 33) * 100
    print(k)
print(y/365)'''
'''a = random_walk(68)
fig = plt.figure(figsize=(9,7))
ax1 = fig.add_subplot()
b = ax1.contourf(range(68),range(68),a,levels=np.arange(-1,2,1))
fig.colorbar(b)
plt.show()
'''
