import tensorflow as tf
import numpy as np 

ifmap=np.zeros((1,56,56,128))
weight=np.zeros((3,3,128,128))
print(weight.shape)
for i in range(56):
    for j in range(56):
        for k in range(128):
            ifmap[0][i][j][k]=i/100

for i in range(3):
    for j in range(3):
        for k in range(128):
            for l in range(128):
                weight[i][j][k][l]=i/100
#weight=np.fliplr(weight)
#weight=np.flipud(weight)



ifmp=tf.convert_to_tensor(ifmap, dtype=tf.float32)
weight=tf.convert_to_tensor(weight, dtype=tf.float32)

ofmap=tf.nn.conv2d(ifmap,weight,1,'SAME')
print(ofmap[0,:,:,12])
