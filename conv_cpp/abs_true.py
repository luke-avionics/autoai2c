import tensorflow as tf
import numpy as np 

ifmap=np.zeros((1,12,12,2))
weight=np.zeros((3,3,2,2))
print(weight.shape)
for i in range(12):
    for j in range(12):
        for k in range(2):
            ifmap[0][i][j][k]=i

for i in range(3):
    for j in range(3):
        for k in range(2):
            for l in range(2):
                weight[i][j][k][l]=i
#weight=np.fliplr(weight)
#weight=np.flipud(weight)



ifmp=tf.convert_to_tensor(ifmap, dtype=tf.float32)
weight=tf.convert_to_tensor(weight, dtype=tf.float32)

ofmap=tf.nn.conv2d(ifmap,weight,1,'SAME')
print(ofmap[0,:,:,0])
