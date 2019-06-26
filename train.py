'''
One part of testing
'''
import numpy as np
import os 
import argparse
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import random as ran
import skimage
from skimage import io
from random import shuffle
import math
import tensorflow as tf
import tensorflow.contrib.layers as tcl
from tensorflow.contrib.framework import arg_scope
import numpy as np
def resBlock(x, num_outputs, kernel_size = 4, stride=1, activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm, scope=None):
    assert num_outputs%2==0 #num_outputs must be divided by channel_factor(2 here)
    with tf.variable_scope(scope, 'resBlock'):
        shortcut = x
        if stride != 1 or x.get_shape()[3] != num_outputs:
            shortcut = tcl.conv2d(shortcut, num_outputs, kernel_size=1, stride=stride, 
                        activation_fn=None, normalizer_fn=None, scope='shortcut')
        x = tcl.conv2d(x, num_outputs/2, kernel_size=1, stride=1, padding='SAME')
        x = tcl.conv2d(x, num_outputs/2, kernel_size=kernel_size, stride=stride, padding='SAME')
        x = tcl.conv2d(x, num_outputs, kernel_size=1, stride=1, activation_fn=None, padding='SAME', normalizer_fn=None)

        x += shortcut       
        x = normalizer_fn(x)
        x = activation_fn(x)
    return x


class resfcn256(object):
    def __init__(self, resolution_inp = 256, resolution_op = 256, channel = 3, name = 'resfcn256'):
        self.name = name
        self.channel = channel
        self.resolution_inp = resolution_inp
        self.resolution_op = resolution_op

    def __call__(self, x, is_training = True):
        with tf.variable_scope(self.name) as scope:
            with arg_scope([tcl.batch_norm], is_training=is_training, scale=True):
                with arg_scope([tcl.conv2d, tcl.conv2d_transpose], activation_fn=tf.nn.relu, 
                                     normalizer_fn=tcl.batch_norm, 
                                     biases_initializer=None, 
                                     padding='SAME',
                                     weights_regularizer=tcl.l2_regularizer(0.0002)):
                    size = 16  
                    # x: s x s x 3
                    se = tcl.conv2d(x, num_outputs=size, kernel_size=4, stride=1) # 256 x 256 x 16
                    se = resBlock(se, num_outputs=size * 2, kernel_size=4, stride=2) # 128 x 128 x 32
                    se = resBlock(se, num_outputs=size * 2, kernel_size=4, stride=1) # 128 x 128 x 32
                    se = resBlock(se, num_outputs=size * 4, kernel_size=4, stride=2) # 64 x 64 x 64
                    se = resBlock(se, num_outputs=size * 4, kernel_size=4, stride=1) # 64 x 64 x 64
                    se = resBlock(se, num_outputs=size * 8, kernel_size=4, stride=2) # 32 x 32 x 128
                    se = resBlock(se, num_outputs=size * 8, kernel_size=4, stride=1) # 32 x 32 x 128
                    se = resBlock(se, num_outputs=size * 16, kernel_size=4, stride=2) # 16 x 16 x 256
                    se = resBlock(se, num_outputs=size * 16, kernel_size=4, stride=1) # 16 x 16 x 256
                    se = resBlock(se, num_outputs=size * 32, kernel_size=4, stride=2) # 8 x 8 x 512
                    se = resBlock(se, num_outputs=size * 32, kernel_size=4, stride=1) # 8 x 8 x 512

                    pd = tcl.conv2d_transpose(se, size * 32, 4, stride=1) # 8 x 8 x 512 
                    pd = tcl.conv2d_transpose(pd, size * 16, 4, stride=2) # 16 x 16 x 256 
                    pd = tcl.conv2d_transpose(pd, size * 16, 4, stride=1) # 16 x 16 x 256 
                    pd = tcl.conv2d_transpose(pd, size * 16, 4, stride=1) # 16 x 16 x 256 
                    pd = tcl.conv2d_transpose(pd, size * 8, 4, stride=2) # 32 x 32 x 128 
                    pd = tcl.conv2d_transpose(pd, size * 8, 4, stride=1) # 32 x 32 x 128 
                    pd = tcl.conv2d_transpose(pd, size * 8, 4, stride=1) # 32 x 32 x 128 
                    pd = tcl.conv2d_transpose(pd, size * 4, 4, stride=2) # 64 x 64 x 64 
                    pd = tcl.conv2d_transpose(pd, size * 4, 4, stride=1) # 64 x 64 x 64 
                    pd = tcl.conv2d_transpose(pd, size * 4, 4, stride=1) # 64 x 64 x 64 
                    
                    pd = tcl.conv2d_transpose(pd, size * 2, 4, stride=2) # 128 x 128 x 32
                    pd = tcl.conv2d_transpose(pd, size * 2, 4, stride=1) # 128 x 128 x 32
                    pd = tcl.conv2d_transpose(pd, size, 4, stride=2) # 256 x 256 x 16
                    pd = tcl.conv2d_transpose(pd, size, 4, stride=1) # 256 x 256 x 16

                    pd = tcl.conv2d_transpose(pd, 3, 4, stride=1) # 256 x 256 x 3
                    pd = tcl.conv2d_transpose(pd, 3, 4, stride=1) # 256 x 256 x 3
                    pos = tcl.conv2d_transpose(pd, 3, 4, stride=1, activation_fn = tf.nn.sigmoid)#,padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))
                                
                    return pos
    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]


uv_kpt_ind = np.loadtxt('uv_kpt_ind.txt').astype(np.int32) # 2 x 68 get keypoints
face_ind = np.loadtxt('face_ind.txt').astype(np.int32) # get valid vertices in the position map
triangles = np.loadtxt('triangles.txt').astype(np.int32) # ntri x 3
end_list = np.array([17, 22, 27, 42, 48, 31, 36, 68], dtype = np.int32) - 1
def get_landmarks(pos):
        '''
        Args:
            pos: the 3D position map. shape = (256, 256, 3).
        Returns:
            kpt: 68 3D landmarks. shape = (68, 3).
        '''
        kpt = pos[uv_kpt_ind[1,:], uv_kpt_ind[0,:], :]
        return kpt

def plot_kpt(image, kpt):
    ''' Draw 68 key points
    Args: 
        image: the input image
        kpt: (68, 3).
    '''
    image = image.copy()
    kpt = np.round(kpt).astype(np.int32)
    for i in range(kpt.shape[0]):
        st = kpt[i, :2]
        image = cv2.circle(image,(st[0], st[1]), 1, (0,0,255), 2)  
        if i in end_list:
            continue
        ed = kpt[i + 1, :2]
        image = cv2.line(image, (st[0], st[1]), (ed[0], ed[1]), (255, 255, 255), 1)
    return image

os.environ['CUDA_VISIBLE_DEVICES']='0'
data_path =[];
test_images=[];
batch_size=16;
uv_weights = cv2.imread('defined_mask.png').astype(np.float32);
uv_weights = cv2.cvtColor(uv_weights, cv2.COLOR_BGR2RGB)
#uv_weights = uv_weights/255. * 16.
uv_weights = np.expand_dims(uv_weights, axis = 0);
def lossFunction(predicted, real):
  return tf.math.reduce_mean(tf.pow((predicted-real)*uv_weights,2));
tf.reset_default_graph() 
epochs = 50;
global_step = tf.Variable(0,trainable=False)
#learning_rate = tf.train.exponential_decay(0.0001, global_step, 5000, 0.9, staircase=False)
learning_rate = tf.train.cosine_decay_restarts(0.0001, global_step, 1000)
def loadTrainImages(index):
  images=[];
  
  for file in data_path[index*batch_size:min(len(data_path), (index+1)*batch_size)]:
    file = 'results/\\' + file[:-1] + '.jpg';
    images.append(io.imread(file)/255.); #Divide image by 255 to ease training? Yes.
  images = np.array(images).astype(np.float32)
  return images;
def loadPosMapTrain(index):
  posmaps=[];
  for file in data_path[index*batch_size:(index+1)*batch_size]:
    pos_map=np.load('results/\\'+file[:-1]+'.npy');
    pos_map = pos_map/abs(pos_map.max());
    posmaps.append(pos_map);
  posmaps = np.array(posmaps).astype(np.float32);
  return posmaps;
input_x = tf.placeholder(tf.float32, shape=[None,256, 256, 3])
posGrd = tf.placeholder(tf.float32, shape =[None, 256, 256, 3])
posPret = resfcn256(input_x);
posPret = posPret(input_x, is_training=True);
loss = lossFunction(posPret, posGrd);
tvars = [var for var in tf.global_variables() if 'resfcn256'  in var.name]
train_op = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8,
         use_locking=False).minimize(loss, global_step=global_step, var_list=tf.trainable_variables())
files = ['AFW', 'IBUG', 'LFPW', 'HELEN'];
for type in files:
    for path in open(type+'_Data.txt','r'):
      data_path.append(path);
index = 1;
posmaps = {};
for file in open('LFPW_Test_Data.txt', 'r'):
  real_value = 'results/uv_maps/\\'+file[:-1]+'_posmap.jpg';
  file = 'results/\\' + file[:-1] + '.jpg';
 # print(file+"\n"+real_value);
  img = io.imread(file).astype(np.float32)/255.
  posmaps[file] = io.imread(real_value).astype(np.float32);
  test_images.append(file)
  index+=1;
plt.xlabel('Steps');
plt.ylabel('Loss rate');
x_graph = [];
save_path = 'results/'
if not os.path.exists(save_path):
  os.mkdir(save_path);
y_graph = [];
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.InteractiveSession(config=config)
#sess = tf.Session(config=config);
init = tf.global_variables_initializer()
sess.run(init)
def testing(step):
  shuffle(test_images);
  tImages = test_images[0:8];
  index=1;
  for image in tImages:
    pos_map = posmaps[image];
    image1 = cv2.imread(image).astype(np.float32)/255.
    pos = sess.run(posPret, feed_dict = {input_x:image1[np.newaxis,:,:,:]});
    pos= pos*256*1.1;
    pos =np.squeeze(pos);
    pos = np.concatenate((pos, pos_map), axis=1);
    image1 = image1*256*1.1;
    pos = np.concatenate((plot_kpt(image1, get_landmarks(pos)), pos), axis=1);
    cv2.imwrite('results/'+'Step ' +str(step)+'_' + str(index)+'_posmap.jpg', pos);
    index +=1;
  '''
  Load testing images from LFPW testing, select 8 only
  Save output of network for each epoch in a different file (Image if possible)
  '''
saver = tf.train.Saver(tvars)
#saver = tf.train.Saver()
#saver = tf.train.import_meta_graph('../../content/gdrive/My Drive/checkpoint/.meta');
#saver.restore(sess, 'results/256_256_resfcn256_weight'); Uncomment if you want to load the model you trained before
batch_index= len(data_path);
step_loss = 0;
print('######\nEpochs: %d\nTraining Dataset size: %d\nIterations: %d\n######'%(epochs, len(data_path), int(math.ceil(len(data_path)/batch_size))));
for i in range(epochs):
    LOSS =0.0;
    shuffle(data_path)
    for j in range(0,int(batch_index/batch_size)):
        batch_images = loadTrainImages(j);
        batch_posmaps = loadPosMapTrain(j);
        _, posPredict, LOSS, lr = sess.run([train_op, posPret, loss, learning_rate], feed_dict={input_x:batch_images, posGrd:batch_posmaps})
        pos = np.squeeze(posPredict);
        #x_graph.append(j);
        #y_graph.append(lr);
        if(j %100 ==0):
          print ('[Step:%d|Epoch:%d], lr:%.6f, loss:%.4f' % (j, i, lr, LOSS))
    testing((i+1)*int(math.ceil(len(data_path)/batch_size)))
    #saver.save(sess, save_path);
    saver.save(sess, save_path+'256_256_resfcn256_weight');
    x_graph.append(i);
    y_graph.append(LOSS);
    plt.plot(x_graph,y_graph);
    plt.savefig('results/Training Model.jpg');
