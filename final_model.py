import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import random
import os
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf  
import pickle # serialization

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

physical_device = tf.config.experimental.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_device[0], True)

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
#cpus = tf.config.experimental.list_physical_devices(device_type='CPU')
#print(gpus, cpus)

def main():

  ############################
  # 0. environment setting
  
  # location of picture
  picdir= "./TrainData/"
  
  # size of picture
  width_size=224
  height_size=224
  
  ############################
  # 1. loading picture
  
  # 1.1. loading picture from image files
  y_labels,x_data = load_pictures(picdir,width_size,height_size)
  
  # 1.2. save samples to one file
  save_sample(y_labels,x_data,'pic_bw_db_224_224_1.p')
  
  # 1.3. load samples from file
  y_labels,x_data = load_sample('pic_bw_db_224_224_1.p')


  ############################
  # 2. data cleaning
  
  # 2.1. data cleaning for x and y
  y_c_labels, x_c_data=clean_data(y_labels,x_data)
  
  # 2.2. partition the data into training and testing sets
  
  # partition the data into training and testing sets
  # 75% of the data for training and the remaining 25% for testing
  (trainX, testX, trainY, testY) = train_test_split(x_c_data, y_c_labels,
              test_size=0.1, stratify=y_c_labels, random_state=42)
  
  ############################
  # 3. training
  
  # 3.1. training setting
  NumberOfOneTraining = 48 # batch size for one training
  Training_times= 100 # times of training
  
  # 3.2. build a training model
  model=Model(width_size,height_size)
  
  # 3.3. start training
  for i in range(Training_times): 

    # 3.3.1. get training data randomly from the testing 
    
    y_batch, x_batch = get_sample(trainY,trainX, NumberOfOneTraining)

    # 3.3.2. training
    # note! sometimes the training result is not as expected because of vanishing/exploding gradient problem
    # covolution & pooling operations are needed.
    model.train(y_batch, x_batch)
    
    # 3.3.3. testing 
    
    accuracy=model.test(testY,testX)
    print("train, {} , accuracy, {}".format( i,accuracy), flush=True)
  pass
  print("#### training finished ")
    
  # 4. testing the final result
  _sample_y, _sample_x = get_sample(testY, testX, 128)
  accuracy=model.test(_sample_y, _sample_x)
  print("final testing: accuracy {}".format( accuracy), flush=True)
  
  # 5. testing a picture
  #PicDir= "./person1946_bacteria_4874.jpeg"
  
  PicDir= "./lung.jpg"
  NArray_2D  = Image_to_Array(PicDir,width_size,height_size)  
  NArray_2D = to_tensor([NArray_2D])
  prediction=model.predict(NArray_2D)
  
  
  print(prediction)
  print(np.argmax(prediction))
  ans = np.argmax(prediction)
  if ans == 0 :
    print("No finding")
  elif ans == 1:
    print("肺炎")
  pass
  
  # 6. save model
  model.save_model("./model_1/")
  
pass

class Model():
  def __init__(self, i_width, i_height):
    self.width=i_width
    self.height=i_height
    # init tensorflow optimizer for learning
    self.opt =tf.keras.optimizers.Adam(learning_rate=1e-4)
    
    self.var_map={}
    self.op_map={}
    self.init_var()
  pass

  def init_var(self):

    # first convolution layer: 32 filters with size 5x5x1
    self.var_map['W_conv1'] = tf.Variable(tf.random.truncated_normal([5,5, 1,32], stddev=0.1))   
    self.var_map['b_conv1'] = tf.Variable(tf.constant(0.1, shape=[32]))  
  
    # second convolution layer: 64 filters with size 5x5x32
    self.var_map['W_conv2'] = tf.Variable(tf.random.truncated_normal([5,5, 32,64], stddev=0.1))   
    self.var_map['b_conv2'] = tf.Variable(tf.constant(0.1, shape=[64]))  
  
    # Y=AX+b     
    # define A
    self.var_map['A_fc2'] = tf.Variable(tf.random.truncated_normal([14* 14* 64, 2], stddev=0.1))  
    # define b
    self.var_map['b_fc2'] = tf.Variable(tf.constant(0.1, shape=[2]))                   

  pass

  def predict(self, i_x):

    stride_w=2
    stride_h=2
    stride_channel=1
    stride_batch=1
  
    # input operation
    self.op_map['tf_input_reshape']=i_x

    # 1st convolution operation
    self.op_map['sample_conv_1'] = tf.nn.relu(
                   tf.nn.conv2d(self.op_map['tf_input_reshape']
                                , self.var_map['W_conv1']
                                , strides=[stride_batch, stride_w,stride_h, stride_channel]
                                , padding='SAME') +
                   self.var_map['b_conv1'])    
    # print(self.op_map['sample_conv_1'].shape) # => (1, 107* 113, 32)
    # 107=width(213)/stride(2)
    # 113=height(226)/stride(2)
    # 32=self.op_map['W_conv1'].shape([5,5, 1,32])
    
    # 1st pooling with filter size 2x2
    self.op_map['sample_pool_1'] = tf.nn.max_pool(self.op_map['sample_conv_1']
                                 , ksize=[1,2,2,1]
                                 , strides=[stride_batch, stride_w,stride_h, stride_channel]
                                 , padding='SAME')
                                 
    # print(self.op_map['sample_pool_1'].shape) # => (1, 54* 57, 32)
    # 54=107/stride(2)
    # 57=113/stride(2)
    # 32=self.op_map['W_conv1'].shape([5,5, 1,32])

    # 2nd convolution operation
    self.op_map['sample_conv_2'] = tf.nn.relu(
                   tf.nn.conv2d(self.op_map['sample_pool_1']
                                , self.var_map['W_conv2']
                                , strides=[stride_batch,stride_w,stride_h, stride_channel]
                                , padding='SAME') +
                   self.var_map['b_conv2']) 
    # print(self.op_map['sample_conv_2'].shape) # => (1, 27, 29, 64)
    # 27=54/stride(2)
    # 29=57/stride(2)
    # 64=self.op_map['W_conv2'] = tf.Variable(tf.random.truncated_normal([5,5, 32,64], stddev=0.1)) 

    # 2nd pooling
    self.op_map['sample_pool_2'] = tf.nn.max_pool(self.op_map['sample_conv_2']
                                 , ksize=[1,2,2,1]
                                 , strides=[stride_batch, stride_w,stride_h, stride_channel]
                                 , padding='SAME')
    # print(self.op_map['sample_pool_2'].shape) # => (1, 14, 15, 64)
    # 14=27/stride(2)
    # 15=29/stride(2)
    # 64=self.op_map['W_conv2'] = tf.Variable(tf.random.truncated_normal([5,5, 32,64], stddev=0.1)) 

    # flate operation the 4-D array to 1-D array

    self.op_map['sample_flat_1'] = tf.reshape(self.op_map['sample_pool_2'], [-1, 14* 14* 64])

    # Y=AX+b operation  
    self.op_map['sample_fc2'] = tf.nn.softmax(tf.matmul(self.op_map['sample_flat_1'], self.var_map['A_fc2']) + self.var_map['b_fc2'])
    self.op_map['prediction']=self.op_map['sample_fc2'] 
    return self.op_map['prediction']
  pass

  def get_loss( self, y , prediction):
    return tf.reduce_mean(-tf.reduce_sum(tf.constant(y, dtype=tf.float32) * tf.math.log(prediction),axis=[1])) # loss
  pass

  def test(self,testY,testX):
    
    _testX = to_tensor(testX)
    prediction=self.predict(_testX)
    prediction=tf.math.argmax(prediction,1)
    z=np.argmax(testY,axis=1)-prediction.numpy()
    _succ=np.sum(z==0)
    total=len(testY)
    return _succ/total
    
  pass

  def train(self, y, x):
    
    channel=1
    with tf.GradientTape() as tape:
      x=to_tensor(x)
      prediction=self.predict(x)
    
      L = self.get_loss (y,prediction) # difference between y & prediction
      g = tape.gradient(L, self.var_map.values())
    pass
    result=zip(g, self.var_map.values())
    self.opt.apply_gradients(result)
    return L
  pass
  
  def save_model(self,i_filename):
    save = tf.train.Checkpoint()
    save.listed = list(self.var_map.values())
    save.save(i_filename)
  pass

  def load_model(self,i_filename):
    save = tf.train.Checkpoint()
    save.listed = list(self.var_map.values())
    save.restore(tf.train.latest_checkpoint(i_filename))
  pass
pass





def Image_to_Array(pic,width,height):

    image = load_img(pic, target_size=(width, height),color_mode='grayscale')
    image_array = img_to_array(image)
    image_preprocess = preprocess_input(image_array) # normalize the value of each element in array

    np_gray_2D = tf.reshape(image_preprocess, [width, height])

    return  np_gray_2D.numpy()
pass

def load_pictures(picdir,width_size,height_size):

  # init variables to store sample
  x_data = []
  y_labels = []
  
  for dirPath, dirNames, fileNames in os.walk(picdir):
      for f in fileNames:               
          # get y (ans)
          ans=int(dirPath[len(picdir):])
          
          # get x (picture)
          PicDir= os.path.join(dirPath, f)
          NArray_2D  = Image_to_Array(PicDir,width_size,height_size)  
          
          # store y, x 
          y_labels.append(ans)
          x_data.append(NArray_2D)
      pass
  pass
  return y_labels, x_data
pass

def clean_data(i_y,i_x):
  # convert the data and labels to NumPy arrays
  x_data = np.array(i_x, dtype="float32")
  y_labels = np.array(i_y)

  # perform one-hot encoding on the labels
  lb = LabelBinarizer()
  #y_labels2 = lb.fit_transform(y_labels)
  y_labels3 = to_categorical(y_labels,2) 
  
  return y_labels3, x_data
pass

def save_sample(i_y,i_x,i_filename):
  f = open(i_filename, 'wb')
  pickle.dump([i_y,i_x], f)
  f.close()
pass

def load_sample(i_filename):
  f = open(i_filename, 'rb')
  r=pickle.load(f)
  y_labels=r[0]
  x_data=r[1]
  return y_labels,x_data
pass

def to_tensor(i_x):
  channel=1 # black & white => 1, rgb=>3
  x=tf.constant(i_x, shape=[np.shape(i_x)[0],np.shape(i_x)[1],np.shape(i_x)[2],channel], dtype=tf.float32)
  return x
pass

def get_sample(i_y, i_x, i_sample_size):
    ##############################
    # get training data randomly
    
    # init x batch and y batch
    x_batch=[]
    y_batch=[]
    
    sample = random.sample(range(len(i_y)), i_sample_size)
  
    x_batch = [i_x[i] for i in sample]
    y_batch = [i_y[i] for i in sample]
   
    return y_batch, x_batch
pass

if __name__ == '__main__':
  main()  # entry function
pass

#### training finished 
#final testing: accuracy 0.75
#tf.Tensor([[2.021322e-09 1.000000e+00]], shape=(1, 2), dtype=float32)
#go to hell

#### training finished 
#final testing: accuracy 0.765625
#tf.Tensor([[1.302609e-04 9.998697e-01]], shape=(1, 2), dtype=float32)
#go to hell

