#coding:utf-8
import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
import struct
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import cv2,csv
import res_dnn

			
def encode_labels( y, k):
	onehot = np.zeros((y.shape[0],k ))
	for idx, val in enumerate(y):
		onehot[idx,val] = 1.0  
	return onehot

def load_mnist(path, kind='train'):
	"""Load MNIST data from `path`"""
	if kind=='train':
		labels_path=os.path.abspath('../mnist/train-labels-idx1-ubyte')		
		images_path=os.path.abspath('../mnist/train-images-idx3-ubyte')
	else:
		labels_path=os.path.abspath('../mnist/t10k-labels-idx1-ubyte')		
		images_path=os.path.abspath('../mnist/t10k-images-idx3-ubyte')
	
	with open(labels_path, 'rb') as lbpath:
		magic, n = struct.unpack('>II',
								 lbpath.read(8))
		labels = np.fromfile(lbpath,
							 dtype=np.uint8)

	with open(images_path, 'rb') as imgpath:
		magic, num, rows, cols = struct.unpack(">IIII",
											   imgpath.read(16))
		images = np.fromfile(imgpath,
							 dtype=np.uint8).reshape(len(labels), 784)

	return images, labels

BATCH_SIZE = 50
LEARNING_RATE_BASE = 0.1
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 20000
MOVING_AVERAGE_DECAY = 0.99 
MODEL_SAVE_PATH = "./res_DNN/"
MODEL_NAME = "res_model"
INPUT_NODE = 784
OUTPUT_NODE = 10
IMAGE_SIZE = 28
NUM_CHANNELS = 1
NUM_LABELS = 10
display_step = 20
learning_rate_flag=True


def train(X_train,y_train_lable,X_test,y_test_lable):
	shuffle=True
	batch_idx=0
	
	batch_len =int( X_train.shape[0]/BATCH_SIZE)
	train_acc=[]
	train_idx=np.random.permutation(batch_len)

	x_ = tf.placeholder(tf.float32, [None, INPUT_NODE],name='x-input')	
	x = tf.reshape(x_, shape=[-1, 28, 28, 1])
	y_ = tf.placeholder(tf.float32, [None,NUM_LABELS], name='y-input')

	regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
	cosine,loss= res_dnn.inference(x,True,regularizer,tf.argmax(y_,1))
	global_step = tf.Variable(0, trainable=False)

        pred_max=tf.argmax(cosine,1)
        y_max=tf.argmax(y_,1)
        correct_pred = tf.equal(pred_max,y_max)
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


	if learning_rate_flag==True:
		learning_rate = tf.train.exponential_decay(
			LEARNING_RATE_BASE,
			global_step,
			X_train.shape[0] / BATCH_SIZE, LEARNING_RATE_DECAY,
			staircase=True)
	else:	
		learning_rate = 0.001 
	train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

	saver = tf.train.Saver()
	with tf.Session() as sess:
		tf.global_variables_initializer().run()
		
		step = 1
		print ("Start  training!")
		while step< TRAINING_STEPS:
			batch_shuffle_idx=train_idx[batch_idx]
			batch_xs=X_train[batch_shuffle_idx*BATCH_SIZE:batch_shuffle_idx*BATCH_SIZE+BATCH_SIZE]
			batch_ys=y_train_lable[batch_shuffle_idx*BATCH_SIZE:batch_shuffle_idx*BATCH_SIZE+BATCH_SIZE]	
			if batch_idx<batch_len:
				batch_idx+=1
				if batch_idx==batch_len:
					batch_idx=0
			else:
				batch_idx=0
		
			reshaped_xs = np.reshape(batch_xs, (
					BATCH_SIZE,
					IMAGE_SIZE,
					IMAGE_SIZE,
					NUM_CHANNELS))
			
			_,step,acc_train,loss_val=sess.run([train_step,global_step,accuracy,loss], feed_dict={x: reshaped_xs, y_: batch_ys})
			loss_txt=[]
			loss_txt.append(loss_val)
			if step % display_step == 0:
				print ('acc on train batch is %g'%(acc_train))
				X_test=np.reshape(X_test[:3000],(3000,28,28,1))
				loss_value,acc=sess.run([loss,accuracy] , feed_dict={x:X_test,y_:y_test_lable[:3000]})
				print("After %d training step(s), loss on test data is %g,acc is %g" % (step, loss_value,acc))
				saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME),global_step=global_step)
			step += 1
		print ("Optimization Finished!")
		print("Save model...")
		saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME))
		file=open('loss_val.txt','w')
		file.writelines(loss_txt)
		file.close()

		
def main(argv=None):
	X_train, y_train = load_mnist('..\mnist', kind='train')
	X_test, y_test = load_mnist('mnist', kind='t10k')	
        X_train=np.vstack((X_train,X_train))
        y_train=np.hstack((y_train,y_train))
        print y_train.shape
        for i in range(60000,120000):
             tmp=np.reshape(X_train[i],(28,28))
  	     t=np.random.random()
             if t < 0.3:
                   new_image = cv2.flip(tmp, 1)
             elif 0.3<=t<0.6:
                   new_image = cv2.flip(tmp, 0)
 	     else:
		   new_image = cv2.flip(tmp, -1)
             new_image=new_image.flatten()
             X_train[i]=new_image
	mms=MinMaxScaler()
 	X_train=mms.fit_transform(X_train)
        X_test=mms.fit_transform(X_test)
	y_train_lable = encode_labels(y_train,10)
	y_test_lable = encode_labels(y_test,10)
	train(X_train,y_train_lable,X_test,y_test_lable)

if __name__ == '__main__':
        start = time.time()
	main()
        end = time.time()
        print  end-start
        print  'I have trained %d mins and %d seconds'%((end-start)/60,(end-start)%60)
#coding:utf-8
