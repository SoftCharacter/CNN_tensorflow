import tensorflow.compat.v1 as tf
from tensorflow_core.examples.tutorials.mnist import input_data

tf.disable_v2_behavior()
#number 1 to 10 data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

def compute_accuracy(v_xs,v_ys):
    global prediction
    y_pre = sess.run(prediction,feed_dict={xs:v_xs,keep_prob:1})
    correct_prediction = tf.equal(tf.argmax(y_pre,1),tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    result = sess.run(accuracy,feed_dict={xs:v_xs,ys:v_ys,keep_prob:1})
    return result

def weight_variable(shape):#卷积核大小
	initial=tf.truncated_normal(shape,stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):#卷积核数 一般为2的整数次方
	initial=tf.constant(0.1,shape=shape)
	return tf.Variable(initial)

def conv2d(x,W):#卷积
    #stride[1,x_movement,y_movement,1]
	return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def max_pooling_2x2(x):#池化
	return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')


#define placeholder for inputs to network
keep_prob = tf.placeholder(tf.float32) #保留值  需要工作的神经元的百分比
xs = tf.placeholder(tf.float32, [None, 784]) #28x28
ys = tf.placeholder(tf.float32, [None, 10])
x_image=tf.reshape(xs,[-1,28,28,1])

#conv1 layer
W_conv1=weight_variable([5,5,1,32])#patch 5x5 in size 1,out size 32
b_conv1=bias_variable([32])
 # 定义好了Weight和bias，我们就可以定义卷积神经网络的第一个卷积层
 # h_conv1=conv2d(x_image,W_conv1)+b_conv1,同时我们对h_conv1进行非线性处理，
 # 也就是激活函数来处理喽，这里我们用的是tf.nn.relu（修正线性单元）来处理，
 # 要注意的是，因为采用了SAME的padding方式，输出图片的大小没有变化依然是28x28，
 # 只是厚度变厚了，因此现在的输出大小就变成了28x28x32
h_conv1=tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)
h_pool1=max_pooling_2x2(h_conv1)#输出大小就变为了14x14x32
#conv2 layer
W_conv2=weight_variable([5,5,32,64])
b_conv2=bias_variable([64])
h_conv2=tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)#输出的大小就是14x14x64
h_pool2=max_pooling_2x2(h_conv2)#输出大小为7x7x64

#func1 layer
h_pool2_flat=tf.reshape(h_pool2,[-1,7*7*64])#[n_samples,7,7,64]->>[n_samples,7*7*64]
W_fc1=weight_variable([7*7*64,1024])
b_fc1=bias_variable([1024])

h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)
h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob)
#func2 layer
W_fc2=weight_variable([1024,10])
b_fc2=bias_variable([10])

prediction=tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2) + b_fc2)

#the error between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                              reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys,keep_prob:0.5})
        if i % 50 == 0:
            print(compute_accuracy(mnist.test.images[:1000], mnist.test.labels[:1000]))