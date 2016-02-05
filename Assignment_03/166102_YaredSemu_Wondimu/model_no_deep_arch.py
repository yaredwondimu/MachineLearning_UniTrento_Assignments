#import the tensorflow library
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#MNIST Dataset
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

#Values that we will input during the computation (Define the placeholders)
x = tf.placeholder("float", [None, 784])
y = tf.placeholder("float", [None, 10])

#reshape vectors of size 784, to squares of size 28x28
'''
To apply the layer, we first reshape x to a 4d tensor, with the second and third dimensions corresponding to image width and height, and the final dimension corresponding to the number of color channels.
'''
x_image = tf.reshape(x,[-1,28,28,1])


W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

#Variable initialization
init = tf.initialize_all_variables()


#Weight Initialization
"""
Initializing variables to zero, when the activation of a layer
is made of ReLUs will yield a null gradient. This generates
dead neuros -> no learning!
More precisely a ReLU is not differentiable in 0, but it is 
differentiable in any  epsilon bubble defined around 0.
"""
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

#Convolution and Pooling  

"""
input : A Tensor, Must be one of the following types : float32, float64
filter : A Tensor, Must have the same type as the input
strides : A list of ints. 1-D of length 4.
		  The stride of the sliding window for each dimension of input
padding : A string from : "SAME", "VALID". 
		  The type of padding algorithm to use.
		  Usually, this leads to simplicity pad the input with zeros.
"""

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')



#First Convolutional Layer
"""
We can now implement our first layer. It will consist of convolution, followed by max pooling. The convolutional will compute 32 features for each 5x5 patch. Its weight tensor will have a shape of [5, 5, 1, 32]. The first two dimensions are the patch size, the next is the number of input channels, and the last is the number of output channels. We will also have a bias vector with a component for each output channel.
"""
# [width, height, depth, output_size]
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)


#Second Convolutional Layer										
"""
The second layer will have 64 features for each 5x5 patch.
"""
# [width, height, depth, output_size]
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)


# Third Layer_I (Densel connected layers : ReLU)										
"""
Now that the image size has been reduced to 7x7, we add a fully-connected layer with 1024 neurons to allow processing on the entire image. We reshape the tensor from the pooling layer into a batch of vectors, multiply by a weight matrix, add a bias, and apply a ReLU.
"""
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Third Layer_II (Add a dropout)
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


#Softmax Regression Layer
#[input_size,output_size]
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_hat=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

#Cross-Entropy
cross_entropy = -tf.reduce_sum(y*tf.log(y_hat))

#Define the training algorithm (Minimization of the cross-entropy)
#Learning-rate = 1e-4 with adaptive gradients.
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

#Start a new session
sess = tf.InteractiveSession()

#Initialize the variables
sess.run(tf.initialize_all_variables())

#Define accuracy
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_hat,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

#Train the model (This may take a while)
for i in range(20000):
  batch = mnist.train.next_batch(50)
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y: batch[1], keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))
  sess.run(train_step,feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
  #train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

#Evaluate the prediction (It should be around 0.99)
print("test accuracy %g"%accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})