#import the tensorflow library
import tensorflow as tf
import plotter

#Values that we will input during the computation (Define the placeholders)
x = tf.placeholder("float", [None, 784])
y = tf.placeholder("float", [None, 10])

#Variable of the model (part of the Tensorflow computational graph)
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

#Variable initialization
init = tf.initialize_all_variables()

#Softmax Regression (output layer) : Softmax Layer
y_hat = tf.nn.softmax(tf.matmul(x, W) + b)

#Cross-Entropy
cross_entropy = -tf.reduce_sum(y_hat*tf.log(y_hat))

#Define the training algorithm (SGD for minimizing the cross-entropy, learning-rate=0.01)
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

#Start a new session
sess = tf.InteractiveSession()
#sess = tf.Session() #Starting a new session

#Initialize the variables
sess.run(tf.initialize_all_variables())

#Train the model
for n in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_hat: batch_ys})
 
#Evaluation the prediction
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_hat,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_hat: mnist.test.labels}))

plotter.plot_mnist_weights(W.eval)