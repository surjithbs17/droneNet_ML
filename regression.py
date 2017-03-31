#import preprocessing
import numpy as np
import matplotlib.pyplot as plt
path = 'numpy_files/'
import tensorflow as tf


def visible_test():
	np_file_p = 'numpy_files/%d_%d_p.npy'%(185,712)
	p_img = np.load(np_file_p)
	p_imgs = p_img

	print(p_imgs.shape)
	mean_img = np.mean(p_imgs,axis =0)
	std_img = np.std(p_imgs,axis=0)
	#avg_std = np.mean(std_img,axis = 2).astype(np.uint8)
	print(mean_img.shape)
	img = p_imgs[300,:,:,1]
	mean = mean_img[:,:,1]
	print(img.shape)
	print(mean.shape)
	#plt.imshow(img.astype(np.uint8) - mean.astype(np.uint8))
	plt.imshow(mean.astype(np.uint8))
	plt.show()

def general(start,end,flag):
	for i in range(start,end):
		start = i*200
		end = 200
		if(flag == 'visible'):
			np_file_n = 'numpy_files/1503_1568_n.npy'
			np_file_p = 'numpy_files/185_712_p.npy'
		else:
			np_file_p = 'numpy_files/%d_%d_p.npy'%(start,end)
			np_file_n = 'numpy_files/%d_%d_n.npy'%(start,end)
		p_img = np.load(np_file_p)
		n_img = np.load(np_file_n)
		if i == 0:
			p_imgs = p_img
			n_imgs = n_img
		else:
			p_imgs = np.row_stack((p_imgs,p_img))
			n_imgs = np.row_stack((n_imgs,n_img))
		print(i)

	print(p_imgs.shape,n_imgs.shape)
	return p_imgs,n_imgs
	#mean_img = np.mean(n_imgs,axis =0)
	#std_img = np.std(n_imgs,axis=0)
	#avg_std = np.mean(std_img,axis = 2).astype(np.uint8)

	#img = p_imgs[10,:,:,:]
	#mean = mean_img[:,:,:]
	#print(img.shape)
	#print(mean.shape)
	#plt.imshow(mean.astype(np.uint8) - img.astype(np.uint8))
	#plt.imshow(img)
	#plt.imshow(std_img.astype(np.uint8))
	#plt.imshow(mean_img.astype(np.uint8))
	#plt.show()



# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


# Create model
def conv_net(x, weights, biases, dropout):
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, 100, 100, ])

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

def cnn(train_x,train_y,test_x,test_y):
	# Parameters
	learning_rate = 0.001
	training_iters = 45
	batch_size = 10
	display_step = 2

	# Network Parameters
	n_input = 30000 # MNIST data input (img shape: 28*28)
	n_classes = 2 # MNIST total classes (0-9 digits)
	dropout = 0.75 # Dropout, probability to keep units

	# tf Graph input
	x = tf.placeholder(tf.float32, [None, n_input])
	y = tf.placeholder(tf.float32, [None, n_classes])
	keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)

		# Store layers weight & bias
	weights = {
	    # 5x5 conv, 1 input, 32 outputs
	    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
	    # 5x5 conv, 32 inputs, 64 outputs
	    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
	    # fully connected, 7*7*64 inputs, 1024 outputs
	    'wd1': tf.Variable(tf.random_normal([7*7*64, 1024])),
	    # 1024 inputs, 10 outputs (class prediction)
	    'out': tf.Variable(tf.random_normal([1024, n_classes]))
	}

	biases = {

	    'bc1': tf.Variable(tf.random_normal([32])),
	    'bc2': tf.Variable(tf.random_normal([64])),
	    'bd1': tf.Variable(tf.random_normal([1024])),
	    'out': tf.Variable(tf.random_normal([n_classes]))
	}

	# Construct model
	pred = conv_net(x, weights, biases, keep_prob)

	# Define loss and optimizer
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

	# Evaluate model
	correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

	# Initializing the variables
	init = tf.global_variables_initializer()

	# Launch the graph
	with tf.Session() as sess:
	    sess.run(init)
	    step = 1
	    # Keep training until reach max iterations
	    while step * batch_size < training_iters:
	        #batch_x, batch_y = mnist.train.next_batch(batch_size)
	        # Run optimization op (backprop
	        batch_x = train_x[((step-1)*batch_size):((step)*batch_size),:,:,:]
	        batch_y = train_y[((step-1)*batch_size):((step)*batch_size),:]
	        batch_y = np.reshape(batch_y,(batch_size,n_classes))
	        batch_x = np.reshape(batch_x,(batch_size,n_input))
	        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,keep_prob: dropout})
	        if step % display_step == 0:
	            # Calculate batch loss and accuracy
	            loss, acc = sess.run([cost, accuracy], feed_dict={x: np.reshape(test_x,(test_x.shape[0],n_input)), y: np.reshape(test_y,(test_y.shape[0],n_classes)),
	                                                              keep_prob: 1.})
	            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
	                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
	                  "{:.5f}".format(acc))
	        step += 1
	    print("Optimization Finished!")

	    # Calculate accuracy for 256 mnist test images
	    print("Testing Accuracy:", \
	        sess.run(accuracy, feed_dict={x: mnist.test.images[:256],
	                                      y: mnist.test.labels[:256],
	                                      keep_prob: 1.}))



def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

def neural(train_x,train_y,test_x,test_y):
    print(train_x.shape,train_y.shape,test_x.shape,test_y.shape)
    n_hidden_1 = 512 # 1st layer number of features
    n_hidden_2 = 512 # 2nd layer number of features
    n_input = 30000
    n_classes = 2 

    learning_rate = 0.01
    training_epochs = 20
    batch_size = 20
    display_step = 10


    weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
    }
    biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
    }


    x = tf.placeholder(tf.float32, [None, n_input])
    y = tf.placeholder("float",[None,n_classes])

    pred = multilayer_perceptron(x, weights, biases)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
    init = tf.initialize_all_variables()

    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(training_epochs):
            avg_cost = 0
            total_batch = int (train_x.shape[0]/batch_size)

            for i in range(total_batch):
                batch_x = train_x[(i*batch_size):((i+1)*batch_size),:,:,:]
                batch_y = train_y[(i*batch_size):((i+1)*batch_size),:]

                #print("batch_x : ",batch_x.shape)
                #print("batch_y : ",batch_y.shape)
               	#print(batch_x)
               	#print(batch_x.shape)
                #print(batch_y)
                batch_y = np.reshape(batch_y,(batch_size,n_classes))
                batch_x = np.reshape(batch_x,(batch_size,n_input))

                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                          y: batch_y})
                avg_cost += c / total_batch



                if epoch % display_step == 0:
                    print("Epoch:", '%04d' % (epoch+1), "cost=","%f"%(avg_cost))
        print(weights['h1'].eval(),weights['h2'].eval(),biases['b1'].eval())
        print("Optimization Finished!")
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        
        #accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        accuracy = (tf.cast(correct_prediction, "float"))

        print (test_y.shape,test_x.shape,test_x.shape[0])
        print("Accuracy:", pred.eval({x: np.reshape(test_x,(test_x.shape[0],n_input)), y: np.reshape(test_y,(test_y.shape[0],n_classes))}))



def train_test_data(train_X,train_Y):
	p_samples = train_X.shape[0]
	rnd_indices = np.random.rand(p_samples) < 0.80

	train_px = np.array(train_X)[rnd_indices]
	#train_y = np.array(train_Y)[rnd_indices]
	train_py = np.tile([1,0],(train_px.shape[0],1))

	test_px = np.array(train_X)[~rnd_indices]
	test_py = np.tile([1,0],(test_px.shape[0],1))
	#test_y = np.array(train_Y)[~rnd_indices]
	n_samples = train_Y.shape[0]
	rndn_indices = np.random.rand(n_samples) < 0.80

	train_nx = np.array(train_Y)[rndn_indices]
	#train_y = np.array(train_Y)[rnd_indices]
	train_ny = np.tile([0,1],(train_nx.shape[0],1))

	test_nx = np.array(train_Y)[~rndn_indices]
	test_ny = np.tile([0,1],(test_nx.shape[0],1))

	
	train_x = np.row_stack((train_px,train_nx))
	train_y = np.row_stack((train_py,train_ny))
	test_x = np.row_stack((test_px,test_nx))
	test_y = np.row_stack((test_py,test_ny))
	#test_y = np.array(train_Y)[~rnd_indices]
	print(train_px.shape,train_py.shape,test_px.shape,test_py.shape)
	print(train_nx.shape,train_ny.shape,test_nx.shape,test_ny.shape)
	print(train_x.shape,train_y.shape,test_x.shape,test_y.shape)
	
	print(train_x.shape,train_y.shape,test_x.shape,test_y.shape)

	return train_x,train_y,test_x,test_y
	#print(train_y,test_y)


p_imgs,n_imgs = general(0,1,'visible')
train_x,train_y,test_x,test_y = train_test_data(p_imgs,n_imgs)
#neural(train_x[-100:,:,:,:],train_y[-100:,:],test_x[-20:,:,:,:],test_y[-20:,:])
cnn(train_x,train_y,test_x,test_y)
#visible_test()