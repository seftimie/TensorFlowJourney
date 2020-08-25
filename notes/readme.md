# TensorFlow in Practice Specialization

## 1. Introduction to TensorFlow for Artificial Intelligence, Machine Learning, and Deep Learning

**A primer in machine learning:**
	
- The new paradigm is that I get lots and lots of examples and then I have labels on those examples and I use the data to say this is what walking looks like, this is what running looks like, this is what biking looks like and yes, even this is what golfing looks like. So, then it becomes answers and data in with rules being inferred by the machine. A machine learning algorithm then figures out the specific patterns in each set of data that determines the distinctiveness of each. That's what's so powerful and exciting about this programming paradigm. It's more than just a new way of doing the same old thing. It opens up new possibilities that were infeasible to do before.

**The ‘Hello World’ of neural networks**	
- A neural network is basically a set of functions which can learn patterns;
- The simplest possible neural network is one that has only one neuron in it;
- In keras, you use the word dense to define a layer of connected neurons; 
- Successive layers are defined in sequence, hence the word sequential;
- Important functions: loss function and optimizers;
- The epochs equals 500 value means that it will go through the training loop 500 times. This training loop is what we described earlier. Make a guess, measure how good or how bad the guesses with the loss function, then use the optimizer and the data to make another guess and repeat this. When the model has finished training, it will then give you back values using the predict method;
- When using neural networks, as they try to figure out the answers for everything, they deal in probability. You'll see that a lot and you'll have to adjust how you handle answers to fit;


**An Introduction to computer vision**
- Computer vision is the field of having a computer understand and label what is present in an image. Consider this slide. When you look at it, you can interpret what a shirt is or what a shoe is, but how would you program for that? If an extra terrestrial who had never seen clothing walked into the room with you, how would you explain the shoes to him? It's really difficult, if not impossible to do right? And it's the same problem with computer vision. So one way to solve that is to use lots of pictures of clothing and tell the computer what that's a picture of and then have the computer figure out the patterns that give you the difference between a shoe, and a shirt, and a handbag, and a coat. 


**The structure of Fashion MNIST data**
- Using a number is a first step in avoiding bias -- instead of labelling it with words in a specific language and excluding people who don’t speak that language! 


**Get hands-on with computer vision**
- Another rule of thumb -- the number of neurons in the last layer should match the number of classes you are classifying for. In this case it's the digits 0-9, so there are 10 of them, hence you should have 10 neurons in your final layer;
- Earlier when you trained for extra epochs you had an issue where your loss might change. It might have taken a bit of time for you to wait for the training to do that, and you might have thought 'wouldn't it be nice if I could stop the training when I reach a desired value?' -- i.e. 95% accuracy might be enough for you, and if you reach that after 3 epochs, why sit around waiting for it to finish a lot more epochs....So how would you fix that? Like any other program...you have callbacks! Let's see them in action;

```
	# better version with explaining at: https://www.tensorflow.org/tutorials/keras/classification
	#import libs
	import tensorflow as tf

	#define class & function for callback
	class myCallback(tf.keras.callbacks.Callback):
	  def on_epoch_end(self, epoch, logs={}):
	    if(logs.get('accuracy')>0.6):
	      print("\nReached 60% accuracy so cancelling training!")
	      self.model.stop_training = True

	#import and load dataset
	mnist = tf.keras.datasets.fashion_mnist
	(x_train, y_train),(x_test, y_test) = mnist.load_data()

	#transform for simplify input data
	x_train, x_test = x_train / 255.0, x_test / 255.0

	callbacks = myCallback()

	#define model
	model = tf.keras.models.Sequential([
	  tf.keras.layers.Flatten(input_shape=(28, 28)),
	  tf.keras.layers.Dense(512, activation=tf.nn.relu),
	  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
	])

	#compile model
	model.compile(optimizer=tf.optimizers.Adam(),
	              loss='sparse_categorical_crossentropy',
	              metrics=['accuracy'])

	#traing model
	model.fit(x_train, y_train, epochs=10, callbacks=[callbacks])
```

- How can I stop training when I reach a point that I want to be at? What do I always have to hard code it to go for certain number of epochs? Well, the good news is that, the training loop does support callbacks. So in every epoch, you can callback to a code function, having checked the metrics. If they're what you want to say, then you can cancel the training at that point;
- The first layer in this network, tf.keras.layers.Flatten, transforms the format of the images from a two-dimensional array (of 28 by 28 pixels) to a one-dimensional array (of 28 * 28 = 784 pixels); After the pixels are flattened, the network consists of a sequence of two tf.keras.layers.Dense layers. These are densely connected, or fully connected, neural layers. The first Dense layer has 128 nodes (or neurons). The second (and last) layer returns a logits array with length of 10. Each node contains a score that indicates the current image belongs to one of the 10 classes;
- Loss function — This measures how accurate the model is during training. You want to minimize this function to "steer" the model in the right direction;
- Optimizer — This is how the model is updated based on the data it sees and its loss function;
- Metrics — Used to monitor the training and testing steps. The following example uses accuracy, the fraction of the images that are correctly classified.
- The gap between training accuracy and test accuracy represents **overfitting**. Overfitting happens when a machine learning model performs worse on new, previously unseen inputs than it does on the training data. An overfitted model "memorizes" the noise and details in the training dataset to a point where it negatively impacts the performance of the model on the new data.
