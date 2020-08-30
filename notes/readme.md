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


**What are convolutions and pooling?**
- So, what's convolution? You might ask. Well, if you've ever done any kind of image processing, it usually involves having a filter and passing that filter over the image in order to change the underlying image;
- pooling is a way of compressing an image. A quick and easy way to do this, is to go over the image of four pixels at a time, i.e, the current pixel and its neighbors underneath and to the right of it. Of these four, pick the biggest value and keep just that. 


**Coding convolutions and pooling layers**
- The concepts introduced in this video are available as Conv2D layers and MaxPooling2D layers in TensorFlow.
- links: https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D
- links: https://www.tensorflow.org/api_docs/python/tf/keras/layers/MaxPool2D


**Implementing pooling layers**
- A really useful method on the model is the model.summary method. This allows you to inspect the layers of the model, and see the journey of the image through the convolutions


**Getting hands-on, your first ConvNet**
- turn your Deep Neural Network into a Convolutional Neural Network by adding convolutional layers on top, and having the network train against the results of the convolutions instead of the raw pixels


**Try it for yourself**

```
	import tensorflow as tf
	mnist = tf.keras.datasets.fashion_mnist
	(training_images, training_labels), (test_images, test_labels) = mnist.load_data()
	training_images=training_images.reshape(60000, 28, 28, 1)
	training_images=training_images / 255.0
	test_images = test_images.reshape(10000, 28, 28, 1)
	test_images=test_images/255.0
```
- You'll notice that there's a bit of a change here in that the training data needed to be reshaped. That's because the first convolution expects a single tensor containing everything, so instead of 60,000 28x28x1 items in a list, we have a single 4D list that is 60,000x28x28x1, and the same for the test images. If you don't do this, you'll get an error when training as the Convolutions do not recognize the shape.


**Experiment with the horse or human classifier**
- In this case, using the RMSprop optimization algorithm is preferable to stochastic gradient descent (SGD), because RMSprop automates learning-rate tuning for us. (Other optimizers, such as Adam and Adagrad, also automatically adapt the learning rate during training, and would work equally well here.)
- Let's set up data generators that will read pictures in our source folders, convert them to float32 tensors, and feed them (with their labels) to our network. We'll have one generator for the training images and one for the validation images. Our generators will yield batches of images of size 300x300 and their labels (binary). As you may already know, data that goes into neural networks should usually be normalized in some way to make it more amenable to processing by the network. (It is uncommon to feed raw pixels into a convnet.) In our case, we will preprocess our images by normalizing the pixel values to be in the [0, 1] range (originally all values are in the [0, 255] range). In Keras this can be done via the keras.preprocessing.image.ImageDataGenerator class using the rescale parameter. This ImageDataGenerator class allows you to instantiate generators of augmented image batches (and their labels) via .flow(data, labels) or .flow_from_directory(directory). These generators can then be used with the Keras model methods that accept data generators as inputs: fit, evaluate_generator, and predict_generator.

```
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1/255)

# Flow training images in batches of 128 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
        '/tmp/horse-or-human/',  # This is the source directory for training images
        target_size=(300, 300),  # All images will be resized to 150x150
        batch_size=128,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')
```


## 2. Convolutional Neural Networks in TensorFlow

**Before you Begin: TensorFlow 2.0 and this Course**
!pip install tensorflow==2.0.0-alpha0 




**Exercise 1 - Cats vs. Dogs**
I should store some helpers/snippets from all code course, things like:

```
# Use os.mkdir to create your directories
# You will need a directory for cats-v-dogs, and subdirectories for training
# and testing. These in turn will need subdirectories for 'cats' and 'dogs'
try:
    os.mkdir('/tmp/cats-v-dogs')
    os.mkdir('/tmp/cats-v-dogs/training')
    os.mkdir('/tmp/cats-v-dogs/testing')
    os.mkdir('/tmp/cats-v-dogs/training/cats')
    os.mkdir('/tmp/cats-v-dogs/training/dogs')
    os.mkdir('/tmp/cats-v-dogs/testing/cats')
    os.mkdir('/tmp/cats-v-dogs/testing/dogs')
except OSError:
    pass
```

or 

```
def split_data(SOURCE, TRAINING, TESTING, SPLIT_SIZE):
    files = []
    for filename in os.listdir(SOURCE):
        file = SOURCE + filename
        if os.path.getsize(file) > 0:
            files.append(filename)
        else:
            print(filename + " is zero length, so ignoring.")

    training_length = int(len(files) * SPLIT_SIZE)
    testing_length = int(len(files) - training_length)
    shuffled_set = random.sample(files, len(files))
    training_set = shuffled_set[0:training_length]
    testing_set = shuffled_set[-testing_length:]

    for filename in training_set:
        this_file = SOURCE + filename
        destination = TRAINING + filename
        copyfile(this_file, destination)

    for filename in testing_set:
        this_file = SOURCE + filename
        destination = TESTING + filename
        copyfile(this_file, destination)


CAT_SOURCE_DIR = "/tmp/PetImages/Cat/"
TRAINING_CATS_DIR = "/tmp/cats-v-dogs/training/cats/"
TESTING_CATS_DIR = "/tmp/cats-v-dogs/testing/cats/"
DOG_SOURCE_DIR = "/tmp/PetImages/Dog/"
TRAINING_DOGS_DIR = "/tmp/cats-v-dogs/training/dogs/"
TESTING_DOGS_DIR = "/tmp/cats-v-dogs/testing/dogs/"

split_size = .9
split_data(CAT_SOURCE_DIR, TRAINING_CATS_DIR, TESTING_CATS_DIR, split_size)
split_data(DOG_SOURCE_DIR, TRAINING_DOGS_DIR, TESTING_DOGS_DIR, split_size)
```


**Transfer learning: Start coding!**
- A pre-trained model is a saved network that was previously trained on a large dataset, typically on a large-scale image-classification task. You either use the pretrained model as is or use transfer learning to customize this model to a given task. The intuition behind transfer learning for image classification is that if a model is trained on a large and general enough dataset, this model will effectively serve as a generic model of the visual world. You can then take advantage of these learned feature maps without having to start from scratch by training a large model on a large dataset;
- Feature Extraction: Use the representations learned by a previous network to extract meaningful features from new samples. You simply add a new classifier, which will be trained from scratch, on top of the pretrained model so that you can repurpose the feature maps learned previously for the dataset. You do not need to (re)train the entire model. The base convolutional network already contains features that are generically useful for classifying pictures. However, the final, classification part of the pretrained model is specific to the original classification task, and subsequently specific to the set of classes on which the model was trained;
- Fine-Tuning: Unfreeze a few of the top layers of a frozen model base and jointly train both the newly-added classifier layers and the last layers of the base model. This allows us to "fine-tune" the higher-order feature representations in the base model in order to make them more relevant for the specific task;
- By specifying the include_top=False argument, you load a network that doesn't include the classification layers at the top, which is ideal for feature extraction;
- Freeze the convolutional base: It is important to freeze the convolutional base before you compile and train the model. Freezing (by setting layer.trainable = False) prevents the weights in a given layer from being updated during training. MobileNet V2 has many layers, so setting the entire model's trainable flag to False will freeze all of them;



**Using dropouts!**
- Another useful tool to explore at this point is the Dropout. The idea behind Dropouts is that they remove a random number of neurons in your neural network. This works very well for two reasons: The first is that neighboring neurons often end up with similar weights, which can lead to overfitting, so dropping some out at random can remove this. The second is that often a neuron can over-weigh the input from a neuron in the previous layer, and can over specialize as a result. Thus, dropping out can break the neural network out of this potential bad habit!;



**Moving from binary to multi-class classification**
- You're coming to the end of Course 2, and you've come a long way! From first principles in understanding how ML works, to using a DNN to do basic computer vision, and then beyond into Convolutions. With Convolutions, you then saw how to extract features from an image, and you saw the tools in TensorFlow and Keras to build with Convolutions and Pooling as well as handling complex, multi-sized images. Through this you saw how overfitting can have an impact on your classifiers, and explored some strategies to avoid it, including Image Augmentation, Dropouts, Transfer Learning and more. To wrap things up, this week you've looked at the considerations in your code that you need for moving towards multi-class classification!

