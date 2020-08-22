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
