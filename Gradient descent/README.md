# Linear regression with GD project

This project takes a house price dataset in Boston that has 16 features and try to build a multivariant linear regression by:

* Direct solution by solving the formula

* Find solution by gradient descent

* Find solution by gradient descent with regulizer
  
* Find solution by mini-batch gradient descent (use chosen batch weights to update the model weights)

* Apply adaptive learning rate to be faster

## Objective

  - Explore math algorithms inside gradient descent and use linear regression here as an example.
  - See difference between *gradient descent* and *mini-batch* and how *learning rates*, *batch size*, *regulization* have impact on converge times and train/test error.

## Good points
  - Vectorization
  - Deep understanding in mathematics
  - How to choose hyparameters
  - Adaptive learning rate methods (Time-decay)

# Gradient descent

Thanks to the great paper, here are something I learnt.

**Batch gradient** descent is to update the paraments by calculating the gradients of the whole training data. In another word, there is one update in each epoch for entire parameters. This is the earliest and original gradient descent form. There is only one loop inside the code of the algorithm, which is to set the number of epochs. The original gradient descent is sort of time consuming and cannot be done online, however, it would finally converge.

**Stochastic gradient** descent is randomly picking one training data point, computing its gradient and then updating the parameters based on that. In short, it updates once inside one epoch so there is an extra loop to go over all single data points in the training data compared to vanilla batch gradient descent. We can see, unlike batch gradient descent computing all gradients in one round, the gradient of each data point in SGD is computed one by one separately, which leads to lots of computation and overshooting behaviour although it is faster and can be done online.

**Mini-batch gradient** descent develops to correct some issues above. Mini-batch gradient descent comprises the first two forms and computes the gradients in a fixed number of training samples. Thus it turns out to be stable and have lower variance in each update.  here are still two loops, but the other loop goes over batches instead of all data points in SGD. It is more efficient but not perfect. It doesn't ensure to converge and may get trapped in saddle points. It is also tough to find a proper learning rate because it is not suitable to apply the same change for all parameters.
