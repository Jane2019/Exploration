# Gradient descent

This project takes a dataset that has 16 features and try to build a multivariant linear regression by:

* Direct solution by solving the formula
  
  ![equation](http://www.sciweavers.org/tex2img.php?eq=w%20%3D%20%28X%5ETX%29%5E%7B-1%7DX%5ETt&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0)

* Using cost function:

  ![image](http://www.sciweavers.org/tex2img.php?eq=%5Cfrac%7B%7B%5Cpartial%20J%7D%7D%7B%5Cpartial%20w%7D%3D%5Cfrac%7B1%7D%7BN%7D%5B%5Csum_%7Bi%3D0%7D%5ENx%5E%7B%28i%29%7D%28y%5E%7B%28i%29%7D-t%5E%7B%28i%29%7D%29%5D&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0)

* Find solution by gradient descent

  ![image](http://www.sciweavers.org/tex2img.php?eq=w_j%3Dw_j-%5Calpha%2A%5Cfrac%7B%7B%5Cpartial%20J%7D%7D%7B%5Cpartial%20w%7D&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0)

* Find solution by gradient descent with regulizer

  ![image](http://www.sciweavers.org/tex2img.php?eq=w_j%3D%281-%5Calpha%5Clambda%29w_j-%5Calpha%2A%5Cfrac%7B%7B%5Cpartial%20J%7D%7D%7B%5Cpartial%20w%7D&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0)
  
* Find solution by mini-batch gradient descent: use the only batch weights to update the model weights

The objective is to explore math theory inside gradient descent and use linear regression here as an example.

By experiment, it will see difference between *gradient descent* and *mini-batch* and how *learning rates*, *batch size*, *regulization* have impact on converge times and train/test error.

## Good points:
  - Vectorization
  - Deep understanding in mathematics
  - How to choose hyparameters
 
