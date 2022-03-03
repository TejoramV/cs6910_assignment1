# Problem Statement:
The goal of this Assignment is to implement our own Feed-forward, back-propagation code without using any of the existing frame works, and use the gradient descents and its variants for classification task.
The report of the project can be found at [this wandb link.](https://wandb.ai/tejoram/CS6910-Assignment-1/reports/Assignment-1--VmlldzoxNjMzMTc2)

# Prerequisites:
```
Python 3.7.10
Numpy 1.19.5
```

## Dataset:
We have used Fashion-MNIST dataset.

## Installing:
+ Clone/download this repository.
+ For running in google colab, install wandb using following command - ```!pip install wandb```.
+ For running locally, install wandb using following command.
```
pip install wandb
pip install numpy
pip install keras
```

## Question-1:
There are ten classes in the Fashion-MNIST data set and here is a dictionary relating the model's numerical labels and corresponding class names.\
Class_labels_names = 
{       "0": "T-shirt/Top",         "1": "Trouser", \
        "2": "Pullover",            "3": "Dress",\
        "4": "Coat",                "5": "Sandal",\
        "6": "Shirt",               "7": "Sneaker",\
        "8": "Bag",                 "9": "Ankle Boot",     }
 
#### Solution Approach:
+ Import data from fashion_mnist.
+ Sort in `sorted_arr` until first occurrence of a class is found.
+ Plot all the classes with associated class names.
+ Integrate wandb to log the images and keep track of the experiment using wandb.ai.

## Question-2:
#### Solution Approach:
+ Feed-forward neural network `feed_forward()` has been implemented which takes in the training dataset(X_train), weights, biases, and activation function.
+ Initialize the randomized weights, biases as per the number of inputs, hidden & output layer specification using `param_inint`.
+ Implement loss functions such as:
        1. cross entropy
        2. Mean squared error
+ Implement Activation functions such as:
     - sigmoid, tanh, relu...etc
+ our code provides the flexibility in choosing the above mentioned parameters.
+ It also provides flexibility in choosing the number of neurons in each hidden layer.


## Question-3:
* Back propagation algorithm implemented with the support of the following optimization function and the code works for any batch size:
    * SGD 
    * Momentum based gradient descent 
    * Nesterov accelerated gradient descent
    * RMS Prop
    * ADAM
    * NADAM

#### Solution Approach:
+ Make use of output of the feed-forward neural network in the previous question.
+ Initialize `one_hot` function to encode the labels of images.
* Implement the activation functions and their gradients.
    * sgd
    * softmax
    * Rel
    * tanh
+ Initialize the randomized parameters using the 'random' in python.
+ Initialize predictions, accuracy and loss functions.
+ loss functions are:
    + Mean squared Error
    + Cross entropy
+ Initialize the gradient descent classes.
+ and Initialize the `train` function to use the above functions.


## Question-4:
#### Solution Approach:
+ Split the training data in the ratio of 9:1.
+ The standard training & test split of fashion_mnist has been used with 60000 training images and 10000 test images & labels.
+ 10% shuffled training data was kept aside as validation data for the hyperparameter search i.e, 6000 images.
+ wandb.sweeps() provides an functionality to comapre the different combinations of the hyperparameters for the training purpose.
+ we are avail with 3 types of search strategies which are:
    + grid
    + random
    + Bayes
+ By considering the number of parameters given, there are totally 5400 combinations are possible.
+ grid : It checks through all the possible combinations of hyperparameters. If there are n hyperparameters and m options of each hyperparameter. 
  There will be m^n number of runs to see the final picture, hence grid search strategy wont work beacause it would be a computationally intensive. 
+ There are 2 options left to choose.
+ we chose random search. and we obtained a maximum validation accuracy of 88.443% after picking the sweep function, set the sweep function of wandb by setting up the different parameters in sweep configuration 

## Question-5:
#### Solution Approach:
* The best accuracy across all the models is a validation accuracy of 88.443%.
+ The graph containing a summary of validation accuracies for all the models is shown in the wandb report.

## Question-10:
#### Solution Approach:
+ Since MNIST is a much simpler dataset, and a very similar image classification task with the same number of classes, the configurations of hyperparameters that worked well for Fashion-MNIST is worked well for MNIST too.
+ Although transfer learning from the pre trained Fashion MNIST dataset's best model configuration for the digits MNIST dataset is an extremely viable option for faster training and better initialization of the network, in the current implementation of the code, transfer learning has not been used. 





















