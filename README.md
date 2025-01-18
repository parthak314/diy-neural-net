# diy-neural-net
This project will take you through understanding a neural network, the maths behind it and how it is used for basic projects. This will also help you to break from 'tutorial hell', if that's you.

Please note that this guide is there to help anyone from a basic level. To view a high level overview please see the [README](README.md)

There is a lot of hype around ai, but what is it actually? Most tutorials will start from using python libraries such as keras, pytorch or tensorflow. But there is more to this.

## Project Goal
To actually understand how this works, let's work on a simple project to recognise a single digit number using basic python (including numpy and pandas - which are only there to save time, our understanding won't be affected by this).

Note that much of this is inspired by the neural network series by [3blue1brown](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi).

## Prereqs
Let's think about how this idea will work.

We need to have some sort of input and output within the system here. Let's begin by taking in some data.

### Dataset
A quick Google search will help us to find the [MNIST dataset](https://www.kaggle.com/datasets/hojjatk/mnist-dataset) which is many numbers put together.

Note that our end goal is to build a system where we can write a number, and the text version is produced.

Each image from this is 784 pixels (28 by 28), with each pixel having a value between 0 and 255 in terms of their 'brightness' with 0 being black and 255 being white. 

For $m$ images in the database we will have a Matrix $X$ representing a collection of all these images:
$ X = \begin{pmatrix} \bf{a_1} \\ \bf{a_2} \\ : \\ \bf{a_m} \end{pmatrix}^T = \begin{pmatrix} \bf{a_1} & \bf{a_2} & ... & \bf{a_m} \end{pmatrix}$
Transposing the Matrix gives us a matrix with dimensions $784 \times m$ with each columns having all the pixels for each image.

### Neural Network
A **neural network** here will help us to classify these images. The first layer should be able to take all the pixels as an input, therefore having 784 nodes. The hidden layers wil have 10 nodes each (for every digit that can be predicted).

Let our neural network be another matri with $A^{[0]}$ being the input layer (784 rows), $A^{[1]}$ being the first hidden layer ($10 \times m$), $A^{[2]}$ being the second hidden layer and then the output.

To train a neural network means to teach it to recognise patterns. A simple way is to:
- Forward Propagation: Provide input data and expect an output
- Back Propagation: going the opposite way to see by how much the prediction deviated.

**Forward Propagation**
$A^{[0]}$: Input

$A^{[1]} = ReLU(Z^{[1]})$
$Z^{[1]} = W^{[1]} \cdot A^{[0]} + b^{[1]}$ represents the weights we added to each pixel and a bias.
At this stage, we form a linear combination of the input layer. If we continue on from here, by processing a linear combnation again and again, it doesn't really improve the system.
Instead, we use an activation function. An ideal one would have Non-linearity, Zero-centeres (reduces bias), avoids vanishing gradients, efficient, smooth and differentiable for backpropagation.
- ReLU which has efficient computation and mitigates the vanishing gradient, but can lead to "dead neurons"
$ReLU(x) = \begin{cases} x & x>0 \\ 0 & x<= 0\end{cases}$
- Sigmoid which outputs between 0 and 1 but may cause vanishing gradients
$\sigma(x) = \frac{1}{1+e^{-x}}$
- Tanh which is centered at around 0 but can have vanishing gradients
$\text{tanh}(x) = \frac{1-e^{-2x}}{1+e^{-2x}}$
- Softmax which normalises it prodcing probability, making sense to genrally use this for the output.
$\text{Softmax}(x_i) = \frac{e^{x_i}}{\sum_{j} e^{x_j}}$

Here we will use ReLU due to its simplicity.

$A^{[2]} = ReLU(Z^{[1]})$
This follows a similar pattern to calculate the next layer:
$Z^{[2]} = W^{[2]} \cdot A^{[1]} + b^{[2]}$
Note that now we want to obtain the probability of the input being the number. The previous layer procudes results which are variable. To limit this between 0 and 1, we need to use another function that operates this way.

**Back Propagtaion**
This is all about finding the error in the network. This is also known as the deviation from the desired value. (Note that the output is $Y$)
In the last layer, this is $dZ^{[2]} = A^{[2]} - Y$ via one-hot encoding
>One hot encoding is simply subtracting from one and one element only by using the matrix defined as $Y = \begin{pmatrix} 0 \\ 1 \\ 0 \\ : \\ 0 \end{pmatrix}$ for y = 1

We repeat this to find the errors from the Weights and the biases for each level. For level 2 this is:
$dW^{[2]} = \frac{1}{m}dZ^{[2]}A^{[1]T}$
$db^{[2]} = \frac {1}{m}\sum dZ^{[2]}$
This tells us by how much we should change our weights and biases.
Note that for the first layer, we also need to find the derivative of the activation function. Which can be done by finding its derivative.

**Training result**
Now we can adjust the Weights and the biases:
$W^{[1]} = W^{[1]}_{prev} - \alpha dW^{[1]}$
$b^{[1]} = b^{[1]}_{prev} - \alpha db^{[1]}$
etc. Where $\alpha$ is a hyper/learning parameter that is set by us. 

Then we keep repeating this until the model gives us the desired outputs i.e. the numbers we want.

## Developing the model
Please see the [MNIST NN Notebook](./MNIST-NN.ipynb)

## Challenge: characters
Using the EMNIST dataset

## Extending this project
Now it's up to you!

Keep me posted on what you decide to go on to do from here, but to help you here are a few ideas:
- Recogniser game - draw an object and the system recognises it
- Create a One-note addon (or any other tool) to change the handwriting to text live
- Multilingual recogniser - such as Arabic or Devanagari characters
- An app to take a picture and retrieve key information such as total amount spent from receipts.


