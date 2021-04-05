# Tensorflow Levenberg-Marquardt
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fabiodimarco/tf-levenberg-marquardt/blob/main/tf-levenberg-marquardt.ipynb)

Implementation of Levenberg-Marquardt training for models that inherits from `tf.keras.Model`. The algorithm has been extended to support **mini-batch** training for both **regression** and **classification** problems.

##### Implemented losses
* MeanSquaredError
* ReducedOutputsMeanSquaredError
* CategoricalCrossentropy
* SparseCategoricalCrossentropy
* BinaryCrossentropy
* SquaredCategoricalCrossentropy
* CategoricalMeanSquaredError
* Support for custom losses

##### Implemented damping algorithms
* Standard: <img src="https://render.githubusercontent.com/render/math?math=\J^\T\!\J %2B \lambda \I" align="top" height="19px"/>
* Fletcher: <img src="https://render.githubusercontent.com/render/math?math=\J^\T\!\J %2B \lambda \diag\!(\J^\T\!\J\!)" align="top" height="23px"/>
* Support for custom damping algorithm

More details on Levenberg-Marquardt can be found on [this page](https://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm).

# Usage
Suppose that `model` and `train_dataset` has been created, it is possible to train the model in two different ways.

##### Custom Model.fit
This is similar to the standard way models are trained with keras, the difference is that the fit method is called on a `ModelWrapper` class instead of calling it directly on the `model` class.
```python
import levenberg_marquardt as lm

model_wrapper = lm.ModelWrapper(model)

model_wrapper.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.1),
    loss=lm.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])

model_wrapper.fit(train_dataset, epochs=5)
```

##### Custom training loop
This alternative is less flexible than Model.fit as for example it does not support callbacks and is only trainable using `tf.data.Dataset`.
```python
import levenberg_marquardt as lm

trainer = lm.Trainer(
    model=model,
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.1),
    loss=lm.SparseCategoricalCrossentropy(from_logits=True))

trainer.fit(
    dataset=train_dataset,
    epochs=5,
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
```

## Memory, Speed and Convergence considerations
In order to achieve the best performances out of the training algorithm some considerations have to be done, so that the batch size and the number of weights of the model are chosen properly. The Levenberg–Marquardt algorithm is used to solve the least squares problem

<p align="center">
  <img src="https://render.githubusercontent.com/render/math?math=\displaystyle\large \argmin_{W} \sum_{i=1}^{N}  [y_i - f(x_i, W)]^2"/><img align="right" src="https://render.githubusercontent.com/render/math?math=\begin{multline*}\tiny\phantom{1}\\ \large (1)\end{multline*}"/>
</p>

where 
<img src="https://render.githubusercontent.com/render/math?math=\begin{multline*} \Large \!\!\! y_i - f(x_i, w) \!\!\! \end{multline*}" align="top" height="27px"/> are the residuals, 
<img src="https://render.githubusercontent.com/render/math?math=W" align="bottom" height="12px"/> 
are weights of the model and 
<img src="https://render.githubusercontent.com/render/math?math=N" align="bottom" height="12px"/> 
the number of residuals, which can be obtained by multiplying the batch size with the number of outputs of the model.

<p align="center">
  <img src="https://render.githubusercontent.com/render/math?math=\displaystyle\large \num\!\_\residuals = \batch\!\_\size \cdot \num\!\_\outputs"/><img align="right" src="https://render.githubusercontent.com/render/math?math=\large (2)"/>
</p>

The Levenberg–Marquardt updates can be computed using two equivalent formulations

<p align="center">
  <img src="https://render.githubusercontent.com/render/math?math=\displaystyle\large \updates = (\J^\T\!\J %2B \:\damping\!)^{-1} \J^\T\!\residuals"/><img align="right" src="https://render.githubusercontent.com/render/math?math=\large (3\a\!)"/>
</p>

<p align="center">
  <img src="https://render.githubusercontent.com/render/math?math=\displaystyle\large \updates = \J^\T\!(\J\J^\T\! %2B \damping\!)^{-1} \residuals"/><img align="right" src="https://render.githubusercontent.com/render/math?math=\large (3\b\!)"/>
</p>

given that the size of the jacobian matrix is **[num_residuals x num_weights]**. 
In the first equation 
<img src="https://render.githubusercontent.com/render/math?math=\begin{multline*} \huge \!\!\!(\J^\T\!\J %2B \:\damping\!)\!\!\! \end{multline*}" align="top" height="25px"/> 
have size **[num_weights x num_weights]**, while in the second equation 
<img src="https://render.githubusercontent.com/render/math?math=\begin{multline*} \huge \!\!\!(\J\J^\T\! %2B \damping\!)\!\!\! \end{multline*}" align="top" height="25px"/> 
have size **[num_residuals x num_residuals]**.  
The first equation is convenient when **num_residuals > num_weights** *overdetermined*, while the second equation is convenient when **num_residuals < num_weights** *underdetermined*. For each batch the algorithm checks whether the problem is *overdetermined* or *underdetermined* and decide the update formula to use.

##### Splitted jacobian matrix computation
Equation 
<img src="https://render.githubusercontent.com/render/math?math=\begin{multline*} \Large \!\!\!(3\a\!)\!\!\! \end{multline*}" align="top" height="27px"/>, 
has some additional properties that can be exploited to reduce memory usage and increase speed as well. In fact, it is possible to split the jacobian computation avoiding the storage of the full matrix.
This is realized by splitting the batch in smaller sub-batches (of size 100 by default) so that the number of rows (number of residuals) of each jacobian matrix is at most

<p align="center">
  <img src="https://render.githubusercontent.com/render/math?math=\displaystyle\large \num\!\_\residuals = \sub\!\_\batch\!\_\size \cdot \num\!\_\outputs"/><img align="right" src="https://render.githubusercontent.com/render/math?math=\large (4)"/>
</p>

Equation 
<img src="https://render.githubusercontent.com/render/math?math=\begin{multline*} \Large \!\!\!(3\a\!)\!\!\! \end{multline*}" align="top" height="27px"/>, 
can be rewritten as

<p align="center">
  <img src="https://render.githubusercontent.com/render/math?math=\displaystyle\large \updates = (\J^\T\!\J %2B \:\damping\!)^{-1} \g"/><img align="right" src="https://render.githubusercontent.com/render/math?math=\large (5)"/>
</p>

where the hessian approximation
<img src="https://render.githubusercontent.com/render/math?math=\J^\T\!\J\!" align="bottom" height="15px"/> 
and the gradient 
<img src="https://render.githubusercontent.com/render/math?math=\begin{multline*}\!\!\!\!\!\!\\\!\!\!\huge\g\!\!\!\!\!\!\!\!\!\end{multline*}" align="top" height="27px"/> 
are computed without storing the full jacobian matrix as

<p align="center">
  <img src="https://render.githubusercontent.com/render/math?math=\displaystyle\large \J^\T\!\J = \sum_{i=1}^{M}  \bar\J_i^\T\!\bar\J_i"/><img align="right" src="https://render.githubusercontent.com/render/math?math=\begin{multline*}\tiny\phantom{1}\\ \large (6\a\!)\end{multline*}"/>
</p>

<p align="center">
  <img src="https://render.githubusercontent.com/render/math?math=\displaystyle\large \g = \sum_{i=1}^{M}  \bar\J_i^\T\!\residuals_i"/><img align="right" src="https://render.githubusercontent.com/render/math?math=\begin{multline*}\tiny\phantom{1}\\ \large (6\b\!)\end{multline*}"/>
</p>

##### Conclusions
From experiments I usually got better convergence using quite large batch sizes which statistically represent the entire dataset, but more experiments needs to be done with small batches and momentum.
However, if the model is big the usage of large batch sizes may not be possible. When the problem is *overdetermined*, the size of the linear system to solve at each step might be too large. When the problem is *underdetermined*, the full jacobian matrix might be too large to be stored.
Possible applications that could benefit from this algorithm are those training only a small number of parameters simultaneously. As for example: fine tuning, composed models and overfitting reduction using small models. In these cases the Levenberg–Marquardt algorithm could converge much faster then first-order methods.

## Results on curve fitting
Simple curve fitting test implemented in `test_curve_fitting.py` and [Colab](https://colab.research.google.com/github/fabiodimarco/tf-levenberg-marquardt/blob/main/tf-levenberg-marquardt.ipynb). The function `y = sinc(10 * x)` is fitted using a Shallow Neural Network with 61 parameters.
Despite the triviality of the problem, first-order methods such as Adam fail to converge, while Levenberg–Marquardt converges rapidly with very low loss values. The values of learning_rate were chosen experimentally on the basis of the results obtained by each algorithm.

Here the results with Adam for 10000 epochs and learning_rate=0.01
```
Train using Adam
Epoch 1/10000
20/20 [==============================] - 0s 500us/step - loss: 0.0479
...
Epoch 10000/10000
20/20 [==============================] - 0s 449us/step - loss: 6.5928e-04
Elapsed time:  157.10102080000001
```
Here the results with Levenberg–Marquardt for 100 epochs and learning_rate=1.0
```
Train using Levenberg-Marquardt
Epoch 1/100
20/20 [==============================] - 0s 6ms/step - damping_factor: 1.0000e-05 - attempts: 2.0000 - loss: 0.0232
...
Epoch 100/100
20/20 [==============================] - 0s 5ms/step - damping_factor: 1.0000e-07 - attempts: 1.0000 - loss: 1.0265e-08
Elapsed time:  14.7972407
```
##### Plot results
<img src="https://user-images.githubusercontent.com/15234505/97504830-fd83ad80-1977-11eb-9ca1-dd7113b980e3.png" width="400"/>

## Results on mnist dataset classification
Common mnist classification test implemented in `test_mnist_classification.py` and [Colab](https://colab.research.google.com/github/fabiodimarco/tf-levenberg-marquardt/blob/main/tf-levenberg-marquardt.ipynb). The classification is performed using a Convolutional Neural Network with 1026 parameters.
This time there were no particular benefits as in the previous case. Even if Levenberg–Marquardt converges with far fewer epochs than Adam, the longer execution time per step nullifies its advantages.
However, both methods achieve roughly the same accuracy values on train and test set.

Here the results with Adam for 300 epochs and learning_rate=0.01
```
Train using Adam
Epoch 1/200
10/10 [==============================] - 0s 28ms/step - loss: 2.0728 - accuracy: 0.3072
...
Epoch 200/200
10/10 [==============================] - 0s 14ms/step - loss: 0.0737 - accuracy: 0.9782
Elapsed time:  58.071762045000014

test_loss: 0.072342 - test_accuracy: 0.977100
```
Here the results with Levenberg–Marquardt for 10 epochs and learning_rate=0.1
```
Train using Levenberg-Marquardt
Epoch 1/10
10/10 [==============================] - 4s 434ms/step - damping_factor: 1.0000e-06 - attempts: 3.0000 - loss: 1.1267 - accuracy: 0.5803
...
Epoch 10/10
10/10 [==============================] - 5s 450ms/step - damping_factor: 1.0000e-05 - attempts: 2.0000 - loss: 0.0683 - accuracy: 0.9777
Elapsed time:  54.859995966999975

test_loss: 0.072407 - test_accuracy: 0.977800
```
## Requirements
 * Tensorflow 2.4+
 * Matplotlib to visualize curve fitting results
