from abc import ABC, abstractmethod

import keras
import tensorflow as tf
from tensorflow import Tensor


class Loss(keras.losses.Loss, ABC):
    """Base class for all loss functions using ABC."""

    @abstractmethod
    def residuals(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        """Computes the residuals between `y_true` and `y_pred`."""
        pass


class MeanSquaredError(keras.losses.MeanSquaredError, Loss):
    """Provides mean squared error metrics: loss / residuals.

    Use mean squared error for regression problems with one or more outputs.
    """

    def residuals(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.convert_to_tensor(y_true)
        return tf.subtract(y_true, y_pred)


class ReducedOutputsMeanSquaredError(Loss):
    """Provides mean squared error metrics: loss / residuals.

    Consider using this reduced outputs mean squared error loss for regression problems
    with a large number of outputs or at least more than one output. This loss function
    reduces the number of outputs from N to 1, reducing both the size of the Jacobian
    matrix and backpropagation complexity. TensorFlow, in fact, uses backward
    differentiation which computational complexity is proportional to the number of
    outputs.
    """

    def __init__(
        self,
        reduction: str = 'sum_over_batch_size',
        name: str = 'reduced_outputs_mean_squared_error',
    ):
        super(ReducedOutputsMeanSquaredError, self).__init__(
            reduction=reduction, name=name
        )

    def call(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.convert_to_tensor(y_true)
        sq_diff = tf.math.squared_difference(y_true, y_pred)
        return tf.math.reduce_mean(sq_diff, axis=1)

    def residuals(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.convert_to_tensor(y_true)
        sq_diff = tf.math.squared_difference(y_true, y_pred)
        eps = keras.backend.epsilon()
        return tf.math.sqrt(eps + tf.math.reduce_mean(sq_diff, axis=1))


"""
The Gauss-Newton algorithm is obtained from the linear approximation of the squared
residuals and is used to solve least squares problems. A way to use cross-entropy
instead of mean squared error is to compute residuals as the square root of the
cross-entropy.
"""


class CategoricalCrossentropy(keras.losses.CategoricalCrossentropy, Loss):
    """Provides cross-entropy metrics: loss / residuals.

    Use this cross-entropy loss for classification problems with two or more label
    classes. The labels are expected to be provided in a `one_hot` representation.
    """

    def residuals(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.convert_to_tensor(y_true)
        eps = keras.backend.epsilon()
        return tf.math.sqrt(eps + self.fn(y_true, y_pred, **self._fn_kwargs))


class SparseCategoricalCrossentropy(keras.losses.SparseCategoricalCrossentropy, Loss):
    """Provides cross-entropy metrics: loss / residuals.

    Use this cross-entropy loss for classification problems with two or more label
    classes. The labels are expected to be provided as integers.
    """

    def residuals(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.convert_to_tensor(y_true)
        eps = keras.backend.epsilon()
        return tf.math.sqrt(eps + self.fn(y_true, y_pred, **self._fn_kwargs))


class BinaryCrossentropy(keras.losses.BinaryCrossentropy, Loss):
    """Provides cross-entropy metrics: loss / residuals.

    Use this cross-entropy loss for classification problems with only two label classes
    (assumed to be 0 and 1). For each example, there should be a single floating-point
    value per prediction.
    """

    def residuals(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.convert_to_tensor(y_true)
        eps = keras.backend.epsilon()
        return tf.math.sqrt(eps + self.fn(y_true, y_pred, **self._fn_kwargs))


"""
Other experimental losses for classification problems.
"""


class SquaredCategoricalCrossentropy(Loss):
    """Provides squared cross-entropy metrics: loss / residuals.

    Use this cross-entropy loss for classification problems with two or more label
    classes. The labels are expected to be provided in a `one_hot` representation.
    """

    def __init__(
        self,
        from_logits: bool = False,
        label_smoothing: float = 0.0,
        reduction: str = 'sum_over_batch_size',
        name: str = 'squared_categorical_crossentropy',
    ):
        super(SquaredCategoricalCrossentropy, self).__init__(
            reduction=reduction, name=name
        )
        self.from_logits = from_logits
        self.label_smoothing = label_smoothing

    def call(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        return tf.math.square(
            keras.losses.categorical_crossentropy(
                y_true,
                y_pred,
                from_logits=self.from_logits,
                label_smoothing=self.label_smoothing,
            )
        )

    def residuals(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        loss = keras.losses.categorical_crossentropy(
            y_true,
            y_pred,
            from_logits=self.from_logits,
            label_smoothing=self.label_smoothing,
        )
        return tf.convert_to_tensor(loss)

    def get_config(self):
        config = {
            'from_logits': self.from_logits,
            'label_smoothing': self.label_smoothing,
        }
        base_config = super(SquaredCategoricalCrossentropy, self).get_config()
        return {**base_config, **config}


class CategoricalMeanSquaredError(Loss):
    """Provides mean squared error metrics: loss / residuals.

    Use this categorical mean squared error loss for classification problems with two or
    more label classes. The labels are expected to be provided in a `one_hot`
    representation and the output activation to be softmax.
    """

    def __init__(
        self,
        reduction: str = 'sum_over_batch_size',
        name: str = 'categorical_mean_squared_error',
    ):
        super(CategoricalMeanSquaredError, self).__init__(
            reduction=reduction, name=name
        )

    def call(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.convert_to_tensor(y_true)
        # Selects the y_pred which corresponds to y_true equal to 1.
        prediction = tf.reduce_sum(tf.math.multiply(y_true, y_pred), axis=1)
        return tf.math.squared_difference(1.0, prediction)

    def residuals(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.convert_to_tensor(y_true)
        # Selects the y_pred which corresponds to y_true equal to 1.
        prediction = tf.reduce_sum(tf.math.multiply(y_true, y_pred), axis=1)
        return 1.0 - prediction
