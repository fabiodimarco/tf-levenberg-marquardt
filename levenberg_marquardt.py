# Copyright (c) 2020 Fabio Di Marco
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ==============================================================================

import tensorflow as tf
from tensorflow.python.keras.engine import data_adapter

# ==============================================================================


class MeanSquaredError(tf.keras.losses.MeanSquaredError):
    """Provides mean squared error metrics: loss / residuals.

    Use mean squared error for regression problems with one or more outputs.
    """

    def residuals(self, y_true, y_pred):
        return y_true - y_pred


class ReducedOutputsMeanSquaredError(tf.keras.losses.Loss):
    """Provides mean squared error metrics: loss / residuals.

    Consider using this reduced outputs mean squared error loss for regression
    problems with a large number of outputs or at least more then one output.
    This loss function reduces the number of outputs from N to 1, reducing both
    the size of the jacobian matrix and backpropagation complexity.
    Tensorflow, in fact, uses backward differentiation which computational
    complexity is  proportional to the number of outputs.
    """

    def __init__(self,
                 reduction=tf.keras.losses.Reduction.AUTO,
                 name='reduced_outputs_mean_squared_error'):
        super(ReducedOutputsMeanSquaredError, self).__init__(
            reduction=reduction,
            name=name)

    def call(self, y_true, y_pred):
        sq_diff = tf.math.squared_difference(y_true, y_pred)
        return tf.math.reduce_mean(sq_diff, axis=1)

    def residuals(self, y_true, y_pred):
        sq_diff = tf.math.squared_difference(y_true, y_pred)
        eps = tf.keras.backend.epsilon()
        return tf.math.sqrt(eps + tf.math.reduce_mean(sq_diff, axis=1))


"""
    The gauss-newthon algorithm is obtained from the linear approximation of the
    squared residuals and it is used solve least square problems.
    A way to use cross-entropy instead of mean squared error is to compute
    residuals as the square root of the cross-entropy.
"""


class CategoricalCrossentropy(tf.keras.losses.CategoricalCrossentropy):
    """Provides cross-entropy metrics: loss / residuals.

    Use this cross-entropy loss for classification problems with two or more
    label classes. The labels are expected to be provided in a `one_hot`
    representation.
    """

    def residuals(self, y_true, y_pred):
        eps = tf.keras.backend.epsilon()
        return tf.math.sqrt(eps + self.fn(y_true, y_pred, **self._fn_kwargs))


class SparseCategoricalCrossentropy(
        tf.keras.losses.SparseCategoricalCrossentropy):
    """Provides cross-entropy metrics: loss / residuals.

    Use this cross-entropy loss for classification problems with two or more
    label classes. The labels are expected to be provided as integers.
    """

    def residuals(self, y_true, y_pred):
        eps = tf.keras.backend.epsilon()
        return tf.math.sqrt(eps + self.fn(y_true, y_pred, **self._fn_kwargs))


class BinaryCrossentropy(tf.keras.losses.BinaryCrossentropy):
    """Provides cross-entropy metrics: loss / residuals.

    Use this cross-entropy loss for classification problems with only two label
    classes (assumed to be 0 and 1). For each example, there should be a single
    floating-point value per prediction.
    """

    def residuals(self, y_true, y_pred):
        eps = tf.keras.backend.epsilon()
        return tf.math.sqrt(eps + self.fn(y_true, y_pred, **self._fn_kwargs))


"""
    Other experimental losses for classification problems.
"""


class SquaredCategoricalCrossentropy(tf.keras.losses.Loss):
    """Provides squared cross-entropy metrics: loss / residuals.

    Use this cross-entropy loss for classification problems with two or more
    label classes. The labels are expected to be provided in a `one_hot`
    representation.
    """

    def __init__(self,
                 from_logits=False,
                 label_smoothing=0,
                 reduction=tf.keras.losses.Reduction.AUTO,
                 name='squared_categorical_crossentropy'):
        super(SquaredCategoricalCrossentropy, self).__init__(
            reduction=reduction,
            name=name)
        self.from_logits = from_logits
        self.label_smoothing = label_smoothing

    def call(self, y_true, y_pred):
        return tf.math.square(tf.keras.losses.categorical_crossentropy(
            y_true,
            y_pred,
            self.from_logits,
            self.label_smoothing))

    def residuals(self, y_true, y_pred):
        return tf.keras.losses.categorical_crossentropy(
            y_true,
            y_pred,
            self.from_logits,
            self.label_smoothing)

    def get_config(self):
        config = {'from_logits': self.from_logits,
                  'label_smoothing': self.label_smoothing}
        base_config = super(SquaredCategoricalCrossentropy, self).get_config()
        return dict(base_config + config)


class CategoricalMeanSquaredError(tf.keras.losses.Loss):
    """Provides mean squared error metrics: loss / residuals.

    Use this categorical mean squared error loss for classification problems
    with two or more label classes. The labels are expected to be provided in a
    `one_hot` representation and the output activation to be softmax.
    """

    def __init__(self,
                 reduction=tf.keras.losses.Reduction.AUTO,
                 name='categorical_mean_squared_error'):
        super(CategoricalMeanSquaredError, self).__init__(
            reduction=reduction,
            name=name)

    def call(self, y_true, y_pred):
        # Selects the y_pred which corresponds to y_true equal to 1.
        prediction = tf.reduce_sum(tf.math.multiply(y_true, y_pred), axis=1)
        return tf.math.squared_difference(1.0, prediction)

    def residuals(self, y_true, y_pred):
        # Selects the y_pred which corresponds to y_true equal to 1.
        prediction = tf.reduce_sum(tf.math.multiply(y_true, y_pred), axis=1)
        return 1.0 - prediction


# ==============================================================================


class DampingAlgorithm:
    """Default Levenberg–Marquardt damping algorithm.

    This is used inside the Trainer as a generic class. Many damping algorithms
    can be implemented using the same interface.
    """

    def __init__(self,
                 starting_value=1e-3,
                 dec_factor=0.1,
                 inc_factor=10.0,
                 min_value=1e-10,
                 max_value=1e+10,
                 fletcher=False):
        """Initializes `DampingAlgorithm` instance.

        Args:
          starting_value: (Optional) Used to initialize the Trainer internal
            damping_factor.
          dec_factor: (Optional) Used in the train_step decrease the
            damping_factor when new_loss < loss.
          inc_factor: (Optional) Used in the train_step increase the
            damping_factor when new_loss >= loss.
          min_value: (Optional) Used as a lower bound for the damping_factor.
            Higher values improve numerical stability in the resolution of the
            linear system, at the cost of slower convergence.
          max_value: (Optional) Used as an upper bound for the damping_factor,
            and as condition to stop the Training process.
          fletcher: Bool (Optional) Replace the identity matrix with
            diagonal of the gauss-newton hessian approximation, so that there is
            larger movement along the directions where the gradient is smaller.
            This avoids slow convergence in the direction of small gradient.
        """
        self.starting_value = starting_value
        self.dec_factor = dec_factor
        self.inc_factor = inc_factor
        self.min_value = min_value
        self.max_value = max_value
        self.fletcher = fletcher

    def init_step(self, damping_factor, loss):
        return damping_factor

    def decrease(self, damping_factor, loss):
        return tf.math.maximum(
            damping_factor * self.dec_factor,
            self.min_value)

    def increase(self, damping_factor, loss):
        return tf.math.minimum(
            damping_factor * self.inc_factor,
            self.max_value)

    def stop_training(self, damping_factor, loss):
        return damping_factor >= self.max_value

    def apply(self, damping_factor, JJ):
        if self.fletcher:
            damping = tf.linalg.tensor_diag(tf.linalg.diag_part(JJ))
        else:
            damping = tf.eye(tf.shape(JJ)[0], dtype=JJ.dtype)

        damping = tf.scalar_mul(damping_factor, damping)
        return tf.add(JJ, damping)


# ==============================================================================

class Trainer:
    """Levenberg–Marquardt training algorithm.
    """

    def __init__(self,
                 model,
                 optimizer=tf.keras.optimizers.SGD(learning_rate=1.0),
                 loss=MeanSquaredError(),
                 damping_algorithm=DampingAlgorithm(),
                 attempts_per_step=10,
                 solve_method='qr',
                 jacobian_max_num_rows=100,
                 experimental_use_pfor=True):
        """Initializes `Trainer` instance.

        Args:
          model: It is the Model to be trained, it is expected to inherit
            from tf.keras.Model and to be already built.
          optimizer: (Optional) Performs the update of the model trainable
            variables. When tf.keras.optimizers.SGD is used it is equivalent
            to the operation `w = w - learning_rate * updates`, where updates is
            the step computed using the Levenberg-Marquardt algorithm.
          loss: (Optional) An object which inherits from tf.keras.losses.Loss
          and have an additional function to compute residuals.
          damping_algorithm: (Optional) Class implementing the damping
            algorithm to use during training.
          attempts_per_step: Integer (Optional) During the train step when new
            model variables are computed, the new loss is evaluated and compared
            with the old loss value. If new_loss < loss, then the new variables
            are accepted, otherwise the old variables are restored and
            new ones are computed using a different damping-factor.
            This argument represents the maximum number of attempts, after which
            the step is taken.
          solve_method: (Optional) Possible values are:
            'qr': Uses QR decomposition which is robust but slower.
            'cholesky': Uses Cholesky decomposition which is fast but may fail
                when the hessian approximation is ill-conditioned.
            'solve': Uses tf.linalg.solve. I don't know what algorithm it
                implements. But it seems a compromise in terms of speed and
                robustness.
          jacobian_max_num_rows: Integer (Optional) When the number of residuals
            is greater then the number of variables (overdetermined), the
            hessian approximation is computed by slicing the input and
            accumulate the result of each computation. In this way it is
            possible to drastically reduce the memory usage and increase the
            speed as well. The input is sliced into blocks of size less than or
            equal to the jacobian_max_num_rows.
          experimental_use_pfor: (Optional) If true, vectorizes the jacobian
            computation. Else falls back to a sequential while_loop.
            Vectorization can sometimes fail or lead to excessive memory usage.
            This option can be used to disable vectorization in such cases.
        """
        if not model.built:
            raise ValueError('Trainer model has not yet been built. '
                             'Build the model first by calling `build()` or '
                             'calling `fit()` with some data, or specify an '
                             '`input_shape` argument in the first layer(s) for '
                             'automatic build.')

        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.damping_algorithm = damping_algorithm
        self.attempts_per_step = attempts_per_step
        self.jacobian_max_num_rows = jacobian_max_num_rows
        self.experimental_use_pfor = experimental_use_pfor

        # Define and select linear system equation solver.
        def qr(matrix, rhs):
            q, r = tf.linalg.qr(matrix, full_matrices=True)
            y = tf.linalg.matmul(q, rhs, transpose_a=True)
            return tf.linalg.triangular_solve(r, y, lower=False)

        def cholesky(matrix, rhs):
            chol = tf.linalg.cholesky(matrix)
            return tf.linalg.cholesky_solve(chol, rhs)

        def solve(matrix, rhs):
            return tf.linalg.solve(matrix, rhs)

        if solve_method == 'qr':
            self.solve_function = qr
        elif solve_method == 'cholesky':
            self.solve_function = cholesky
        elif solve_method == 'solve':
            self.solve_function = solve
        else:
            raise ValueError('Invalid solve_method.')

        # Keep track of the current damping_factor.
        self.damping_factor = tf.Variable(
            self.damping_algorithm.starting_value,
            trainable=False,
            dtype=self.model.dtype)

        # Used to backup and restore model variables.
        self._backup_variables = []

        # Since training updates are computed with shape (num_variables, 1),
        # self._splits and self._shapes are needed to split and reshape the
        # updates so that they can be applied to the model trainable_variables.
        self._splits = []
        self._shapes = []

        for variable in self.model.trainable_variables:
            variable_shape = tf.shape(variable)
            variable_size = tf.reduce_prod(variable_shape)
            backup_variable = tf.Variable(
                tf.zeros_like(variable),
                trainable=False)

            self._backup_variables.append(backup_variable)
            self._splits.append(variable_size)
            self._shapes.append(variable_shape)

        self._num_variables = tf.reduce_sum(self._splits).numpy().item()
        self._num_outputs = None

    @tf.function
    def _compute_jacobian(self, inputs, targets):
        with tf.GradientTape(persistent=True) as tape:
            outputs = self.model(inputs, training=True)
            residuals = self.loss.residuals(targets, outputs)

        jacobians = tape.jacobian(
            residuals,
            self.model.trainable_variables,
            experimental_use_pfor=self.experimental_use_pfor)

        del tape

        num_residuals = tf.reduce_prod(tf.shape(residuals))
        jacobians = [tf.reshape(j, (num_residuals, -1)) for j in jacobians]
        jacobian = tf.concat(jacobians, axis=1)
        residuals = tf.reshape(residuals, (num_residuals, -1))

        return jacobian, residuals, outputs

    def _init_gauss_newton_overdetermined(self, inputs, targets):
        # Perform the following computation:
        # J, residuals, outputs = self._compute_jacobian(inputs, targets)
        # JJ = tf.linalg.matmul(J, J, transpose_a=True)
        # rhs = tf.linalg.matmul(J, residuals, transpose_a=True)
        #
        # But reduce memory usage by slicing the inputs so that the jacobian
        # matrix will have maximum shape (jacobian_max_num_rows, num_variables)
        # instead of (batch_size, num_variables).
        slice_size = self.jacobian_max_num_rows // self._num_outputs
        batch_size = tf.shape(inputs)[0]
        num_slices = batch_size // slice_size
        remainder = batch_size % slice_size

        JJ = tf.zeros(
            [self._num_variables, self._num_variables],
            dtype=self.model.dtype)

        rhs = tf.zeros(
            [self._num_variables, 1],
            dtype=self.model.dtype)

        outputs_array = tf.TensorArray(tf.float32, size=0, dynamic_size=True)

        for i in tf.range(num_slices):
            tf.autograph.experimental.set_loop_options(
                shape_invariants=[
                    (rhs, tf.TensorShape((self._num_variables, None)))])

            _inputs = inputs[i * slice_size:(i + 1) * slice_size]
            _targets = targets[i * slice_size:(i + 1) * slice_size]

            J, residuals, _outputs = self._compute_jacobian(_inputs, _targets)

            outputs_array = outputs_array.write(i, _outputs)

            JJ += tf.linalg.matmul(J, J, transpose_a=True)
            rhs += tf.linalg.matmul(J, residuals, transpose_a=True)

        if remainder > 0:
            _inputs = inputs[num_slices * slice_size::]
            _targets = targets[num_slices * slice_size::]

            J, residuals, _outputs = self._compute_jacobian(_inputs, _targets)

            if num_slices > 0:
                outputs = tf.concat([outputs_array.concat(), _outputs], axis=0)
            else:
                outputs = _outputs

            JJ += tf.linalg.matmul(J, J, transpose_a=True)
            rhs += tf.linalg.matmul(J, residuals, transpose_a=True)
        else:
            outputs = outputs_array.concat()

        return 0.0, JJ, rhs, outputs

    def _init_gauss_newton_underdetermined(self, inputs, targets):
        J, residuals, outputs = self._compute_jacobian(inputs, targets)
        JJ = tf.linalg.matmul(J, J, transpose_b=True)
        rhs = residuals
        return J, JJ, rhs, outputs

    def _compute_gauss_newton_overdetermined(self, J, JJ, rhs):
        updates = self.solve_function(JJ, rhs)
        return updates

    def _compute_gauss_newton_underdetermined(self, J, JJ, rhs):
        updates = self.solve_function(JJ, rhs)
        updates = tf.linalg.matmul(J, updates, transpose_a=True)
        return updates

    def _train_step(self, inputs, targets,
                    init_gauss_newton, compute_gauss_newton):
        # J: jacobian matrix not used in the overdetermined case.
        # JJ: gauss-newton hessian approximation
        # rhs: gradient when overdetermined, residuals when underdetermined.
        # outputs: prediction of the model for the current inputs.
        J, JJ, rhs, outputs = init_gauss_newton(inputs, targets)

        # Perform normalization for numerical stability.
        batch_size = tf.shape(inputs)[0]
        normalization_factor = 1.0 / tf.dtypes.cast(
            batch_size,
            dtype=self.model.dtype)

        JJ *= normalization_factor
        rhs *= normalization_factor

        # Compute the current loss value.
        loss = self.loss(targets, outputs)

        stop_training = False
        attempt = 0
        damping_factor = self.damping_algorithm.init_step(
            self.damping_factor, loss)

        attempts = tf.constant(self.attempts_per_step, dtype=tf.int32)

        while tf.constant(True, dtype=tf.bool):
            # Apply the damping to the gauss-newton hessian approximation.
            JJ_damped = self.damping_algorithm.apply(damping_factor, JJ)

            # Compute the updates:
            # overdetermined: updates = (J'*J + damping)^-1*J'*residuals
            # underdetermined: updates = J'*(J*J' + damping)^-1*residuals
            updates = compute_gauss_newton(J, JJ_damped, rhs)

            # Split and Reshape the updates
            updates = tf.split(tf.squeeze(updates, axis=-1), self._splits)
            updates = [tf.reshape(update, shape)
                       for update, shape in zip(updates, self._shapes)]

            # Apply the updates to the model trainable_variables.
            self.optimizer.apply_gradients(
                zip(updates, self.model.trainable_variables))

            if attempt < attempts:
                attempt += 1

                # Compute the new loss value.
                outputs = self.model(inputs, training=False)
                new_loss = self.loss(targets, outputs)

                if new_loss < loss:
                    # Accept the new model variables and backup them.
                    loss = new_loss
                    damping_factor = self.damping_algorithm.decrease(
                        damping_factor, loss)
                    self.backup_variables()
                    break

                # Restore the old variables and try a new damping_factor.
                damping_factor = self.damping_algorithm.increase(
                    damping_factor, loss)
                self.restore_variables()

                stop_training = self.damping_algorithm.stop_training(
                    damping_factor, loss)
                if stop_training:
                    break
            else:
                break

        # Update the damping_factor which will be used in the next train_step.
        self.damping_factor.assign(damping_factor)
        return loss, outputs, attempt, stop_training

    def _compute_num_outputs(self, inputs, targets):
        input_shape = inputs.shape[1::]
        target_shape = targets.shape[1::]
        _inputs = tf.keras.Input(shape=input_shape,
                                 dtype=inputs.dtype)
        _targets = tf.keras.Input(shape=target_shape,
                                  dtype=targets.dtype)
        outputs = self.model(_inputs)
        residuals = self.loss.residuals(_targets, outputs)
        return tf.reduce_prod(residuals.shape[1::])

    def reset_damping_factor(self):
        self.damping_factor.assign(self.damping_algorithm.starting_value)

    def backup_variables(self):
        zip_args = (self.model.trainable_variables, self._backup_variables)
        for variable, backup in zip(*zip_args):
            backup.assign(variable)

    def restore_variables(self):
        zip_args = (self.model.trainable_variables, self._backup_variables)
        for variable, backup in zip(*zip_args):
            variable.assign(backup)

    def train_step(self, inputs, targets):
        if self._num_outputs is None:
            self._num_outputs = self._compute_num_outputs(inputs, targets)

        batch_size = tf.shape(inputs)[0]
        num_residuals = batch_size * self._num_outputs
        overdetermined = num_residuals >= self._num_variables

        if overdetermined:
            loss, outputs, attempts, stop_training = self._train_step(
                inputs,
                targets,
                self._init_gauss_newton_overdetermined,
                self._compute_gauss_newton_overdetermined)
        else:
            loss, outputs, attempts, stop_training = self._train_step(
                inputs,
                targets,
                self._init_gauss_newton_underdetermined,
                self._compute_gauss_newton_underdetermined)

        return loss, outputs, attempts, stop_training

    def fit(self, dataset, epochs=1, metrics=None):
        """Trains self.model on the dataset for a fixed number of epochs.

        Arguments:
            dataset: A `tf.data` dataset, must return a tuple (inputs, targets).
            epochs: Integer. Number of epochs to train the model.
            metrics: List of metrics to be evaluated during training.
        """
        self.backup_variables()
        steps = dataset.cardinality().numpy().item()
        stop_training = False

        if metrics is None:
            metrics = []

        pl = tf.keras.callbacks.ProgbarLogger(
            count_mode='steps',
            stateful_metrics=["damping_factor", "attempts"])

        pl.set_params(
            {"verbose": 1, "epochs": epochs, "steps": steps})

        pl.on_train_begin()

        for epoch in range(epochs):
            if stop_training:
                break

            # Reset metrics.
            for m in metrics:
                m.reset_states()

            pl.on_epoch_begin(epoch)

            iterator = iter(dataset)

            for step in range(steps):
                if stop_training:
                    break

                pl.on_train_batch_begin(step)

                data = next(iterator)

                data = data_adapter.expand_1d(data)
                inputs, targets, sample_weight = \
                    data_adapter.unpack_x_y_sample_weight(data)

                loss, outputs, attempts, stop_training = \
                    self.train_step(inputs, targets)

                # Update metrics.
                for m in metrics:
                    m.update_state(targets, outputs)

                logs = {"damping_factor": self.damping_factor,
                        "attempts": attempts,
                        "loss": loss}
                logs.update({m.name: m.result() for m in metrics})

                pl.on_train_batch_end(step, logs)

            pl.on_epoch_end(epoch)

        pl.on_train_end()


# ==============================================================================


class ModelWrapper(tf.keras.Sequential):
    """Wraps a keras model.

    When fit is called, the wrapped model is trained using Levenberg–Marquardt.
    """

    def __init__(self, model):
        if not model.built:
            raise ValueError('This model has not yet been built. '
                             'Build the model first by calling `build()` or '
                             'calling `fit()` with some data, or specify an '
                             '`input_shape` argument in the first layer(s) for '
                             'automatic build.')

        super(ModelWrapper, self).__init__([model])
        self.model = model
        self.trainer = None

    def compile(self,
                optimizer=tf.keras.optimizers.SGD(learning_rate=1.0),
                loss=MeanSquaredError(),
                damping_algorithm=DampingAlgorithm(),
                attempts_per_step=10,
                solve_method='qr',
                jacobian_max_num_rows=100,
                experimental_use_pfor=True,
                metrics=None,
                loss_weights=None,
                weighted_metrics=None,
                **kwargs):
        super(ModelWrapper, self).compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
            loss_weights=loss_weights,
            weighted_metrics=weighted_metrics,
            run_eagerly=True)

        self.trainer = Trainer(
            model=self.model,
            optimizer=optimizer,
            loss=loss,
            damping_algorithm=damping_algorithm,
            attempts_per_step=attempts_per_step,
            solve_method=solve_method,
            jacobian_max_num_rows=jacobian_max_num_rows,
            experimental_use_pfor=experimental_use_pfor)

    def train_step(self, data):
        data = data_adapter.expand_1d(data)
        inputs, targets, sample_weight = \
            data_adapter.unpack_x_y_sample_weight(data)

        loss, outputs, attempts, stop_training = \
            self.trainer.train_step(inputs, targets)

        self.compiled_metrics.update_state(targets, outputs)

        logs = {"damping_factor": self.trainer.damping_factor,
                "attempts": attempts,
                "loss": loss}
        logs.update({m.name: m.result() for m in self.metrics})

        # BUG: In tensorflow v2.2.0 and v2.3.0 setting model.stop_training=True
        # does not stop training immediately, but only at the end of the epoch.
        # https://github.com/tensorflow/tensorflow/issues/41174
        self.stop_training = stop_training

        return logs

    def fit(self,
            x=None,
            y=None,
            batch_size=None,
            epochs=1,
            verbose=1,
            callbacks=None,
            **kwargs):
        if verbose > 0:
            if callbacks is None:
                callbacks = []

            callbacks.append(tf.keras.callbacks.ProgbarLogger(
                count_mode='steps',
                stateful_metrics=["damping_factor", "attempts"]))

        super(ModelWrapper, self).fit(
            x=x,
            y=y,
            batch_size=batch_size,
            epochs=epochs,
            verbose=verbose,
            callbacks=callbacks,
            **kwargs)
