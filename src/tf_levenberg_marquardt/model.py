import keras
from tensorflow import Tensor
from tensorflow.python.keras.engine import data_adapter

from .damping import DampingAlgorithm
from .loss import Loss, MeanSquaredError
from .training import Trainer


class ModelWrapper(keras.Model):
    """Wraps a keras model.

    When fit is called, the wrapped model is trained using Levenberg-Marquardt.
    """

    def __init__(self, model: keras.Model) -> None:
        super(ModelWrapper, self).__init__()
        self.model = model
        self.trainer = None

    def call(self, inputs: Tensor, training=None, mask=None) -> Tensor:
        return self.model(inputs, training=training, mask=mask)

    def compute_output_shape(self, input_shape):
        return self.model.compute_output_shape(input_shape)

    def compile(
        self,
        optimizer=keras.optimizers.SGD(learning_rate=1.0),
        loss: Loss = MeanSquaredError(),
        damping_algorithm=DampingAlgorithm(),
        attempts_per_step: int = 10,
        solve_method: str = 'qr',
        jacobian_max_num_rows: int = 100,
        experimental_use_pfor: bool = True,
        metrics=None,
        loss_weights=None,
        weighted_metrics=None,
        **kwargs,
    ) -> None:
        super(ModelWrapper, self).compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
            loss_weights=loss_weights,
            weighted_metrics=weighted_metrics,
            run_eagerly=True,
        )

        self.built = self.model.built

        self.trainer = Trainer(
            model=self,
            optimizer=optimizer,
            loss=loss,
            damping_algorithm=damping_algorithm,
            attempts_per_step=attempts_per_step,
            solve_method=solve_method,
            jacobian_max_num_rows=jacobian_max_num_rows,
            experimental_use_pfor=experimental_use_pfor,
        )

    def train_step(self, data):
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

        if self.trainer is None:
            raise ValueError(
                'Trainer is not initialized. Please compile the model first.'
            )

        loss, y_pred, attempts, stop_training = self.trainer.train_step(x, y)

        logs = {
            'damping_factor': self.trainer.damping_factor.numpy().item(),
            'attempts': attempts,
            'loss': loss.numpy().item(),
        }

        metrics = self.compute_metrics(x, y, y_pred, sample_weight)

        if 'loss' in metrics:
            del metrics['loss']

        logs.update({k: v for k, v in metrics.items()})

        # BUG: In TensorFlow v2.2.0 and v2.3.0 setting model.stop_training=True
        # does not stop training immediately, but only at the end of the epoch.
        # https://github.com/tensorflow/tensorflow/issues/41174
        self.stop_training = stop_training

        return logs

    def fit(
        self,
        x=None,
        y=None,
        batch_size=None,
        epochs: int = 1,
        verbose: int = 1,
        callbacks=None,
        **kwargs,
    ):
        if verbose > 0:
            if callbacks is None:
                callbacks = []

            callbacks.append(keras.callbacks.ProgbarLogger())

        return super(ModelWrapper, self).fit(
            x=x,
            y=y,
            batch_size=batch_size,
            epochs=epochs,
            verbose=verbose,
            callbacks=callbacks,
            **kwargs,
        )
