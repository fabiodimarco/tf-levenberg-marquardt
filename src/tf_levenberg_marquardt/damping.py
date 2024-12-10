import tensorflow as tf
from tensorflow import Tensor


class DampingAlgorithm:
    """Default Levenberg-Marquardt damping algorithm.

    This is used inside the Trainer as a generic class. Many damping algorithms can be
    implemented using the same interface.
    """

    def __init__(
        self,
        starting_value: float = 1e-3,
        dec_factor: float = 0.1,
        inc_factor: float = 10.0,
        min_value: float = 1e-10,
        max_value: float = 1e10,
        adaptive_scaling: bool = False,
        fletcher: bool = False,
    ):
        """Initializes `DampingAlgorithm` instance.

        Args:
          starting_value: (Optional) Used to initialize the Trainer internal
            damping_factor.
          dec_factor: (Optional) Used in the train_step to decrease the damping_factor
            when new_loss < loss.
          inc_factor: (Optional) Used in the train_step to increase the damping_factor
            when new_loss >= loss.
          min_value: (Optional) Used as a lower bound for the damping_factor. Higher
            values improve numerical stability in the resolution of the linear system,
            at the cost of slower convergence.
          max_value: (Optional) Used as an upper bound for the damping_factor, and as
            a condition to stop the training process.
          adaptive_scaling: Bool (Optional) Scales the damping_factor adaptively
            multiplying it with max(diagonal(JJ)).
          fletcher: Bool (Optional) Replace the identity matrix with the diagonal of the
            Gauss-Newton Hessian approximation, so that there is larger movement along
            the directions where the gradient is smaller. This avoids slow convergence
            in the direction of small gradient.
        """
        self.starting_value = starting_value
        self.dec_factor = dec_factor
        self.inc_factor = inc_factor
        self.min_value = min_value
        self.max_value = max_value
        self.adaptive_scaling = adaptive_scaling
        self.fletcher = fletcher

    def init_step(self, damping_factor: Tensor, loss: Tensor) -> Tensor:
        return damping_factor

    def decrease(self, damping_factor: Tensor, loss: Tensor) -> Tensor:
        return tf.math.maximum(damping_factor * self.dec_factor, self.min_value)

    def increase(self, damping_factor: Tensor, loss: Tensor) -> Tensor:
        return tf.math.minimum(damping_factor * self.inc_factor, self.max_value)

    def stop_training(self, damping_factor: Tensor, loss: Tensor) -> Tensor:
        return damping_factor >= self.max_value

    def apply(self, damping_factor: Tensor, JJ: Tensor) -> Tensor:
        if self.fletcher:
            damping = tf.linalg.tensor_diag(tf.linalg.diag_part(JJ))
        else:
            damping = tf.eye(tf.shape(JJ)[0], dtype=JJ.dtype)

        scaler = 1.0
        if self.adaptive_scaling:
            scaler = tf.math.reduce_max(tf.linalg.diag_part(JJ))

        damping = tf.scalar_mul(scaler * damping_factor, damping)
        return tf.add(JJ, damping)
