import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import levenberg_marquardt as lm

input_size = 20000
batch_size = 1000

x_train = np.linspace(-1, 1, input_size, dtype=np.float64)
y_train = np.sinc(10 * x_train)

x_train = tf.expand_dims(tf.cast(x_train, tf.float32), axis=-1)
y_train = tf.expand_dims(tf.cast(y_train, tf.float32), axis=-1)

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(input_size)
train_dataset = train_dataset.batch(batch_size).cache()
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(20, activation='tanh', input_shape=(1,)),
    tf.keras.layers.Dense(1, activation='linear')])

model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    loss=tf.keras.losses.MeanSquaredError())

model_wrapper = lm.ModelWrapper(
    tf.keras.models.clone_model(model))

model_wrapper.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=1.0),
    loss=lm.MeanSquaredError())

print("Train using Adam")
t1_start = time.perf_counter()
model.fit(train_dataset, epochs=1000)
t1_stop = time.perf_counter()
print("Elapsed time: ", t1_stop - t1_start)

print("\n_________________________________________________________________")
print("Train using Levenberg-Marquardt")
t2_start = time.perf_counter()
model_wrapper.fit(train_dataset, epochs=100)
t2_stop = time.perf_counter()
print("Elapsed time: ", t2_stop - t2_start)

print("\n_________________________________________________________________")
print("Plot results")
plt.plot(x_train, y_train, 'b-', label="reference")
plt.plot(x_train, model.predict(x_train), 'g--', label="adam")
plt.plot(x_train, model_wrapper.predict(x_train), 'r--', label="lm")
plt.legend()
plt.show()
