import time

import keras
import tensorflow as tf

import tf_levenberg_marquardt as lm

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = tf.cast(x_train / 255.0, dtype=tf.float32)
x_test = tf.cast(x_test / 255.0, dtype=tf.float32)

x_train = tf.expand_dims(x_train, axis=-1)
x_test = tf.expand_dims(x_test, axis=-1)

y_train = tf.cast(y_train, dtype=tf.float32)
y_test = tf.cast(y_test, dtype=tf.float32)

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(60000)
train_dataset = train_dataset.batch(6000).cache()
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)


model = keras.models.Sequential(
    [
        keras.layers.Conv2D(
            filters=8,
            kernel_size=4,
            strides=2,
            padding='valid',
            activation='tanh',
            input_shape=(28, 28, 1),
        ),
        keras.layers.Conv2D(
            filters=4, kernel_size=4, strides=2, padding='valid', activation='tanh'
        ),
        keras.layers.Conv2D(
            filters=4, kernel_size=2, strides=1, padding='valid', activation='tanh'
        ),
        keras.layers.Conv2D(
            filters=4, kernel_size=2, strides=1, padding='valid', activation='tanh'
        ),
        keras.layers.Conv2D(
            filters=4, kernel_size=2, strides=1, padding='valid', activation='tanh'
        ),
        keras.layers.Flatten(),
        keras.layers.Dense(10, activation='linear'),
    ]
)

model.summary()

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.01),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
)

model_wrapper = lm.model.ModelWrapper(keras.models.clone_model(model))

model_wrapper.compile(
    optimizer=keras.optimizers.SGD(learning_rate=0.1),
    loss=lm.loss.SparseCategoricalCrossentropy(from_logits=True),
    solve_method='solve',
    metrics=['accuracy'],
)

print('Train using Adam')
t1_start = time.perf_counter()
model.fit(train_dataset, epochs=200)
t1_stop = time.perf_counter()
print('Elapsed time: ', t1_stop - t1_start)

print('\nTrain using Levenberg-Marquardt')
t2_start = time.perf_counter()
model_wrapper.fit(train_dataset, epochs=10)
t2_stop = time.perf_counter()
print('Elapsed time: ', t2_stop - t2_start)

print('\nTest set results')

test_loss, test_acc = model.evaluate(x=x_test, y=y_test, verbose=0)
print('adam - test_loss: %f - test_accuracy: %f' % (test_loss, test_acc))

test_loss, test_acc = model_wrapper.evaluate(x=x_test, y=y_test, verbose=0)
print('lm - test_loss: %f - test_accuracy: %f' % (test_loss, test_acc))
