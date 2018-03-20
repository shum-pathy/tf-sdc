
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from scipy import misc
import csv

training_file = "./training-file.txt"
validation_file = "./validation-file.txt"
logs_path = "./logs"

tf.reset_default_graph()

def cnn_model_fn(features, labels, mode):

    input_layer = tf.reshape(features["x"], [-1, 60, 90, 3])

    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[2, 2],
        padding="same",
        activation=tf.nn.relu)

    pool1 = tf.layers.max_pooling2d(
        inputs=conv1,
        pool_size=[2, 2],
        strides=[2, 2],
        padding="same"
    )

    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[2, 2],
        padding="same",
        activation=tf.nn.relu
    )

    pool2 = tf.layers.max_pooling2d(
        inputs=conv2,
        pool_size=[2, 2],
        strides=[2, 2],
        padding="same"
    )

    conv3 = tf.layers.conv2d(
        inputs=pool2,
        filters=128,
        kernel_size=[2, 2],
        padding="same",
        activation=tf.nn.relu
    )

    pool3 = tf.layers.max_pooling2d(
        inputs=conv3,
        pool_size=[2, 2],
        strides=[2, 2],
        padding="same"
    )

    # Flatten tensor into a batch of vectors
    # Input Tensor Shape: [batch_size, 7, 7, 64]
    # Output Tensor Shape: [batch_size, 7 * 7 * 64]
    flat = tf.reshape(pool3, [-1, 8 * 12 * 128])

    # Dense Layer
    # Densely connected layer with 1024 neurons
    # Input Tensor Shape: [batch_size, 7 * 7 * 64]
    # Output Tensor Shape: [batch_size, 1024]
    dense1 = tf.layers.dense(inputs=flat, units=512, activation=tf.nn.relu)

    # Add dropout operation; 0.6 probability that element will be kept
    dropout1 = tf.layers.dropout(
        inputs=dense1, rate=0.3, training=mode == tf.estimator.ModeKeys.TRAIN)

    dense2 = tf.layers.dense(inputs=dropout1, units=64, activation=tf.nn.relu)

    # Add dropout operation; 0.6 probability that element will be kept
    dropout2 = tf.layers.dropout(
        inputs=dense2, rate=0.1, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Input Tensor Shape: [batch_size, 64]
    # Output Tensor Shape: [batch_size, 10]
    logits = tf.layers.dense(inputs=dropout2, units=11)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.00005)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


summary_op = tf.summary.merge_all()

init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)
    writer = tf.summary.FileWriter(logs_path, graph=sess.graph)

    training_image_list = []
    training_category_list = []
    validation_image_list = []
    validation_category_list = []

    trainfile = open(training_file)
    reader = csv.reader(trainfile, delimiter=',')
    for row in reader:
        image_file_name = row[0]
        image_file_cat = row[1]
        img = (misc.imread(image_file_name)).astype(np.float32)
        training_image_list.append(img)
        training_category_list.append(image_file_cat)
    trainfile.close()

    vf = open(validation_file)
    reader = csv.reader(vf, delimiter=',')
    for row in reader:
        image_file_name = row[0]
        image_file_cat = row[1]
        img = (misc.imread(image_file_name)).astype(np.float32)
        validation_image_list.append(img)
        validation_category_list.append(image_file_cat)
    vf.close()

    train_data = np.asarray(training_image_list, dtype=np.float32)
    train_labels = np.asarray(training_category_list, dtype=np.int32)
    eval_data = np.asarray(validation_image_list, dtype=np.float32)
    eval_labels = np.asarray(validation_category_list, dtype=np.int32)

    summary_hook = tf.train.SummarySaverHook(
        save_steps=100,
        output_dir=logs_path,
        scaffold=tf.train.Scaffold(summary_op)
    )

    # Create the Estimator
    classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn,
        model_dir="./checkpoint",
    )

    # Set up logging for predictions
    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=128,
        num_epochs=None,
        shuffle=True,
    )

    classifier.train(
        input_fn=train_input_fn,
        steps=2000,
        hooks=[summary_hook]
    )
