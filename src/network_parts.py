import tensorflow as tf


def _nonlinearity(X):
    return tf.nn.relu(X, name='ReLU')


def _dropout(X, rate, is_training):
    keep_prob = tf.constant(
        1.0 - rate, tf.float32,
        [], 'keep_prob'
    )
    result = tf.cond(
        is_training,
        lambda: tf.nn.dropout(X, keep_prob),
        lambda: tf.identity(X),
        name='dropout'
    )
    return result


def _conv(X, filters, kernel, strides=(1, 1), padding='VALID', trainable=False):

    in_channels = X.shape.as_list()[-1]

    K = tf.get_variable(
        'kernel', [kernel[0], kernel[1], in_channels, filters],
        tf.float32, trainable=trainable
    )

    b = tf.get_variable(
        'bias', [filters], tf.float32,
        tf.zeros_initializer(), trainable=trainable
    )

    tf.add_to_collection(tf.GraphKeys.MODEL_VARIABLES, K)
    tf.add_to_collection(tf.GraphKeys.MODEL_VARIABLES, b)

    return tf.nn.bias_add(
        tf.nn.conv2d(X, K, [1, strides[0], strides[1], 1], padding), b
    )


def _fire(X, squeeze, expand, trainable=False):

    with tf.variable_scope('squeeze'):
        result = _conv(X, squeeze, (1, 1), trainable=trainable)
    result = _nonlinearity(result)

    with tf.variable_scope('expand1x1'):
        left = _conv(
            result, expand, (1, 1),
            trainable=trainable
        )

    with tf.variable_scope('expand3x3'):
        right = _conv(
            result, expand, (3, 3),
            padding='SAME', trainable=trainable
        )

    left = _nonlinearity(left)
    right = _nonlinearity(right)
    return tf.concat([left, right], -1)


def _global_average_pooling(X):
    return tf.reduce_mean(
        X, axis=[1, 2],
        name='global_average_pooling'
    )


def _max_pooling(X):
    return tf.nn.max_pool(
        X, [1, 3, 3, 1], [1, 2, 2, 1], 'VALID',
        name='max_pooling'
    )


def _mapping(X, num_classes, is_training):

    fire_init = tf.contrib.layers.xavier_initializer_conv2d()
    with tf.variable_scope('features', initializer=fire_init):

        with tf.variable_scope('conv1'):
            result = _conv(X, 64, (3, 3), strides=(2, 2))
        result = _nonlinearity(result)
        result = _max_pooling(result)

        with tf.variable_scope('fire2'):
            result = _fire(result, 16, 64)
        with tf.variable_scope('fire3'):
            result = _fire(result, 16, 64)
        result = _max_pooling(result)

        with tf.variable_scope('fire4'):
            result = _fire(result, 32, 128)
        with tf.variable_scope('fire5'):
            result = _fire(result, 32, 128)
        result = _max_pooling(result)

        with tf.variable_scope('fire6'):
            result = _fire(result, 48, 192)
        with tf.variable_scope('fire7'):
            result = _fire(result, 48, 192)
        with tf.variable_scope('fire8'):
            result = _fire(result, 64, 256)
        with tf.variable_scope('fire9'):
            result = _fire(result, 64, 256)

    classifier_init = tf.random_normal_initializer(mean=0.0, stddev=0.01)
    with tf.variable_scope('classifier', initializer=classifier_init):

        result = _dropout(result, 0.5, is_training)
        with tf.variable_scope('conv10'):
            result = _conv(result, num_classes, (1, 1), trainable=True)
        result = _nonlinearity(result)

        logits = _global_average_pooling(result)
        return logits


def _add_weight_decay(weight_decay):

    weight_decay = tf.constant(
        weight_decay, tf.float32,
        [], 'weight_decay'
    )

    trainable = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    kernels = [v for v in trainable if 'kernel' in v.name]

    for K in kernels:
        l2_loss = tf.multiply(
            weight_decay, tf.nn.l2_loss(K), name='l2_loss'
        )
        tf.losses.add_loss(l2_loss)
