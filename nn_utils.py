import tensorflow as tf


def _get_data_old(num_classes):

    X_train = tf.Variable(
        tf.placeholder(tf.float32, [None, 224, 224, 3], 'X_train'),
        trainable=False, collections=[],
        validate_shape=False, expected_shape=[None, 224, 224, 3]
    )
    Y_train = tf.Variable(
        tf.placeholder(tf.float32, [None, num_classes], 'Y_train'),
        trainable=False, collections=[],
        validate_shape=False, expected_shape=[None, num_classes]
    )
    batch_size = tf.Variable(
        tf.placeholder(tf.int32, [], 'batch_size'),
        trainable=False, collections=[]
    )
    init = tf.variables_initializer([X_train, Y_train, batch_size])

    # three values that you need to tweak
    min_after_dequeue = 10000
    capacity = min_after_dequeue + 3*64
    num_threads = 2

    x_batch, y_batch = tf.train.shuffle_batch(
        [X_train, Y_train], batch_size, capacity, min_after_dequeue,
        num_threads, enqueue_many=True,
        shapes=[[224, 224, 3], [num_classes]]
    )
    return init, x_batch, y_batch


def _get_data(num_classes):

    filename_queue = tf.train.string_input_producer(['/home/ubuntu/data/train.tfrecords'])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = {
        'image_raw': tf.FixedLenFeature([], tf.string),
        'target': tf.FixedLenFeature([], tf.int64)
    }

    features = tf.parse_single_example(serialized_example, features)

    image = tf.decode_raw(features['image_raw'], tf.uint8)
    target = tf.cast(features['target'], tf.int32)

    image_shape = tf.stack([224, 224, 3])
    image = tf.reshape(image, image_shape)
    image = tf.cast(image, tf.float32)

    mean = tf.constant([0.485, 0.456, 0.406], tf.float32, [3])
    std = tf.constant([0.229, 0.224, 0.225], tf.float32, [3])
    image = tf.subtract(image, mean)
    image = tf.realdiv(image, std)

    # three values that you need to tweak
    min_after_dequeue = 10000
    capacity = min_after_dequeue + 3*64
    num_threads = 2
    batch_size = 64

    x_batch, y_batch = tf.train.shuffle_batch(
        [image, target], batch_size, capacity,
        min_after_dequeue, num_threads
    )

    y_batch = tf.one_hot(y_batch, num_classes, axis=1, dtype=tf.float32)

    return x_batch, y_batch


def _add_summaries():
    summaries = []
    trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

    for v in trainable_vars:
        summaries += [tf.summary.histogram(v.name[:-2] + '_hist', v)]

    return tf.summary.merge(summaries)


def _is_early_stopping(losses, patience, index_to_watch):
    test_losses = [x[index_to_watch] for x in losses]
    if len(losses) > (patience + 4):
        average = (test_losses[-(patience + 4)] +
                   test_losses[-(patience + 3)] +
                   test_losses[-(patience + 2)] +
                   test_losses[-(patience + 1)] +
                   test_losses[-patience])/5.0
        return test_losses[-1] > average
    else:
        return False


def _assign_weights():

    assign_weights_dict = {}
    model_vars = tf.get_collection(tf.GraphKeys.MODEL_VARIABLES)

    for v in model_vars:
        assign_weights_dict[v.name] = v.assign(
            tf.placeholder(tf.float32, v.shape, v.name[:-2])
        )

    return assign_weights_dict
