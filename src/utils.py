import tensorflow as tf


def _get_data(num_classes, image_size, is_training):
    
    train_file = tf.Variable(
        tf.placeholder(tf.string, [], 'train_file'),
        trainable=False, collections=[]
    )
    val_file = tf.Variable(
        tf.placeholder(tf.string, [], 'val_file'),
        trainable=False, collections=[]
    )
    batch_size = tf.Variable(
        tf.placeholder(tf.int32, [], 'batch_size'),
        trainable=False, collections=[]
    )
    init_data = tf.variables_initializer([train_file, val_file, batch_size])

    train_x_batch, train_y_batch = _get_batch(
        num_classes, train_file, image_size, batch_size
    )
    val_x_batch, val_y_batch = _get_val_batch(
        num_classes, val_file, image_size, batch_size
    )

    x_batch, y_batch = tf.cond(
        is_training,
        lambda: (train_x_batch, train_y_batch),
        lambda: (val_x_batch, val_y_batch)
    )
        
    return init_data, x_batch, y_batch


def _get_batch(num_classes, tfrecords_file_name, image_size, batch_size):

    images, targets = _get_images_and_targets(tfrecords_file_name, image_size)

    # three values that you need to tweak
    min_after_dequeue = 1000
    num_threads = 4
    capacity = min_after_dequeue + (num_threads + 2)*64
    # 64 is a typical batch size

    x_batch, y_batch = tf.train.shuffle_batch(
        [images, targets], batch_size, capacity,
        min_after_dequeue, num_threads, enqueue_many=True
    )

    y_batch = tf.one_hot(y_batch, num_classes, axis=1, dtype=tf.float32)
    return x_batch, y_batch


def _get_val_batch(num_classes, tfrecords_file_name, image_size, batch_size):

    images, targets = _get_images_and_targets(tfrecords_file_name, image_size)

    num_threads = 1
    capacity = 256

    x_batch, y_batch = tf.train.batch(
        [images, targets], batch_size, 
        num_threads, capacity,
        enqueue_many=True
    )

    y_batch = tf.one_hot(y_batch, num_classes, axis=1, dtype=tf.float32)
    return x_batch, y_batch


def _get_images_and_targets(tfrecords_file_name, image_size):

    filename_queue = tf.train.string_input_producer([tfrecords_file_name])
    reader = tf.TFRecordReader()
    enqueue_many_size = 10
    _, serialized_example = reader.read_up_to(filename_queue, enqueue_many_size)

    features = {
        'image_raw': tf.FixedLenFeature([], tf.string),
        'target': tf.FixedLenFeature([], tf.int64)
    }

    features = tf.parse_example(serialized_example, features)

    images = tf.decode_raw(features['image_raw'], tf.uint8)
    targets = tf.cast(features['target'], tf.int32)
    
    n_images = tf.shape(images)[0]
    image_shape = tf.stack([n_images, image_size, image_size, 3])
    images = tf.reshape(images, image_shape)
    images = tf.cast(images, tf.float32)

    return images, targets


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
