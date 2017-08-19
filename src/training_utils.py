import tensorflow as tf
import shutil
import os
import time


def _is_early_stopping(losses, patience, index_to_watch):
    test_losses = [x[index_to_watch] for x in losses]
    if len(losses) > (patience + 4):
        # running average
        average = (test_losses[-(patience + 4)] +
                   test_losses[-(patience + 3)] +
                   test_losses[-(patience + 2)] +
                   test_losses[-(patience + 1)] +
                   test_losses[-patience])/5.0
        return test_losses[-1] > average
    else:
        return False


def fit(run, graph, ops, train_tfrecords, val_tfrecords, batch_size,
        num_epochs, steps_per_epoch, validation_steps, patience=10,
        initial_weights=None, warm=False, initial_epoch=1, verbose=True):
    """Fit the defined network.

    Arguments:
        run: An integer that determines a folder where logs and the fitted model
            will be saved.
        X_train: A numpy array of shape [n_train_samples, n_features]
            and of type 'float32'.
        Y_train: A numpy array of shape [n_train_samples, n_classes]
            and of type 'float32'.
        X_test: A numpy array of shape [n_test_samples, n_features]
            and of type 'float32'.
        Y_test: A numpy array of shape [n_test_samples, n_classes]
            and of type 'float32'.
        batch_size: An integer.
        num_epochs: An integer.
        validation_step: An integer, establishes the step when train and test
            logloss/accuracy will be computed and shown.
        patience: An integer, number of validation steps before early stopping if
            test logloss isn't improving.
        initial_weights: A dictionary of weights to initialize network with.
        warm: Boolean, if `True` then resume training from the previously
            saved model.
        verbose: Boolean, whether to print train and test logloss/accuracy
            during fitting.

    Returns:
        losses: A list of tuples containing train and test logloss/accuracy.
        is_early_stopped: Boolean, if `True` then fitting is stopped early.
    """

    # create folders for logging and saving
    dir_to_log = 'logs/run' + str(run)
    dir_to_save = 'saved/run' + str(run)
    if os.path.exists(dir_to_log) and not warm:
        shutil.rmtree(dir_to_log)
    if os.path.exists(dir_to_save) and not warm:
        shutil.rmtree(dir_to_save)
    if not os.path.exists(dir_to_save):
        os.makedirs(dir_to_save)

    sess = tf.Session(graph=graph)
    writer = tf.summary.FileWriter(dir_to_log, sess.graph)

    # get graph's ops
    data_init_op, predictions_op, log_loss_op, optimize_op,\
        grad_summaries_op, init_op, saver_op, assign_weights_op,\
        accuracy_op, summaries_op = ops

    if warm:
        saver_op.restore(sess, dir_to_save + '/model')
    else:
        sess.run(init_op)
        if initial_weights is not None:
            for w in initial_weights:
                op = assign_weights_op[w]
                sess.run(op, {'utilities/' + w: initial_weights[w]})

    losses = []
    is_early_stopped = False

    training_epochs = range(
        initial_epoch,
        initial_epoch + num_epochs
    )

    # initialize data sources
    data_dict = {
        'input_pipeline/train_file:0': train_tfrecords,
        'input_pipeline/val_file:0': val_tfrecords,
        'input_pipeline/batch_size:0': batch_size
    }
    sess.run(data_init_op, data_dict)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for epoch in training_epochs:

        start_time = time.time()
        running_loss, running_accuracy = 0.0, 0.0

        # at zeroth step also collect metadata
        run_options = tf.RunOptions(
            trace_level=tf.RunOptions.FULL_TRACE
        )
        run_metadata = tf.RunMetadata()
        _, batch_loss, batch_accuracy, summary, grad_summary = sess.run(
            [optimize_op, log_loss_op, accuracy_op, summaries_op, grad_summaries_op],
            options=run_options, run_metadata=run_metadata
        )
        writer.add_run_metadata(run_metadata, str(epoch))
        writer.add_summary(summary, epoch)
        writer.add_summary(grad_summary, epoch)
        running_loss += batch_loss
        running_accuracy += batch_accuracy

        # main training loop
        for step in range(1, steps_per_epoch):

            _, batch_loss, batch_accuracy = sess.run(
                [optimize_op, log_loss_op, accuracy_op]
            )

            running_loss += batch_loss
            running_accuracy += batch_accuracy

        test_loss, test_accuracy = evaluate(
            sess, validation_steps, log_loss_op, accuracy_op
        )
        train_loss = running_loss/steps_per_epoch
        train_accuracy = running_accuracy/steps_per_epoch

        if verbose:
            print('{0}  {1:.3f} {2:.3f} {3:.3f} {4:.3f}  {5:.3f}'.format(
                epoch, train_loss, test_loss,
                train_accuracy, test_accuracy, time.time() - start_time
            ))

        losses += [(epoch, train_loss, test_loss, train_accuracy, test_accuracy)]

        if _is_early_stopping(losses, patience, 2):
            is_early_stopped = True
            break

    coord.request_stop()
    coord.join(threads)

    saver_op.save(sess, dir_to_save + '/model')
    sess.close()

    return losses, is_early_stopped


def evaluate(sess, validation_steps, log_loss_op, accuracy_op):

    test_loss, test_accuracy = 0.0 , 0.0
    for i in range(validation_steps):
        batch_loss, batch_accuracy = sess.run(
            [log_loss_op, accuracy_op], {'control/is_training:0': False}
        )
        test_loss += batch_loss
        test_accuracy += batch_accuracy

    test_loss /= validation_steps
    test_accuracy /= validation_steps

    return test_loss, test_accuracy


def predict_proba(graph, X, run):
    """Predict classes with the fitted model.

    Arguments:
        X: A numpy array of shape [n_samples, n_features]
            and of type 'float32'.
        network_weights: A dictionary of weights to initialize
            network with or None.

    Returns:
        predictions: A numpy array of shape [n_samples, n_classes]
            and of type 'float32'.
    """
    sess = tf.Session(graph=graph)

    saver_op = graph.get_operation_by_name('utilities/saver')
    predictions_op = graph.get_operation_by_name('softmax/predictions')

    saver_op.restore(sess, '../saved/run' + str(run) + '/model')

    feed_dict = {'inputs/X:0': X, 'control/is_training:0': False}
    predictions = sess.run(predictions_op, feed_dict)
    sess.close()

    return predictions
