import tensorflow as tf
import shutil
import os
import time
from utils import _add_summaries, _is_early_stopping, _assign_weights, _get_data
from parts import _mapping, _add_weight_decay


class SqueezeNet:
    def __init__(self, optimizer, weight_decay=None, image_size=224, num_classes=1000):
        """Create the SqueezeNet computational graph.

        Arguments:
            optimizer: A Tensorflow optimizer.
            num_classes: An integer.
            weight_decay: A scalar or None.
        """

        self.graph = tf.Graph()
        with self.graph.as_default():
            
            with tf.variable_scope('control'):
                is_training = tf.placeholder_with_default(True, [], 'is_training')
                
            with tf.device('/cpu:0'), tf.variable_scope('input_pipeline'):                
                self.data_init, x_batch, y_batch = _get_data(
                    num_classes, image_size, is_training
                )
                
            with tf.variable_scope('inputs'):
                X = tf.placeholder_with_default(x_batch, [None, image_size, image_size, 3], 'X')
                Y = tf.placeholder_with_default(y_batch, [None, num_classes], 'Y')
            
            with tf.variable_scope('preprocessing'):
                f255 = tf.constant(255.0, tf.float32, [])
                mean = tf.constant([0.485, 0.456, 0.406], tf.float32, [3])
                std = tf.constant([0.229, 0.224, 0.225], tf.float32, [3])
                X /= f255
                X -= mean
                X /= std

            logits = _mapping(X, num_classes, is_training)

            with tf.variable_scope('softmax'):
                self.predictions = tf.nn.softmax(logits)

            with tf.variable_scope('log_loss'):
                self.log_loss = tf.losses.softmax_cross_entropy(Y, logits)

            if weight_decay is not None:
                _add_weight_decay(weight_decay)

            with tf.variable_scope('total_loss'):
                total_loss = tf.losses.get_total_loss()

            with tf.variable_scope('optimizer'):
                grads_and_vars = optimizer.compute_gradients(total_loss)
                self.optimize = optimizer.apply_gradients(grads_and_vars)

            self.grad_summaries = tf.summary.merge(
                [tf.summary.histogram(v.name[:-2] + '_grad_hist', g)
                 for g, v in grads_and_vars]
            )

            with tf.variable_scope('utilities'):
                self.init = tf.global_variables_initializer()
                self.saver = tf.train.Saver()
                self.assign_weights = _assign_weights()
                is_equal = tf.equal(tf.argmax(self.predictions, 1), tf.argmax(Y, 1))
                self.accuracy = tf.reduce_mean(tf.cast(is_equal, tf.float32))

            self.merged = _add_summaries()

        self.graph.finalize()

    def fit(self, run, train_tfrecords, val_tfrecords, batch_size, 
            num_epochs, steps_per_epoch, validation_steps, patience=10,
            initial_weights=None, warm=False, verbose=True):
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

        dir_to_log = '../logs/run' + str(run)
        dir_to_save = '../saved/run' + str(run)
        if os.path.exists(dir_to_log) and not warm:
            shutil.rmtree(dir_to_log)
        if os.path.exists(dir_to_save) and not warm:
            shutil.rmtree(dir_to_save)
        if not os.path.exists(dir_to_save):
            os.makedirs(dir_to_save)

        sess = tf.Session(graph=self.graph)
        self.writer = tf.summary.FileWriter(dir_to_log, sess.graph)
        
        if warm:
            self.saver.restore(sess, dir_to_save + '/model')
        else:
            sess.run(self.init)
            if initial_weights is not None:
                for w in initial_weights:
                    op = self.assign_weights[w]
                    sess.run(op, {'utilities/' + w: initial_weights[w]})
            
        losses = []
        self.steps_per_epoch = steps_per_epoch
        self.validation_steps = validation_steps
        self.verbose = verbose
        is_early_stopped = False
        self.steps_trained = 0 if not warm else self.steps_trained
        training_steps = range(
            self.steps_trained + 1,
            self.steps_trained + num_epochs*steps_per_epoch + 1
        )
        
        data_dict = {
            'input_pipeline/train_file:0': train_tfrecords,
            'input_pipeline/val_file:0': val_tfrecords,
            'input_pipeline/batch_size:0': batch_size
        }
        sess.run(self.data_init, data_dict)
        
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        
        self.start = time.time()
        self.running_loss, self.running_accuracy = 0.0, 0.0
        
        for step in training_steps:
            if (step % steps_per_epoch != 0):
                _, batch_loss, batch_accuracy = sess.run(
                    [self.optimize, self.log_loss, self.accuracy]
                )
                self.running_loss += batch_loss
                self.running_accuracy += batch_accuracy
            else:
                losses += [self._evaluate(step, sess)]
                if _is_early_stopping(losses, patience, 1):
                    is_early_stopped = True
                    break
        
        coord.request_stop()
        coord.join(threads)
        
        self.saver.save(sess, dir_to_save + '/model')
        self.run = run
        self.steps_trained = step
        sess.close()

        return losses, is_early_stopped

    def predict_proba(self, X, network_weights=None):
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
        sess = tf.Session(graph=self.graph)

        if network_weights is not None:
            for w in network_weights:
                op = self.assign_weights[w]
                sess.run(op, {'utilities/' + w: network_weights[w]})
        else:
            self.saver.restore(sess, '../saved/run' + str(self.run) + '/model')

        feed_dict = {'inputs/X:0': X, 'control/is_training:0': False}
        predictions = sess.run(self.predictions, feed_dict)
        sess.close()
        return predictions

    def _evaluate(self, step, sess):

        entry = '{0:.2f}'.format(step/self.steps_per_epoch)
        run_options = tf.RunOptions(
            trace_level=tf.RunOptions.FULL_TRACE
        )
        run_metadata = tf.RunMetadata()
        _, batch_loss, batch_accuracy, train_summary, train_grad_summary = sess.run(
            [self.optimize, self.log_loss, self.accuracy, self.merged, self.grad_summaries],
            options=run_options, run_metadata=run_metadata
        )
        self.writer.add_run_metadata(run_metadata, entry)
        self.writer.add_summary(train_summary, step)
        self.writer.add_summary(train_grad_summary, step)
        self.running_loss += batch_loss
        self.running_accuracy += batch_accuracy

        test_loss, test_accuracy = 0.0 , 0.0
        for i in range(self.validation_steps):
            batch_loss, batch_accuracy = sess.run(
                [self.log_loss, self.accuracy], {'control/is_training:0': False}
            )
            test_loss += batch_loss
            test_accuracy += batch_accuracy

        train_loss = self.running_loss/self.steps_per_epoch
        train_accuracy = self.running_accuracy/self.steps_per_epoch
        test_loss /= self.validation_steps
        test_accuracy /= self.validation_steps

        if self.verbose:
            print('{0}  {1:.3f} {2:.3f} {3:.3f} {4:.3f}  {5:.3f}'.format(
                entry, train_loss, test_loss,
                train_accuracy, test_accuracy, time.time() - self.start
            ))
        
        self.start = time.time()
        self.running_loss, self.running_accuracy = 0.0, 0.0
        
        return train_loss, test_loss, train_accuracy, test_accuracy
