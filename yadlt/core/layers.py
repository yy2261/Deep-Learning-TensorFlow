"""Collection of common layers."""

import tensorflow as tf


class Layers(object):
    """Collection of computational NN layers."""

    @staticmethod
    def linear(prev_layer, out_dim, name="linear"):
        """Create a linear fully-connected layer.

        Parameters
        ----------

        prev_layer : tf.Tensor
            Last layer's output tensor.

        out_dim : int
            Number of output units.

        Returns
        -------

        tuple (
            tf.Tensor : Linear output tensor
            tf.Tensor : Linear weights variable
            tf.Tensor : Linear biases variable
        )
        """
        with tf.name_scope(name):
            in_dim = prev_layer.get_shape()[1].value
            W = tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=0.1))
            b = tf.Variable(tf.constant(0.1, shape=[out_dim]))
            out = tf.add(tf.matmul(prev_layer, W), b)
            return (out, W, b)

    @staticmethod
    def regularization(variables, regtype, regcoef, name="regularization"):
        """Compute the regularization tensor.

        Parameters
        ----------

        variables : list of tf.Variable
            List of model variables.

        regtype : str
            Type of regularization. Can be ["none", "l1", "l2"]

        regcoef : float,
            Regularization coefficient.

        name : str, optional (default = "regularization")
            Name for the regularization op.

        Returns
        -------

        tf.Tensor : Regularization tensor.
        """
        with tf.name_scope(name):
            if regtype != 'none':
                regs = tf.constant(0.0)
                for v in variables:
                    if regtype == 'l2':
                        regs = tf.add(regs, tf.nn.l2_loss(v))
                    elif regtype == 'l1':
                        regs = tf.add(regs, tf.reduce_sum(tf.abs(v)))

                return tf.multiply(regcoef, regs)
            else:
                return None


class Evaluation(object):
    """Collection of evaluation methods."""

    @staticmethod
    def accuracy(mod_y, ref_y, summary=True, name="accuracy"):
        """Accuracy computation op.

        Parameters
        ----------

        mod_y : tf.Tensor
            Model output tensor.

        ref_y : tf.Tensor
            Reference input tensor.

        summary : bool, optional (default = True)
            Whether to save tf summary for the op.

        Returns
        -------

        tf.Tensor : accuracy op. tensor
        """
        with tf.name_scope(name):
            mod_pred = tf.argmax(mod_y, 1)
            correct_pred = tf.equal(mod_pred, tf.argmax(ref_y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
            if summary:
                tf.summary.scalar('accuracy', accuracy)
            return accuracy

    @staticmethod
    def precision(mod_y, ref_y, summary=True, name="precision"):
    	with tf.name_scope(name):
    		predictions = tf.argmax(mod_y, 1)
    		actuals = tf.argmax(ref_y, 1)
    		
    		ones_likes = tf.ones_like(actuals)
    		zeros_likes = tf.zeros_like(actuals)

    		tp = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(actuals, zeros_likes), tf.equal(predictions, zeros_likes)), tf.float32))
    		fp = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(actuals, ones_likes), tf.equal(predictions, zeros_likes)), tf.float32))
    		precision = tf.div(tp, tf.add(tp, fp))
    		if summary:
    			tf.summary.scalar('precision', precision)
    		return precision


    @staticmethod
    def recall(mod_y, ref_y, summary=True, name="recall"):
        with tf.name_scope(name):
                predictions = tf.argmax(mod_y, 1)
                actuals = tf.argmax(ref_y, 1)

                ones_likes = tf.ones_like(actuals)
                zeros_likes = tf.zeros_like(actuals)

                tp = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(actuals, zeros_likes), tf.equal(predictions, zeros_likes)), tf.float32))
                fn = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(actuals, zeros_likes), tf.equal(predictions, ones_likes)), tf.float32))
                recall = tf.div(tp, tf.add(tp, fn))
                if summary:
                        tf.summary.scalar('recall', recall)
                return recall
