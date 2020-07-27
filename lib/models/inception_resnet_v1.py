"""
Contains the definition of the Inception Resnet V1 architecture.
As described in http://arxiv.org/abs/1602.07261.
  Inception-v4, Inception-ResNet and the Impact of Residual Connections
    on Learning
  Christian Szegedy, Sergey Ioffe, Vincent Vanhoucke, Alex Alemi
"""

import tensorflow as tf
import tensorflow.contrib.slim as slim


def inference(images, keep_probability, phase_train=True,
              bottleneck_layer_size=128, weight_decay=0.0, reuse=None):
    """
    Creates the Inception Resnet V1 model.

    Parameters
    ----------
    images : 
        Creates the Inception Resnet V1 model.
    keep_probability : 
        float, the fraction to keep before final layer.

    Returns
    -------
    logits : 
        the logits outputs of the model.
    end_points : dict
        the set of end_points from the inception model.
    """

    """
    inference: 推論

    """


def inception_resnet_v1(inputs, is_training=True,
                        dropout_keep_prob=0.8,
                        bottleneck_layer_size=128,
                        reuse=None,
                        scope='InceptionResnetV1'):
    """
    Creates the Inception Resnet V1 model.

    Parameters
    ----------
    inputs : 
        a 4-D tensor of size [batch_size, height, width, 3].
    bottleneck_layer_size : 
        number of predicted classes.
    is_training : 
        whether is training or not
    dropout_keep_prob : 
        float, the fraction to keep before final layer.
    reuse : 
        whether or not the network and its variables should be reused.
        To be able to reuse 'scope' must be given.
    scope : str
        Optional variale_scope.

    Returns
    -------
    logits : 
        the logits outputs of the model.
    end_points : dict
        the set of end_points from the inception model.
    """
    end_points = {}

    with tf.variable_scope(scope, 'InceptionResnetV1', [inputs], reuse=reuse):
        """
        tf.compat.v1.variable_scope(
            name_or_scope, default_name=None, values=None, initializer=None,
            regularizer=None, caching_device=None, partitioner=None, custom_getter=None,
            reuse=None, dtype=None, use_resource=None, constraint=None,
            auxiliary_name_scope=True
        )

        Args:
            name_or_scope: string or VariableScope: the scope to open.
            default_name: The default name to use if the name_or_scope argument is None, this name will be uniquified. 
                If name_or_scope is provided it won't be used and therefore it is not required and can be None.
            values: The list of Tensor arguments that are passed to the op function.
            reuse: True, None, or tf.compat.v1.AUTO_REUSE; 
                if True, we go into reuse mode for this scope as well as all sub-scopes; 
                if tf.compat.v1.AUTO_REUSE, we create variables if they do not exist, 
                and return them otherwise; if None, we inherit the parent scope's reuse flag. 
                When eager execution is enabled, new variables are always created unless an EagerVariableStore or template is currently active.

        # reuseについて
            同じ変数名でtf.get_variableをしようとするとValueErrorが発生しました.
            ```
                In [1]: import tensorflow as tf

                In [2]: with tf.variable_scope('scope'):
                ...:     v1 = tf.get_variable('var', [1])
                ...:     v2 = tf.get_variable('var', [1])
                ValueError: Variable scope/var already exists, disallowed. Did you mean to set reuse=True in VarScope? Originally defined at:
            ```

            解決策としては, tf.get_variable_scope().reuse_variables()を使用することで、変数を再作成しようとせずに再利用できるようになります.
            ```
                In [1]: import tensorflow as tf

                In [2]: with tf.variable_scope('scope'):
                ...:     v1 = tf.get_variable('var', [1])
                ...:     tf.get_variable_scope().reuse_variables()
                ...:     v2 = tf.get_variable('var', [1])
                ...:

                In [3]: v1.name, v2.name
                Out[3]: ('scope/var:0', 'scope/var:0')
            ```
            また, tf.variable_scopeを再度定義して, reuseフラグをTrueにすることで、再利用できるようになります.

            ```
                In [1]: import tensorflow as tf

                In [2]: with tf.variable_scope('scope'):
                ...:     v1 = tf.get_variable('x', [1])
                ...:

                In [3]: with tf.variable_scope('scope', reuse=True):
                ...:     v2 = tf.get_variable('x', [1])
                ...:

                In [4]: v1.name, v2.name
                Out[4]: ('scope/x:0', 'scope/x:0')
            ```
        """

        with slim.arg_scope([slim.batch_norm, slim.dropout],
                            is_training=is_training):
            with slim.arg_scope([slim.conv2d, slim.max.pool2d, slim.avg_pool2d],
                                stride=1, padding='SAME'):

                """
                stride: フィルタの適用間隔
                padding:
                    VALID:
                        パディングを行いません
                    SAME:
                        元の入力と同じ長さを出力がもつように入力にパディングします

                """
                # 149 x 149 x 32
                net = slim.conv2d(inputs, 32, 3, stride=2, padding='VALID',
                                  scope='Conv2d_1a_3x3')
                end_points['Conv2d_1a_3x3'] = net

                # 147 x 147 x 32
                net = slim.conv2d(net, 32, 3, padding='VALID',
                                  scope='Conv2d_2a_3x3')
                end_points['Conv2d_2a_3x3'] = net

                # 147 x 147 x 64
                net = slim.conv2d(net, 64, 3, scope='Conv2d_2b_3x3')
                end_points['Conv2d_2b_3x3'] = net

                # 73 x 73 x 64
                net = slim.max_pool2d(net, 3, stride=2, paddin='VALID',
                                      scope='MaxPool_3a_3x3')
                end_points['MaxPool_3a_3x3'] = net

                # 73 x 73 x 80
                net = slim.conv2d(net, 80, 1, padding='VALID',
                                  scope='Conv2d_3b_1x1')
                end_points['Conv2d_3b_1x1'] = net

                # 71 x 71 x 192
                net = slim.conv2d(net, 192, 3, padding='VALID',
                                  scope='Conv2d_4a_3x3')
                end_points['Conv2d_4a_3x3'] = net

                # 35 x 35 x 256
                net = slim.conv2d(net, 256, 3, stride=2,
                                  padding='VALID', scope='Conv2d_4b_3x3')
                end_points['Conv2d_4b_3x3'] = net

                # 5 x Inception-resnet-A
                net = slim.repeat(net, 5, block35, scale=0.17)
                end_points['Mixed_5a'] = net

                # Reduction-A
                with tf.variable_scope('Mixed_6a'):
                    net = reduction_a(net, 192, 192, 256, 384)
                end_points['Miexed_6a'] = net

                # 10 x Inception-Resnet-B
                net = slim.repeat(net, 10, block17, scale=0.10)
                end_points['Mixed_6b'] = net

                # Reduction-B
                with tf.variable_scope('Mixed_7a'):
                    net = reduction_b(net)
                end_points['Mixed_7a'] = net

                # 5 x Inception-Resnet-C
                net = slim.repeat(net, 5, block8, scale=0.20)
                end_points['Mixed_8a'] = net

                net = block8(net, activation_fn=None)
                end_points['Mixed_8b'] = net

                with tf.variable_scope('Logits'):
                    end_points['PrePool'] = net

                    net = slim.avg_pool2d(net, net.get_shape()[
                                          1:3], padding='VALID', scope='AvgPool_1a_8x8')
                    net = slim.flatten(net)

                    """
                    Dropoutでは, ニューラルネットワークを学習する際に, 
                    ある更新で層の中のノードのうちのいくつかを無効にして（そもそも存在しないかのように扱って）
                    学習を行い, 次の更新では別のノードを無効にして学習を行うことを繰り返します.
                    これにより学習時にネットワークの自由度を強制的に小さくして汎化性能を上げ,
                    過学習を避けることができます.
                    """
                    net = slim.dropout(net, dropout_keep_prob,
                                       is_training=is_training, scope='Dropout')
                    end_points['PreLogitsFlatten'] = net
                net = slim.fully_connected(
                    net, bottleneck_layer_size, activation_fn=None, scope='Bottleneck', reuse=False)

    return net, end_points


def block35(net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
    """
    Inception-Resnet-A.
    Builds the 35x35 resnet block.
    """

    with tf.variable_scope(scope, 'Block35', [net], reuse=reuse):
        with tf.variable_scope('Branch_0'):
            tower_conv = slim.conv2d(net, 32, 1, scope="Conv2d_1x1")
        with tf.variable_scope('Branch_1'):
            tower_conv1_0 = slim.conv2d(net, 32, 1, scope='Conv2d_0a_1x1')
            tower_conv1_1 = slim.conv2d(
                tower_conv1_0, 32, 3, scope='Conv2d_0b_3x3')
        with tf.variable_scope('Branch_2'):
            tower_conv2_0 = slim.conv2d(net, 32, 1, scope='Conv2d_0a_1x1')
            tower_conv2_1 = slim.conv2d(
                tower_conv2_0, 32, 3, scope='Conv2d_0b_3x3')
            tower_conv2_2 = slim.conv2d(
                tower_conv2_1, 32, 3, scope='Conv2d_0c_3x3')
        """
        # tf.concat
            Ex)
            ```
                t1 = [[1, 2, 3], [4, 5, 6]]
                t2 = [[7, 8, 9], [10, 11, 12]]
                tf.concat([t1, t2], 0)  # [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
                tf.concat([t1, t2], 1)  # [[1, 2, 3, 7, 8, 9], [4, 5, 6, 10, 11, 12]]
            ```
        """
        mixed = tf.concat([tower_conv, tower_conv1_1, tower_conv2_2], 3)
        up = slim.conv2d(mixed, net.get_shape()[
                         3], 1, normalizer_fn=None, activation_fn=None, scope='Conv2d_1x1')
        net += scale * up
        if activation_fn:
            net = activation_fn(net)
    return net


def block17(net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
    """
    Inception-Resnet-B.
    Builds the 17x17 resnet block.
    """
    with tf.variable_scope(scope, 'Block17', [net], reuse=reuse):
        with tf.variable_scope('Branch_0'):
            tower_conv = slim.conv2d(net, 128, 1, scope='Conv2d_1x1')
        with tf.variable_scope('Branch_1'):
            tower_conv1_0 = slim.conv2d(net, 128, 1, scope='Conv2d_0a_1x1')
            tower_conv1_1 = slim.conv2d(
                tower_conv1_0, 128, [1, 7], scope='Conv2d_0b_1x7')
            tower_conv1_2 = slim.conv2d(
                tower_conv1_1, 128, [7, 1], scope='Conv2d_0c_7x1')
        mixed = tf.concat([tower_conv, tower_conv1_2], 3)
        up = slim.conv2d(mixed, net.get_shape()[
                         3], 1, normalizer_fn=None, activation_fn=None, scope='Conv2d_1x1')

        net += scale * up
        if activation_fn:
            net = activation_fn(net)
        return net


def block8(net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
    """
    Inception-Resnet-C.
    Builds the 8x8 resnet block.
    """

    with tf.variable_scope(scope, 'Block8', [net], reuse=reuse):
        with tf.variable_scope('Branch_0'):
            tower_conv = slim.conv2d(net, 192, 1, scope='Conv2d_1x1')
        with tf.variable_scope('Branch_1'):
            tower_conv1_0 = slim.conv2d(net, 192, 1, scope='Conv2d_0a_1x1')
            tower_conv1_1 = slim.conv2d(
                tower_conv1_0, 192, [1, 3], scope='Conv2d_0b_1x3')
            tower_conv1_2 = slim.conv2d(
                tower_conv1_1, 192, [3, 1], scope='Conv2d_0c_3x1')
        mixed = tf.concat([tower_conv, tower_conv1_2], 3)
        up = slim.conv2d(mixed, net.get_shape()[
                         3], 1, normalizer_fn=None, activation_fn=None, scope='Conv2d_1x1')
        net += scale * up
        if activation_fn:
            net = activation_fn(net)
    return net


def reduction_a(net, k, l, m, n):
    with tf.variable_scope('Branch_0'):
        tower_conv = slim.conv2d(
            net, n, 3, stride=2, padding='VALID', scope='Conv2d_1a_3x3')
    with tf.variable_scope('Branch_1'):
        tower_conv1_0 = slim.conv2d(net, k, 1, scope='Conv2d_0a_1x1')
        tower_conv1_1 = slim.conv2d(tower_conv1_0, l, 3, scope='Conv2d_0b_3x3')
        tower_conv1_2 = slim.conv2d(
            tower_conv1_1, m, 3, stride=2, padding='VALID', scope='Conv2d_1a_3x3')
    with tf.variable_scope('Branch_2'):
        tower_pool = slim.max_pool2d(
            net, 3, stride=2, padding='VALID', scope='MaxPool_1a_3x3')
    net = tf.concat([tower_conv, tower_conv1_2, tower_pool], 3)
    return net


def reduction_b(net):
    with tf.variable_scope('Branch_0'):
        tower_conv = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
        tower_conv_1 = slim.conv2d(
            tower_conv, 384, 3, stride=2, padding='VALID', scope='Conv2d_1a_3x3')
    with tf.variable_scope('Branch_1'):
        tower_conv1 = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
        tower_conv1_1 = slim.conv2d(
            tower_conv1, 256, 3, stride=2, padding='VALID', scope='Conv2d_1a_3x3')
    with tf.variable_scope('Branch_2'):
        tower_conv2 = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
        tower_conv2_1 = slim.conv2d(tower_conv2, 256, 3, scope='Conv2d_0b_3x3')
        tower_conv2_2 = slim.conv2d(
            tower_conv2_1, 256, 3, stride=2, padding='VALID', scope='Conv2d_1a_3x3')
    with tf.variable_scope('Branch_3'):
        tower_pool = slim.max_pool2d(
            net, 3, stride=2, padding='VALID', scope='MaxPool_1a_3x3')
    net = tf.concat([tower_conv_1, tower_conv1_1,
                     tower_conv2_2, tower_pool], 3)

    return net
