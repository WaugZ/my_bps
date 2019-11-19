import tensorflow as tf

slim = tf.contrib.slim


def placeholder_input(batch_size, num_point, channel=3):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, channel), name='input')
    return pointclouds_pl


def placeholder_label(batch_size):
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size), name='label')
    return labels_pl


def get_network(encoded, is_training, scale=1., weight_decay=0.00004):
    """ Classification PointNet, input is BxNx3, output Bx40 """
    with tf.variable_scope("Dense"):
        batch_size = encoded.get_shape()[0].value
        end_points = {}
        bn_params = {"is_training": is_training,
                     'epsilon': 1e-3
                     }
        pc = tf.reshape(encoded, [batch_size, 1, 1, -1])
        net = slim.conv2d(pc,
                          # 400,
                          max(int(round(400 * scale)), 32),
                          [1, 1],
                          padding='SAME',
                          stride=1,
                          normalizer_fn=slim.batch_norm,
                          normalizer_params=bn_params,
                          biases_initializer=tf.zeros_initializer(),
                          weights_regularizer=slim.l2_regularizer(weight_decay),
                          scope='fc1',
                          activation_fn=tf.nn.relu6)
        net = slim.dropout(net, keep_prob=0.8, is_training=is_training, scope='dp1')
        net = slim.conv2d(net,
                          # 400,
                          max(int(round(400 * scale)), 32),
                          [1, 1],
                          padding='SAME',
                          stride=1,
                          normalizer_fn=slim.batch_norm,
                          normalizer_params=bn_params,
                          biases_initializer=tf.zeros_initializer(),
                          weights_regularizer=slim.l2_regularizer(weight_decay),
                          scope='fc2',
                          activation_fn=tf.nn.relu6)
        net = slim.dropout(net, keep_prob=0.4, is_training=is_training, scope='dp2')
        net = slim.conv2d(net,
                          40, [1, 1],
                          padding='SAME',
                          stride=1,
                          # normalizer_fn=slim.batch_norm,
                          # normalizer_params=bn_params,
                          biases_initializer=tf.zeros_initializer(),
                          weights_regularizer=slim.l2_regularizer(weight_decay),
                          scope='fc3',
                          # activation_fn=tf.nn.relu6,
                          activation_fn=None,
                          )
        # print(net)
        net = tf.reshape(net, [batch_size, -1])
        return net, end_points

def get_loss(pred, label, end_points):
    """ pred: B*NUM_CLASSES,
      label: B, """
    labels = tf.one_hot(indices=label, depth=40)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=pred, label_smoothing=0.2)
    classify_loss = tf.reduce_mean(loss)
    return classify_loss