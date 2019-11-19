import argparse
import numpy as np
import tensorflow as tf
import socket
import importlib
import os
import sys
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
sys.path.append(os.path.join(BASE_DIR, 'tools'))
import provider
import tf_util
import bps_util
import bps_h5_generator

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='fully_connected', help='Model name: [fully_connected/conv3d]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=512, help='BPS Point Number [512/1000] [default: 512]')
parser.add_argument('--sample_point', type=int, default=2048,
                    help='Point Cloud Sample Number [256/512/1024/2048] [default: 2048]')
parser.add_argument('--max_epoch', type=int, default=2500, help='Epoch to run [default: 2500]')
parser.add_argument('--batch_size', type=int, default=1024, help='Batch Size during training [default: 1024]')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='Initial learning rate [default: 0.0001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Pram:momentum for MOMENTUM optimizer[default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_epoch', type=int, default=2000, help='Decay epoch for lr decay [default: 2000]')
parser.add_argument('--decay_rate', type=float, default=0.1, help='Decay rate for lr decay [default: 0.1]')
parser.add_argument('--quantize_delay', type=int, default=None, help='Quantization decay, >0 for open [default:0]')
parser.add_argument('--checkpoint', default=None, help='Restore checkpoint')
parser.add_argument('--bps_type', default='rect_grid',
                    help='BPS type [rect_grid/ball_grid/random_uniform_ball/hcp] [default: rect_grid]')
parser.add_argument('--encode_method', default='sub',
                    help='BPS encode method [sub/dis] [default: sub]')
parser.add_argument('--scale', type=float, default=1., help="dgcnn depth scale")
FLAGS = parser.parse_args()

print(FLAGS)
BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_EPOCH = FLAGS.decay_epoch
DECAY_RATE = FLAGS.decay_rate
CHECKPOINT = FLAGS.checkpoint
SAMPLE_POINT = FLAGS.sample_point
BPS_TYPE = FLAGS.bps_type
ENCODE = FLAGS.encode_method
SCALE = FLAGS.scale

MODEL = importlib.import_module(FLAGS.model)  # import network module
MODEL_FILE = os.path.join(BASE_DIR, 'models', FLAGS.model + '.py')
LOG_DIR = FLAGS.log_dir
if CHECKPOINT:
    assert CHECKPOINT.endswith('ckpt')
    tmp = CHECKPOINT + '.index'
    LOG_DIR = os.path.dirname(tmp)
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp %s %s' % (MODEL_FILE, LOG_DIR))  # bkp of model def
os.system('cp train.py %s' % (LOG_DIR))  # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS) + '\n')

NUM_CLASSES = 40

HOSTNAME = socket.gethostname()

# ModelNet40 official train/test split
# todo: sample points and bps type should be optional parameters
if not os.path.exists('data/modelnet40_ply_hdf5_{}_{}_{}_{}'.format(SAMPLE_POINT, BPS_TYPE, NUM_POINT, ENCODE)):
    print("generating bps h5 files...")
    bps_h5_generator.convert_to_bps_h5(NUM_POINT, SAMPLE_POINT, BPS_TYPE, ENCODE)

TRAIN_FILES = provider.getDataFiles(
    os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_{}_{}_{}_{}/train_files.txt'.
                 format(SAMPLE_POINT, BPS_TYPE, NUM_POINT, ENCODE)))
TEST_FILES = provider.getDataFiles(
    os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_{}_{}_{}_{}/test_files.txt'.
                 format(SAMPLE_POINT, BPS_TYPE, NUM_POINT, ENCODE)))

TOTAL_TRAIN_FILES_NUM = 0
for fn in range(len(TRAIN_FILES)):
    current_data, _ = provider.loadDataFile(TRAIN_FILES[fn])
    TOTAL_TRAIN_FILES_NUM += current_data.shape[0]
print(TOTAL_TRAIN_FILES_NUM)

def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)


def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
        BASE_LEARNING_RATE,  # Base learning rate.
        batch * BATCH_SIZE,  # Current index into the dataset.
        DECAY_EPOCH * TOTAL_TRAIN_FILES_NUM,  # Decay step.
        DECAY_RATE,  # Decay rate.
        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001)  # CLIP THE LEARNING RATE!
    return learning_rate


def train():
    with tf.Graph().as_default():
        with tf.device('/gpu:' + str(GPU_INDEX)):
            if ENCODE == 'sub':
                pointclouds_pl = MODEL.placeholder_input(BATCH_SIZE, NUM_POINT, 3)
            elif ENCODE == 'dis':
                pointclouds_pl = MODEL.placeholder_input(BATCH_SIZE, NUM_POINT, 1)
            labels_pl = MODEL.placeholder_label(BATCH_SIZE)
            if not FLAGS.quantize_delay:
                is_training = tf.placeholder(tf.bool, shape=(), name="is_training")
            else:
                is_training = True
            # print(is_training_pl)

            # Note the global_step=batch parameter to minimize.
            # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
            batch = tf.Variable(0)

            # Get model
            pred, end_points = MODEL.get_network(pointclouds_pl, is_training,
                                                 scale=SCALE,
                                                 )
            if FLAGS.quantize_delay and FLAGS.quantize_delay > 0:
                tf.contrib.quantize.create_training_graph(
                    quant_delay=FLAGS.quantize_delay)

            # Get loss
            loss = MODEL.get_loss(pred, labels_pl, end_points)
            regularization_losses = tf.get_collection(
                tf.GraphKeys.REGULARIZATION_LOSSES)
            all_losses = []
            all_losses.append(loss)
            all_losses.append(tf.add_n(regularization_losses))
            total_loss = tf.add_n(all_losses)

            # tf.summary.scalar('loss', loss)
            tf.summary.scalar('loss', total_loss)

            correct = tf.equal(tf.argmax(pred, 1), tf.to_int64(labels_pl))
            accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE)
            tf.summary.scalar('accuracy', accuracy)

            # if update_ops:
            #     print("BN parameters: ", update_ops)
            #     updates = tf.group(*update_ops)
            #     train_step = control_flow_ops.with_dependencies([updates], batch)

            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies([tf.group(*update_ops)]):
                # train_op = optimizer.minimize(loss, global_step=batch)
                train_op = optimizer.minimize(total_loss, global_step=batch)

            # Add ops to save and restore all the variables.
            saver = tf.train.Saver(max_to_keep=50)

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        # Add summary writers
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'),
                                             sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'))

        # Init variables
        init = tf.global_variables_initializer()
        # To fix the bug introduced in TF 0.12.1 as in
        # http://stackoverflow.com/questions/41543774/invalidargumenterror-for-tensor-bool-tensorflow-0-12-1
        sess.run(init)
        # sess.run(init, {is_training_pl: True})
        if FLAGS.quantize_delay and FLAGS.quantize_delay > 0:
            ops = {'pointclouds_pl': pointclouds_pl,
                   'labels_pl': labels_pl,
                   # 'is_training_pl': is_training,
                   'pred': pred,
                   'loss': loss,
                   'train_op': train_op,
                   'merged': merged,
                   'step': batch}
        else:
            ops = {'pointclouds_pl': pointclouds_pl,
                   'labels_pl': labels_pl,
                   'is_training_pl': is_training,
                   'pred': pred,
                   'loss': loss,
                   'train_op': train_op,
                   'merged': merged,
                   'step': batch}

        ever_best = 0
        if CHECKPOINT:
            saver.restore(sess, CHECKPOINT)
        for epoch in range(MAX_EPOCH):
            log_string(('**** EPOCH %03d ****' % (epoch))
                       + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + '****')
            sys.stdout.flush()

            ma = train_one_epoch(sess, ops, train_writer)
            if not FLAGS.quantize_delay:
                ma = eval_one_epoch(sess, ops, test_writer)

                # Save the variables to disk.

                if ma > ever_best:
                    save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
                    log_string("Model saved in file: %s" % save_path)
                    ever_best = ma
                log_string("Current model mean accuracy: {}".format(ma))
                log_string("Best model mean accuracy: {}".format(ever_best))
            else:
                if epoch % 5 == 0:
                    if CHECKPOINT:
                        save_path = saver.save(sess, os.path.join(LOG_DIR, "model-r-{}.ckpt".format(str(epoch))))
                    else:
                        save_path = saver.save(sess, os.path.join(LOG_DIR, "model-{}.ckpt".format(str(epoch))))
                    log_string("Model saved in file: %s" % save_path)


def train_one_epoch(sess, ops, train_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = True

    # Shuffle train files
    train_file_idxs = np.arange(0, len(TRAIN_FILES))
    np.random.shuffle(train_file_idxs)

    mean_acc = 0
    for fn in range(len(TRAIN_FILES)):
        log_string('----' + str(fn) + '-----')
        current_data, current_label = provider.loadDataFile(TRAIN_FILES[train_file_idxs[fn]])
        current_data, current_label, _ = provider.shuffle_data(current_data, np.squeeze(current_label))
        current_label = np.squeeze(current_label)

        file_size = current_data.shape[0]
        num_batches = file_size // BATCH_SIZE

        total_correct = 0
        total_seen = 0
        loss_sum = 0

        for batch_idx in range(num_batches):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = (batch_idx + 1) * BATCH_SIZE


            if not FLAGS.quantize_delay:
                feed_dict = {
                    # ops['pointclouds_pl']: jittered_data,
                    ops['pointclouds_pl']: current_data[start_idx:end_idx, :, :],
                    ops['labels_pl']: current_label[start_idx:end_idx],
                    ops['is_training_pl']: is_training,
                }
            else:
                feed_dict = {
                    # ops['pointclouds_pl']: jittered_data,
                    ops['pointclouds_pl']: current_data[start_idx:end_idx, :, :],
                    ops['labels_pl']: current_label[start_idx:end_idx],
                    # ops['is_training_pl']: is_training,
                }
            summary, step, _, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
                                                             ops['train_op'], ops['loss'], ops['pred']],
                                                            feed_dict=feed_dict)
            train_writer.add_summary(summary, step)
            pred_val = np.argmax(pred_val, 1)
            correct = np.sum(pred_val == current_label[start_idx:end_idx])
            total_correct += correct
            total_seen += BATCH_SIZE
            loss_sum += loss_val

        log_string('mean loss: %f' % (loss_sum / float(num_batches)))
        log_string('accuracy: %f' % (total_correct / float(total_seen)))

        mean_acc += total_correct / float(total_seen)
    return mean_acc / len(TRAIN_FILES)


def eval_one_epoch(sess, ops, test_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = False
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]

    for fn in range(len(TEST_FILES)):
        log_string('----' + str(fn) + '-----')
        current_data, current_label = provider.loadDataFile(TEST_FILES[fn])
        current_label = np.squeeze(current_label)

        file_size = current_data.shape[0]
        num_batches = file_size // BATCH_SIZE


        for batch_idx in range(num_batches):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = (batch_idx + 1) * BATCH_SIZE

            feed_dict = {ops['pointclouds_pl']: current_data[start_idx:end_idx, :, :],
                         ops['labels_pl']: current_label[start_idx:end_idx],
                         ops['is_training_pl']: is_training
                         }

            summary, step, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
                                                          ops['loss'], ops['pred']], feed_dict=feed_dict)
            pred_val = np.argmax(pred_val, 1)
            correct = np.sum(pred_val == current_label[start_idx:end_idx])
            total_correct += correct
            total_seen += BATCH_SIZE
            loss_sum += (loss_val * BATCH_SIZE)
            for i in range(start_idx, end_idx):
                l = current_label[i]
                total_seen_class[l] += 1
                total_correct_class[l] += (pred_val[i - start_idx] == l)

    log_string('eval mean loss: %f' % (loss_sum / float(total_seen)))
    acc = total_correct / float(total_seen)
    log_string('eval accuracy: %f' % acc)
    aca = np.mean(np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float))
    log_string('eval avg class acc: %f' % aca)

    return acc


if __name__ == "__main__":
    train()
    LOG_FOUT.close()
