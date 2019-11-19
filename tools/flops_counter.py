import os
import sys
import tensorflow as tf
import importlib
import argparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '..', 'models'))
sys.path.append(os.path.join(BASE_DIR, '..'))

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='fully_connected', help='Model name: fully_connected')
parser.add_argument('--num_point', type=int, default=512, help='BPS Point Number [512/1000] [default: 512]')
parser.add_argument('--sample_point', type=int, default=2048,
                    help='Point Cloud Sample Number [256/512/1024/2048] [default: 2048]')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during training [default: 32]')
parser.add_argument('--quantize_delay', type=int, default=None, help='Quantization decay, >0 for open [default:0]')
parser.add_argument('--scale', type=float, default=1., help="dgcnn depth scale")
parser.add_argument('--bps_type', default='rect_grid',
                    help='BPS type [rect_grid/ball_grid/random_uniform_ball/hcp] [default: rect_grid]')
parser.add_argument('--encode_method', default='sub',
                    help='BPS encode method [sub/dis] [default: sub]')
FLAGS = parser.parse_args()


BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
SCALE = FLAGS.scale
SAMPLE_POINT = FLAGS.sample_point
BPS_TYPE = FLAGS.bps_type
ENCODE = FLAGS.encode_method

MODEL = importlib.import_module(FLAGS.model)  # import network module
MODEL_FILE = os.path.join(BASE_DIR, '..', 'models', FLAGS.model + '.py')


if __name__ == "__main__":
    g = tf.Graph()
    run_meta = tf.RunMetadata()
    with g.as_default():
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

        opts = tf.profiler.ProfileOptionBuilder.float_operation()
        flops = tf.profiler.profile(g, run_meta=run_meta, cmd='op', options=opts)

        opts = tf.profiler.ProfileOptionBuilder.trainable_variables_parameter()
        params = tf.profiler.profile(g, run_meta=run_meta, cmd='op', options=opts)
        if flops and params:
            print("FLOPs = {:,}".format(flops.total_float_ops))
            print("#params = {:,}".format(params.total_parameters))
