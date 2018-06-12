#!../../venv/tf15py34/bin/python3
# -*- coding: utf-8 -*-
"""This is the top-level file to train, evaluate or test your summarization model"""
import glob
import json
import time
import zipfile
from collections import namedtuple
from datetime import datetime

import numpy as np
import os, sys

if 'LD_LIBRARY_PATH' not in os.environ or 'cuda-8.0' not in os.environ['LD_LIBRARY_PATH']:
    os.environ['LD_LIBRARY_PATH'] = '/opt/cuda-8.0/lib64:/opt/cuda-9.0/extras/CUPTI/lib64:/opt/cudnn-6.0/lib/'
    try:
        os.execv(sys.argv[0], sys.argv)
    except Exception as exc:
        print('\033[91m', 'Failed re-exec:', exc, '\033[0m')
        sys.exit(1)

print('\033[92m', 'Success Loading CUDA 9:', os.environ['LD_LIBRARY_PATH'], '\033[0m')
from util import bcolors
import tensorflow as tf
from tensorflow.python import debug as tf_debug

import util

from batcher import Batcher
from data import Vocab
from decode import BeamSearchDecoder
from gpu_cluster import get_available_gpu
from model import SummarizationModel

FLAGS = tf.app.flags.FLAGS

# Where to find data
nowpath = os.getcwd()
#nowpath+"\finished_files\chunked\train_*"
#nowpath+"\finished_files\vocab"
tf.app.flags.DEFINE_string('data_path', './finished_files/chunked/train_*',
                           'Path expression to tf.Example datafiles. Can include wildcards to access multiple datafiles.')
tf.app.flags.DEFINE_string('vocab_path', './finished_files/vocab', 'Path expression to text vocabulary file.')

# Important settings
tf.app.flags.DEFINE_string('mode', 'train', 'must be one of train/eval/decode')
tf.app.flags.DEFINE_boolean('single_pass', False,
                            'For decode mode only. If True, run eval on the full dataset using a fixed checkpoint, i.e.'
                            ' take the current checkpoint, and use it to produce one summary for each example in the dataset, '
                            'write the summaries to file and then get ROUGE scores for the whole dataset. '
                            'If False (default), run concurrent decoding, i.e. repeatedly load latest checkpoint,'
                            ' use it to produce summaries for randomly-chosen examples and log the results to screen, indefinitely.')

# Where to save output
#nowpath+'\log'
tf.app.flags.DEFINE_string('log_root', '', 'Root directory for all logging.')
tf.app.flags.DEFINE_string('exp_name', 'myexperiment',
                           'Name for experiment. Logs will be saved in a directory with this name, under log_root.')

# Hyperparameters
tf.app.flags.DEFINE_integer('hidden_dim', 512, 'dimension of RNN hidden states')
tf.app.flags.DEFINE_integer('emb_dim', 200, 'dimension of word embeddings')
tf.app.flags.DEFINE_integer('batch_size', 64, 'minibatch size')
tf.app.flags.DEFINE_integer('max_enc_steps', 150, 'max timesteps of encoder (max source text tokens)')
tf.app.flags.DEFINE_integer('max_dec_steps', 30, 'max timesteps of decoder (max summary tokens)')
tf.app.flags.DEFINE_integer('beam_size', 4, 'beam size for beam search decoding.')
tf.app.flags.DEFINE_integer('min_dec_steps', 20,
                            'Minimum sequence length of generated summary. Applies only for beam search decoding mode')
tf.app.flags.DEFINE_integer('vocab_size', 50000,
                            'Size of vocabulary. These will be read from the vocabulary file in order.'
                            ' If the vocabulary file contains fewer words than this number, or if this number is set to 0, '
                            'will take all words in the vocabulary file.')
tf.app.flags.DEFINE_float('lr', 0.15, 'learning rate')
tf.app.flags.DEFINE_float('adagrad_init_acc', 0.1, 'initial accumulator value for Adagrad')
tf.app.flags.DEFINE_float('rand_unif_init_mag', 0.02, 'magnitude for lstm cells random uniform inititalization')
tf.app.flags.DEFINE_float('trunc_norm_init_std', 1e-4, 'std of trunc norm init, used for initializing everything else')
tf.app.flags.DEFINE_float('max_grad_norm', 2.0, 'for gradient clipping')

# Pointer-generator or baseline model
tf.app.flags.DEFINE_boolean('pointer_gen', True, 'If True, use pointer-generator model. If False, use baseline model.')

# Coverage hyperparameters
tf.app.flags.DEFINE_boolean('coverage', False,
                            'Use coverage mechanism. Note, the experiments reported in the ACL paper train WITHOUT coverage until converged,'
                            ' and then train for a short phase WITH coverage afterwards. i.e. to reproduce the results in the ACL paper, turn this off for most of training then turn on for a short phase at the end.')
tf.app.flags.DEFINE_float('cov_loss_wt', 1.0,
                          'Weight of coverage loss (lambda in the paper). If zero, then no incentive to minimize coverage loss.')

# Utility flags, for restoring and changing checkpoints
tf.app.flags.DEFINE_boolean('convert_to_coverage_model', False,
                            'Convert a non-coverage model to a coverage model. Turn this on and run in train mode. Your current training model will be copied to a new version (same name with _cov_init appended) that will be ready to run with coverage flag turned on, for the coverage training stage.')
tf.app.flags.DEFINE_boolean('restore_best_model', False,
                            'Restore the best model in the eval/ dir and save it in the train/ dir, ready to be used for further training. Useful for early stopping, or if your training checkpoint has become corrupted with e.g. NaN values.')

# Debugging. See https://www.tensorflow.org/programmers_guide/debugger
tf.app.flags.DEFINE_boolean('debug', False, "Run in tensorflow's debug mode (watches for NaN/inf values)")
tf.app.flags.DEFINE_string('tb_debug_url', 'localhost:6064', "using tensorboard debugger")
tf.app.flags.DEFINE_integer('device', None, "use which gpu (from 0 to 3) ")
tf.app.flags.DEFINE_boolean('decode_rouge', False, "decode generated sentence in Rouge format")
tf.app.flags.DEFINE_boolean('decode_bleu', False, "decode generated sentence in BLEU format")


tf.logging.set_verbosity('DEBUG')#获得日志输出


def calc_running_avg_loss(loss, running_avg_loss, summary_writer, step, decay=0.99):
    """Calculate the running average loss via exponential decay.
    This is used to implement early stopping w.r.t. a more smooth loss curve than the raw loss curve.

    Args:
      loss: loss on the most recent eval step
      running_avg_loss: running_avg_loss so far
      summary_writer: FileWriter object to write for tensorboard
      step: training iteration step
      decay: rate of exponential decay, a float between 0 and 1. Larger is smoother.

    Returns:
      running_avg_loss: new running average loss
    """
    if running_avg_loss == 0:  # on the first iteration just take the loss
        running_avg_loss = loss
    else:
        running_avg_loss = running_avg_loss * decay + (1 - decay) * loss
    running_avg_loss = min(running_avg_loss, 12)  # clip
    loss_sum = tf.Summary()
    tag_name = 'running_avg_loss/decay=%f' % (decay)
    loss_sum.value.add(tag=tag_name, simple_value=running_avg_loss)
    summary_writer.add_summary(loss_sum, step)
    tf.logging.info('running_avg_loss: %f', running_avg_loss)
    return running_avg_loss


def restore_best_model():
    """Load bestmodel file from eval directory, add variables for adagrad, and save to train directory"""
    tf.logging.info("Restoring bestmodel for training...")

    # Initialize all vars in the model
    sess = tf.Session(config=util.get_config())
    print("Initializing all variables...")
    sess.run(tf.initialize_all_variables())

    # Restore the best model from eval dir
    saver = tf.train.Saver([v for v in tf.all_variables() if "Adagrad" not in v.name])
    print("Restoring all non-adagrad variables from best model in eval dir...")
    curr_ckpt = util.load_ckpt(saver, sess, "eval")
    print("Restored %s." % curr_ckpt)

    # Save this model to train dir and quit
    new_model_name = curr_ckpt.split("/")[-1].replace("bestmodel", "model")
    new_fname = os.path.join(FLAGS.log_root, "train", new_model_name)
    print("Saving model to %s..." % (new_fname))
    new_saver = tf.train.Saver()  # this saver saves all variables that now exist, including Adagrad variables
    new_saver.save(sess, new_fname)
    print("Saved.")
    exit()


def convert_to_coverage_model():
    """Load non-coverage checkpoint, add initialized extra variables for coverage, and save as new checkpoint"""
    tf.logging.info("converting non-coverage model to coverage model..")

    # initialize an entire coverage model from scratch
    sess = tf.Session(config=util.get_config())
    print("initializing everything...")
    sess.run(tf.global_variables_initializer())

    # load all non-coverage weights from checkpoint
    saver = tf.train.Saver([v for v in tf.global_variables() if "coverage" not in v.name and "Adagrad" not in v.name])
    print("restoring non-coverage variables...")
    curr_ckpt = util.load_ckpt(saver, sess)
    print("restored.")

    # save this model and quit
    new_fname = curr_ckpt + '_cov_init'
    print("saving model to %s..." % (new_fname))
    new_saver = tf.train.Saver()  # this one will save all variables that now exist
    new_saver.save(sess, new_fname)
    print("saved.")
    exit()


def setup_training(model, batcher):
    """Does setup before starting training (run_training)"""
    train_dir = os.path.join(FLAGS.log_root, "train")
    if not os.path.exists(train_dir): os.makedirs(train_dir)
    model.build_graph()  # build the graph

    if FLAGS.convert_to_coverage_model:
        assert FLAGS.coverage, "To convert your non-coverage model to a coverage model, run with convert_to_coverage_model=True and coverage=True"
        convert_to_coverage_model()
    if FLAGS.restore_best_model:
        restore_best_model()
    saver = tf.train.Saver(max_to_keep=30)  # keep 3 checkpoints at a time
    saver_hook = tf.train.CheckpointSaverHook(train_dir, save_steps=5000, saver=saver)#Saves checkpoints every N steps or seconds.
    summary_writer = tf.summary.FileWriter(train_dir)#将汇总的写入文件
    #Saves summaries every N steps.
    summary_hook = tf.train.SummarySaverHook(save_steps=20, summary_op=model._summaries, summary_writer=summary_writer)
    tf.logging.info("Created session.")

    session = tf.train.MonitoredTrainingSession(hooks=[saver_hook, summary_hook], config=util.get_config(),
                                          checkpoint_dir=train_dir)

    summary_writer.add_graph(session.graph)
    try:

        run_training(model, batcher, session, summary_writer)  # this is an infinite loop until interrupted
    except KeyboardInterrupt:
        tf.logging.info("Caught keyboard interrupt on worker. Stopping supervisor...")
        session.close()


def run_training(model, batcher, session, summary_writer):
    """Repeatedly runs training iterations, logging loss to screen and writing summaries"""
    tf.logging.info("starting run_training")
    if FLAGS.debug:  # start the tensorflow debugger
        # session = tf_debug.LocalCLIDebugWrapperSession(session)
        tf.logging.info(type(session))
        tf.logging.info(type(session._sess))
        real_session = session
        while True:
            if hasattr(real_session, '_sess') and type(real_session._sess) != tf.Session:
                real_session = real_session._sess
            elif hasattr(real_session, '_sess') and type(real_session._sess) == tf.Session:
                real_session._sess = tf_debug.TensorBoardDebugWrapperSession(real_session._sess, FLAGS.tb_debug_url)
                break
        tf.logging.info(type(session))
        tf.logging.info(type(session._sess))

        # session.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
    batch_time_list = []
    train_step = -1
    while not session.should_stop():  # repeats until interrupted
        batch = batcher.next_batch()

        t0 = time.time()
        if train_step % 5000 == 0:  # from 0 step every 1000 step record the running statistic
            tf.logging.info('Recording running statistic')
            results = model.run_train_step(session, batch, metadata=True, summary_writer=summary_writer)
        else:
            results = model.run_train_step(session, batch)
        t1 = time.time()

        loss = results['loss']
        train_step = results['global_step']  # we need this to update our running average loss
        batch_time_list.append(t1 - t0)
        if train_step % 20 == 0:
            tf.logging.info('step: %d | loss: %f | avg batch time: %.3f', train_step, loss,
                            sum(batch_time_list) / len(batch_time_list))
            batch_time_list.clear()

        if not np.isfinite(loss):
            raise Exception("Loss is not finite. Stopping.")

        if FLAGS.coverage:
            coverage_loss = results['coverage_loss']
            tf.logging.info("coverage_loss: %f", coverage_loss)  # print the coverage loss to screen


def run_eval(model, batcher, vocab):
    """Repeatedly runs eval iterations, logging to screen and writing summaries. Saves the model with the best loss seen so far."""
    model.build_graph()  # build the graph
    saver = tf.train.Saver(max_to_keep=3)  # we will keep 3 best checkpoints at a time
    sess = tf.Session(config=util.get_config())
    eval_dir = os.path.join(FLAGS.log_root, "eval")  # make a subdir of the root dir for eval data
    bestmodel_save_path = os.path.join(eval_dir, 'bestmodel')  # this is where checkpoints of best models are saved
    summary_writer = tf.summary.FileWriter(eval_dir)
    running_avg_loss = 0  # the eval job keeps a smoother, running average loss to tell it when to implement early stopping
    best_loss = None  # will hold the best loss achieved so far

    while True:
        _ = util.load_ckpt(saver, sess)  # load a new checkpoint
        batch = batcher.next_batch()  # get the next batch

        # run eval on the batch
        t0 = time.time()
        results = model.run_eval_step(sess, batch)
        t1 = time.time()
        tf.logging.info('seconds for batch: %.2f', t1 - t0)

        # print the loss and coverage loss to screen
        loss = results['loss']
        tf.logging.info('loss: %f', loss)
        if FLAGS.coverage:
            coverage_loss = results['coverage_loss']
            tf.logging.info("coverage_loss: %f", coverage_loss)

        # add summaries
        summaries = results['summaries']
        train_step = results['global_step']
        summary_writer.add_summary(summaries, train_step)

        # calculate running avg loss
        running_avg_loss = calc_running_avg_loss(np.asscalar(loss), running_avg_loss, summary_writer, train_step)

        # If running_avg_loss is best so far, save this checkpoint (early stopping).
        # These checkpoints will appear as bestmodel-<iteration_number> in the eval dir
        if best_loss is None or running_avg_loss < best_loss:
            tf.logging.info('Found new best model with %.3f running_avg_loss. Saving to %s', running_avg_loss,
                            bestmodel_save_path)
            saver.save(sess, bestmodel_save_path, global_step=train_step, latest_filename='checkpoint_best')
            best_loss = running_avg_loss

        # flush the summary writer every so often
        if train_step % 100 == 0:
            summary_writer.flush()


def main(unused_argv):
    # GPU tricks
    if FLAGS.device == None:
        index_of_gpu = get_available_gpu()
        if index_of_gpu < 0:
            index_of_gpu = ''
        FLAGS.device = index_of_gpu
        tf.logging.info(bcolors.OKGREEN + 'using {}'.format(FLAGS.device) + bcolors.ENDC)#终端颜色
    else:
        index_of_gpu = FLAGS.device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(index_of_gpu)
    tf.logging.info('try to occupy GPU memory!')
    placeholder_session = tf.Session()
    #tf.contrib.memory_stats.BytesLimit()：Generates an op that measures the total memory (in bytes) of a device.
    limit = placeholder_session.run(tf.contrib.memory_stats.BytesLimit()) / 1073741824
    tf.logging.info('occupy GPU memory %f GB', limit)

    if len(unused_argv) != 1:  # prints a message if you've entered flags incorrectly
        raise Exception("Problem with flags: %s" % unused_argv)

    tf.logging.info('Starting seq2seq_attention in %s mode...', (FLAGS.mode))

    # Change log_root to FLAGS.log_root/FLAGS.exp_name and create the dir if necessary
    FLAGS.log_root = os.path.join(FLAGS.log_root, FLAGS.exp_name)
    if not os.path.exists(FLAGS.log_root):
        if FLAGS.mode == "train":
            os.makedirs(FLAGS.log_root)
        else:
            raise Exception("Logdir %s doesn't exist. Run in train mode to create it." % (FLAGS.log_root))
    tf.logging.info("vocab path is %s ", FLAGS.vocab_path)
    vocab = Vocab(FLAGS.vocab_path, FLAGS.vocab_size)  # create a vocabulary

    # If in decode mode, set batch_size = beam_size
    # Reason: in decode mode, we decode one example at a time.
    # On each step, we have beam_size-many hypotheses in the beam, so we need to make a batch of these hypotheses.
    if FLAGS.mode == 'decode':
        FLAGS.batch_size = FLAGS.beam_size

    # If single_pass=True, check we're in decode mode
    if FLAGS.single_pass and FLAGS.mode != 'decode':
        raise Exception("The single_pass flag should only be True in decode mode")

    # Make a namedtuple hps, containing the values of the hyperparameters that the model needs
    hparam_list = ['mode', 'lr', 'adagrad_init_acc', 'rand_unif_init_mag', 'trunc_norm_init_std', 'max_grad_norm',
                   'hidden_dim', 'emb_dim', 'batch_size', 'max_dec_steps', 'max_enc_steps', 'coverage', 'cov_loss_wt',
                   'pointer_gen']
    hps_dict = {}
    export_json = {}
    for key, val in FLAGS.__flags.items():

        export_json[key] = val
        if key in hparam_list:
            hps_dict[key] = val
            tf.logging.info('{} {}'.format(key, val))
    #
    hps = namedtuple("HParams", hps_dict.keys())(**hps_dict)
    ######################
    # save parameters and python script
    ######################
    # save parameters
    tf.logging.info('saving parameters')
    current_time_str = datetime.now().strftime('%m-%d-%H-%M')
    json_para_file = open(os.path.join(FLAGS.log_root, 'flags-' + current_time_str + '-' + FLAGS.mode + '.json'), 'w')
    json_para_file.write(json.dumps(export_json, indent=4) + '\n')
    json_para_file.close()
    # save python source code
    tf.logging.info('saving source code')
    python_list = glob.glob('./*.py')
    zip_file = zipfile.ZipFile(
        os.path.join(FLAGS.log_root, 'source_code_bak-' + current_time_str + '-' + FLAGS.mode + '.zip'), 'w')
    for d in python_list:
        zip_file.write(d)
    zip_file.close()

    # Create a batcher object that will create minibatches of data
    batcher = Batcher(FLAGS.data_path, vocab, hps, single_pass=FLAGS.single_pass)

    tf.set_random_seed(111)  # a seed value for randomness

    if hps.mode == 'train':
        tf.logging.info("creating model...")
        model = SummarizationModel(hps, vocab)
        placeholder_session.close()
        setup_training(model, batcher)
    elif hps.mode == 'eval':
        model = SummarizationModel(hps, vocab)
        placeholder_session.close()
        run_eval(model, batcher, vocab)
    elif hps.mode == 'decode':
        decode_model_hps = hps  # This will be the hyperparameters for the decoder model
        decode_model_hps = hps._replace(
            max_dec_steps=1)  # The model is configured with max_dec_steps=1 because we only ever run one step of the decoder at a time (to do beam search). Note that the batcher is initialized with max_dec_steps equal to e.g. 100 because the batches need to contain the full summaries
        model = SummarizationModel(decode_model_hps, vocab)
        decoder = BeamSearchDecoder(model, batcher, vocab)
        placeholder_session.close()
        try:
            decoder.decode()  # decode indefinitely (unless single_pass=True, in which case deocde the dataset exactly once)
        except KeyboardInterrupt:
            tf.logging.info('stop decoding!')
    else:
        raise ValueError("The 'mode' flag must be one of train/eval/decode")


if __name__ == '__main__':
    tf.app.run()#处理flag解析，运行main函数
