import tensorflow as tf
FLAGS = tf.app.flags.FLAGS
## The following flags are related to save paths, tensorboard outputs and screen outputs
tf.app.flags.DEFINE_string('version', 'cri_group_glpool_all', '''A version number defining the directory to save logs and checkpoints''')
tf.app.flags.DEFINE_integer('report_freq', 200, '''Steps takes to output errors on the screen and write summaries''')
tf.app.flags.DEFINE_float('train_ema_decay', 0.9, '''The decay factor of the train error's moving average shown on tensorboard''')
tf.app.flags.DEFINE_float('keep_prob', 0.9, '''The factor of the dropout''')
## The following flags define hyper-parameters regards training
tf.app.flags.DEFINE_integer('train_steps', 60000, '''Total steps that you want to train''')
tf.app.flags.DEFINE_boolean('is_full_validation', True, '''Validation w/ full validation set or a random batch''')
tf.app.flags.DEFINE_integer('train_batch_size', 250, '''Train batch size''')
tf.app.flags.DEFINE_integer('validation_batch_size', 250, '''Validation batch size, better to be a divisor of 10000 for this task''')
tf.app.flags.DEFINE_integer('test_batch_size', 128, '''Test batch size''')
tf.app.flags.DEFINE_float('init_lr', 0.095, '''Initial learning rate''')
tf.app.flags.DEFINE_float('lr_decay_factor', 0.5, '''How much to decay the learning rate each time''')
tf.app.flags.DEFINE_integer('decay_step0', 10000, '''At which step to decay the learning rate''')
tf.app.flags.DEFINE_integer('decay_step1', 20000, '''At which step to decay the learning rate''')
tf.app.flags.DEFINE_integer('decay_step2', 30000, '''At which step to decay the learning rate''')
tf.app.flags.DEFINE_integer('decay_step3', 40000, '''At which step to decay the learning rate''')
tf.app.flags.DEFINE_integer('decay_step4', 50000, '''At which step to decay the learning rate''')
## The following flags define hyper-parameters modifying the training network
#tf.app.flags.DEFINE_integer('num_inception5_blocks', 3, '''How many inception5 blocks do you want''')
tf.app.flags.DEFINE_float('weight_decay', 0.0001, '''scale for l2 regularization''')
## The following flags are related to data-augmentation
tf.app.flags.DEFINE_integer('padding_size', 2, '''In data augmentation, layers of zero padding on each side of the image''')
## If you want to load a checkpoint and continue training
tf.app.flags.DEFINE_string('ckpt_path', 'D:/dataset/CIFAR/save-ckpt/logs_repeat20/model.ckpt-1000', '''Checkpoint directory to restore''')
tf.app.flags.DEFINE_boolean('is_use_ckpt', False, '''Whether to load a checkpoint and continue training''')
tf.app.flags.DEFINE_string('test_ckpt_path', 'model_110.ckpt-99999', '''Checkpoint
directory to restore''')

train_dir = 'logs_' + FLAGS.version + '/'