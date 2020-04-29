import tensorflow as tf
import os
from tensor2tensor.utils.trainer_lib import create_run_config, create_experiment
from tensor2tensor.utils.trainer_lib import create_hparams
from tensor2tensor.utils import registry
from tensor2tensor import models
from tensor2tensor import problems
from zh_vi import translate_zhvi

DATA_DIR = os.path.expanduser("C:\\Users\\NguyenPM\\PycharmProjects\\transformer\\testdata\\data")  # This folder contain the data
TMP_DIR = os.path.expanduser("C:\\Users\\NguyenPM\\PycharmProjects\\transformer\\testdata\\tmp")
TRAIN_DIR = os.path.expanduser("C:\\Users\\NguyenPM\\PycharmProjects\\transformer\\testdata\\train")  # This folder contain the model
EXPORT_DIR = os.path.expanduser("C:\\Users\\NguyenPM\\PycharmProjects\\transformer\\testdata\\export")  # This folder contain the exported model for production
TRANSLATIONS_DIR = os.path.expanduser("C:\\Users\\NguyenPM\\PycharmProjects\\transformer\\testdata\\translation")  # This folder contain  all translated sequence
EVENT_DIR = os.path.expanduser("C:\\Users\\NguyenPM\\PycharmProjects\\transformer\\testdata\\event")  # Test the BLEU score
USR_DIR = os.path.expanduser("C:\\Users\\NguyenPM\\PycharmProjects\\transformer\\testdata\\user")  # This folder contains our data that we want to add

tf.io.gfile.makedirs(DATA_DIR)
tf.io.gfile.makedirs(TMP_DIR)
tf.io.gfile.makedirs(TRAIN_DIR)
tf.io.gfile.makedirs(EXPORT_DIR)
tf.io.gfile.makedirs(TRANSLATIONS_DIR)
tf.io.gfile.makedirs(EVENT_DIR)
tf.io.gfile.makedirs(USR_DIR)

#Init parameters

PROBLEM = "translate_zhvi" # We chose a problem translation English to French zh
MODEL = "transformer" # Our model
HPARAMS = "transformer_big" # Hyperparameters for the model by default
                            # If you have a one gpu, use transformer_big_single_gpu

# Tham so
# train_steps = 1 # Total number of train steps for all Epochs
# eval_steps = 100 # Number of steps to perform for each evaluation
# batch_size = 5
# save_checkpoints_steps = 1
# ALPHA = 0.1
# schedule = "continuous_train_and_eval"
train_steps = 300000 # Total number of train steps for all Epochs
eval_steps = 100 # Number of steps to perform for each evaluation
batch_size = 4096
save_checkpoints_steps = 1000
ALPHA = 0.1
schedule = "continuous_train_and_eval"

# Huan luyen

# Init Hparams object from T2T Problem
hparams = create_hparams(HPARAMS)

# Make Changes to Hparams
hparams.batch_size = batch_size
hparams.learning_rate = ALPHA
#hparams.max_length = 256

# Can see all Hparams with code below
#print(json.loads(hparams.to_json())

RUN_CONFIG = create_run_config(
      model_dir=TRAIN_DIR,
      model_name=MODEL,
      save_checkpoints_steps= save_checkpoints_steps
)

tensorflow_exp_fn = create_experiment(
        run_config=RUN_CONFIG,
        hparams=hparams,
        model_name=MODEL,
        problem_name=PROBLEM,
        data_dir=DATA_DIR,
        train_steps=train_steps,
        eval_steps=eval_steps,
        #use_xla=True # For acceleration
    )

tensorflow_exp_fn.train_and_evaluate()
print("OK")