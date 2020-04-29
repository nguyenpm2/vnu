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

PROBLEM = "translate_zhvi" # We chose a problem translation English to French zh
MODEL = "transformer" # Our model
HPARAMS = "transformer_big" # Hyperparameters for the model by default
                            # If you have a one gpu, use transformer_big_single_gpu

# Data generation
t2t_problem = problems.problem(PROBLEM)
t2t_problem.generate_data(DATA_DIR, TMP_DIR)
print("OK")