import tensorflow as tf

#After training the model, re-run the environment but run this code in first, then predict.

tfe = tf.contrib.eager
tfe.enable_eager_execution()
Modes = tf.estimator.ModeKeys

#Config

from tensor2tensor import models
from tensor2tensor import problems
from tensor2tensor.layers import common_layers
from tensor2tensor.utils import trainer_lib
from tensor2tensor.utils import t2t_model
from tensor2tensor.utils import registry
from tensor2tensor.utils import metrics
import os
import numpy as np
from zh_vi import translate_zhvi

DATA_DIR = os.path.expanduser("C:\\Users\\NguyenPM\\PycharmProjects\\transformer\\testdata\\data")  # This folder contain the data
TRAIN_DIR = os.path.expanduser("C:\\Users\\NguyenPM\\PycharmProjects\\transformer\\testdata\\train")  # This folder contain the model
tf.io.gfile.makedirs(DATA_DIR)
tf.io.gfile.makedirs(TRAIN_DIR)

PROBLEM = "translate_zhvi"
MODEL = "transformer" # Our model
HPARAMS = "transformer_big"

zhvi_problem = problems.problem(PROBLEM)

# Copy the vocab file locally so we can encode inputs and decode model outputs
vocab_name = "vocab.translate_zhvi.16384.subwords"
vocab_file = os.path.join(DATA_DIR, vocab_name)

# Get the encoders from the problem
encoders = zhvi_problem.feature_encoders(DATA_DIR)

ckpt_path = tf.train.latest_checkpoint(os.path.join(TRAIN_DIR))
print(ckpt_path)

def translate(inputs):
  encoded_inputs = encode(inputs)
  with tfe.restore_variables_on_create(ckpt_path):
    model_output = translate_model.infer(encoded_inputs)["outputs"]
  return decode(model_output)

def encode(input_str, output_str=None):
  """Input str to features dict, ready for inference"""
  inputs = encoders["inputs"].encode(input_str) + [1]  # add EOS id
  batch_inputs = tf.reshape(inputs, [1, -1, 1])  # Make it 3D.
  return {"inputs": batch_inputs}

def decode(integers):
  """List of ints to str"""
  integers = list(np.squeeze(integers))
  if 1 in integers:
    integers = integers[:integers.index(1)]
  return encoders["inputs"].decode(np.squeeze(integers))

#Predict

hparams = trainer_lib.create_hparams(HPARAMS, data_dir=DATA_DIR, problem_name=PROBLEM)
translate_model = registry.model(MODEL)(hparams, Modes.PREDICT)

inputs = "那是一条狗"
ref = "Đó là một con chó" ## this just a reference for evaluate the quality of the traduction
outputs = translate(inputs)

file_input = open("outputs.vi", "w+", encoding="utf8")
file_input.write(outputs)
file_input.close()

print("Inputs: %s" % inputs)
print("Outputs: %s" % outputs)