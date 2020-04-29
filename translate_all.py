from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import os
import shutil
from tensor2tensor.utils import bleu_hook

#TMP_DIR = os.path.expanduser("C:\\Users\\NguyenPM\\PycharmProjects\\transformer\\testdata\\tmp")
TRAIN_DIR = os.path.expanduser("C:\\Users\\NguyenPM\\PycharmProjects\\transformer\\testdata\\train")
TRANSLATIONS_DIR = os.path.expanduser("C:\\Users\\NguyenPM\\PycharmProjects\\transformer\\testdata\\translation")
DATA_DIR = os.path.expanduser("C:\\Users\\NguyenPM\\PycharmProjects\\transformer\\testdata\\data")
USR_DIR = os.path.expanduser("C:\\Users\\NguyenPM\\PycharmProjects\\transformer\\testdata\\user")
SOURCE_TEST_TRANSLATE_DIR = os.path.expanduser("C:\\Users\\NguyenPM\\PycharmProjects\\transformer\\testdata\\source")
REFERENCE_TEST_TRANSLATE_DIR = os.path.expanduser("C:\\Users\\NguyenPM\\PycharmProjects\\transformer\\testdata\\source")
BEAM_SIZE=1
PROBLEM = "translate_zhvi" # We chose a problem translation English to French zh
MODEL = "transformer" # Our model
HPARAMS = "transformer_big"

flags = tf.flags
FLAGS = flags.FLAGS
# t2t-translate-all specific options
flags.DEFINE_string("decoder_command", "t2t-decoder {params}",
                    "Which command to execute instead t2t-decoder. "
                    "{params} is replaced by the parameters. Useful e.g. for "
                    "qsub wrapper.")
flags.DEFINE_string("model_dir", TRAIN_DIR,
                    "Directory to load model checkpoints from.")
flags.DEFINE_string("source", SOURCE_TEST_TRANSLATE_DIR,
                    "Path to the source-language file to be translated")
flags.DEFINE_string("translations_dir", TRANSLATIONS_DIR,
                    "Where to store the translated files.")
flags.DEFINE_integer("min_steps", 0, "Ignore checkpoints with less steps.")
flags.DEFINE_integer("wait_minutes", 0,
                     "Wait upto N minutes for a new checkpoint")

# options derived from t2t-decoder
flags.DEFINE_integer("beam_size", BEAM_SIZE, "Beam-search width.")
flags.DEFINE_float("alpha", 0.6, "Beam-search alpha.")
flags.DEFINE_string("model", MODEL, "see t2t-decoder")
flags.DEFINE_string("t2t_usr_dir", USR_DIR, "see t2t-decoder")
flags.DEFINE_string("data_dir", DATA_DIR, "see t2t-decoder")
flags.DEFINE_string("problem", PROBLEM, "see t2t-decoder")
flags.DEFINE_string("hparams_set", HPARAMS,
                    "see t2t-decoder")
def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)
  # pylint: disable=unused-variable
  model_dir = os.path.expanduser(FLAGS.model_dir)
  translations_dir = os.path.expanduser(FLAGS.translations_dir)
  source = os.path.expanduser(FLAGS.source)
  tf.gfile.MakeDirs(translations_dir)
  translated_base_file = os.path.join(translations_dir, FLAGS.problem)

  # Copy flags.txt with the original time, so t2t-bleu can report correct
  # relative time.
  flags_path = os.path.join(translations_dir, FLAGS.problem + "-flags.txt")
  if not os.path.exists(flags_path):
    shutil.copy2(os.path.join(model_dir, "flags.txt"), flags_path)

  locals_and_flags = {"FLAGS": FLAGS}
  for model in bleu_hook.stepfiles_iterator(model_dir, FLAGS.wait_minutes,
                                            FLAGS.min_steps):
    tf.logging.info("Translating " + model.filename)
    out_file = translated_base_file + "-" + str(model.steps)
    locals_and_flags.update(locals())
    if os.path.exists(out_file):
      tf.logging.info(out_file + " already exists, so skipping it.")
    else:
      tf.logging.info("Translating " + out_file)
      params = (
          "--t2t_usr_dir={FLAGS.t2t_usr_dir} --output_dir={model_dir} "
          "--data_dir={FLAGS.data_dir} --problem={FLAGS.problem} "
          "--decode_hparams=beam_size={FLAGS.beam_size},alpha={FLAGS.alpha} "
          "--model={FLAGS.model} --hparams_set={FLAGS.hparams_set} "
          "--checkpoint_path={model.filename} --decode_from_file={source} "
          "--decode_to_file={out_file} --keep_timestamp"
      ).format(**locals_and_flags)
      command = FLAGS.decoder_command.format(**locals())
      tf.logging.info("Running:\n" + command)
      os.system(command)
  # pylint: enable=unused-variable


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()