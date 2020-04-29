from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators import translate
from tensor2tensor.utils import registry

# End-of-sentence marker.
EOS = text_encoder.EOS_ID
_ZHVI_TRAIN_DATASETS = [[
    "https://github.com/nguyenpm2/vnu/blob/master/train-zh-vi.tar",  # pylint: disable=line-too-long
    ("train.zh", "train.vi")
]]

_ZHVI_TEST_DATASETS = [[
    "https://github.com/nguyenpm2/vnu/blob/master/test-zh-vi.tar",  # pylint: disable=line-too-long
    ("test.zh", "test.vi")
]]

@registry.register_problem
class TranslateZhvi(translate.TranslateProblem):
  """Problem spec for IWSLT'15 En-Vi translation."""

  @property
  def approx_vocab_size(self):
    return 2**14  # 16384

  @property
  def source_vocab_name(self):
      return "%s.zh" % self.vocab_filename

  @property
  def target_vocab_name(self):
      return "%s.vi" % self.vocab_filename

  def source_data_files(self, dataset_split):
    train = dataset_split == problem.DatasetSplit.TRAIN
    return _ZHVI_TRAIN_DATASETS if train else _ZHVI_TEST_DATASETS