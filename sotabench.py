from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
import tensorflow as tf

import models.official.efficientnet.efficientnet_builder as efficientnet_builder
import models.official.efficientnet.preprocessing as preprocessing
import models.official.efficientnet.utils as utils

flags.DEFINE_string('model_name', 'efficientnet-b0', 'Model name to eval.')
flags.DEFINE_string('runmode', 'examples', 'Running mode: examples or imagenet')
flags.DEFINE_string(
    'imagenet_eval_glob', None, 'Imagenet eval image glob, '
    'such as ./data/vision/imagenet/ILSVRC2012_img_val/ILSVRC2012*.JPEG')
flags.DEFINE_string(
    'imagenet_eval_label', None, 'Imagenet eval label file path, '
    'such as ./data/vision/imagenet/ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt')
flags.DEFINE_string('ckpt_dir', '/tmp/ckpt/', 'Checkpoint folders')
flags.DEFINE_boolean('enable_ema', True, 'Enable exponential moving average.')
flags.DEFINE_string('export_ckpt', None, 'Exported ckpt for eval graph.')
flags.DEFINE_string('example_img', '/tmp/panda.jpg',
                    'Filepath for a single example image.')
flags.DEFINE_string('labels_map_file', '/tmp/labels_map.txt',
                    'Labels map from label id to its meaning.')
flags.DEFINE_bool('include_background_label', False,
                  'Whether to include background as label #0')
flags.DEFINE_integer('num_images', 5000,
                     'Number of images to eval. Use -1 to eval all images.')


class EvalCkptDriver(utils.EvalCkptDriver):
  """A driver for running eval inference."""

  def build_model(self, features, is_training):
    """Build model with input features."""
    if self.model_name.startswith('efficientnet'):
      model_builder = efficientnet_builder
    else:
      raise ValueError(
          'Model must be either efficientnet-b* or efficientnet-edgetpu*')

    features -= tf.constant(
        model_builder.MEAN_RGB, shape=[1, 1, 3], dtype=features.dtype)
    features /= tf.constant(
        model_builder.STDDEV_RGB, shape=[1, 1, 3], dtype=features.dtype)
    logits, _ = model_builder.build_model(
        features, self.model_name, is_training)
    probs = tf.nn.softmax(logits)
    probs = tf.squeeze(probs)
    return probs

  def get_preprocess_fn(self):
    """Build input dataset."""
    return preprocessing.preprocess_image


def get_eval_driver(model_name, include_background_label=False):
  """Get a eval driver."""
  if model_name.startswith('efficientnet'):
    _, _, image_size, _ = efficientnet_builder.efficientnet_params(model_name)
  else:
    raise ValueError(
        'Model must be either efficientnet-b* or efficientnet-edgetpu*')

  return EvalCkptDriver(
      model_name=model_name,
      batch_size=1,
      image_size=image_size,
      include_background_label=include_background_label)


# FLAGS should not be used before main.
FLAGS = flags.FLAGS


def main(unused_argv):
  tf.logging.set_verbosity(tf.logging.ERROR)
  driver = get_eval_driver(FLAGS.model_name, FLAGS.include_background_label)
  driver.eval_imagenet(FLAGS.ckpt_dir, FLAGS.imagenet_eval_glob,
                         FLAGS.imagenet_eval_label, FLAGS.num_images,
                         FLAGS.enable_ema, FLAGS.export_ckpt)

if __name__ == '__main__':
  app.run(main)
