import logging
import sys
import tensorflow as tf
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
import tensorflow_datasets as tfds

# FIXME TODO: update location where model needs to be loaded from
MODEL_LOCATION = "/afs/crc.nd.edu/user/a/amaltar2/tensor_vgg19/vgg19_trained"
LOG_LEVEL = logging.DEBUG

def setup_logger():
    root = logging.getLogger()
    root.setLevel(LOG_LEVEL)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(LOG_LEVEL)
    formatter = logging.Formatter("%(asctime)s:%(levelname)s:%(module)s:%(message)s")
    handler.setFormatter(formatter)
    root.addHandler(handler)
    return root


def preprocess(image, label):
    image = tf.image.resize(image, [224, 224])  # Resize the image to the required input size for VGG19
    image = tf.cast(image, tf.float32) / 255.0  # Normalize the pixel values
    return image, label


logger = setup_logger()
logger.info(f"Using tensorflow version: {tf.__version__}")

# print GPU devices configuration
gpu_devices = tf.config.list_physical_devices('GPU')
logger.info(f"Retrieved GPU devices: {gpu_devices}")
if gpu_devices:
    details = tf.config.experimental.get_device_details(gpu_devices[0])
    logger.info(f"GPU details: {details}")

model = tf.keras.models.load_model(MODEL_LOCATION)
logger.debug(f"Attributes of model: {dir(model)}")
logger.info(f"Loaded model summary: {model.summary()}")

# Loading the input dataset
inputDset = 'imagenette'  # 'imagenette/160px-v2'
logger.info(f"Loading tensorflow dataset name: {inputDset}.")
# doc: https://www.tensorflow.org/datasets/api_docs/python/tfds/load
tfds.disable_progress_bar()
input_ds, metadata = tfds.load(inputDset,
                               split=['validation'],
                               shuffle_files=True,
                               as_supervised=True,
                               with_info=True)

logger.debug(f"Dataset val_ds type: {type(input_ds)}")
logger.debug(f"Dataset val_ds type first: {type(input_ds[0])}")
logger.debug(f"Dataset val_ds first: {input_ds[0]}")
logger.debug(f"Dataset metadata: {metadata}")

logger.info("Starting training of VGG19 with imagenette dataset.")
#input_ds = map(preprocess, input_ds)
input_ds = input_ds.map(preprocess).batch(32).prefetch(buffer_size=tf.data.AUTOTUNE)

logger.info(f"Dataset val_ds type: {type(input_ds)}")

# Evaluate the restored model
#loss, acc = model.evaluate(val_ds, metadata, verbose=2)
array_predictions = model.predict(input_ds, verbose=2)
logger.info(f"Predictions: {array_predictions}")

logger.info(model.predict(input_ds).shape)