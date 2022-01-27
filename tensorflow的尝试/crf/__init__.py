import tensorflow as tf 
from .crf import CRF

tf.keras.utils.get_custom_objects().update(
    {'CRF': CRF}
)