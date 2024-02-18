"""
    This module builds model defined in a configuration files
"""
import importlib
import logging
import os

from tensorflow.keras.models import load_model

from ml_investing_wne import config

logger = logging.getLogger(__name__)

# load model dynamically
try:
    build_model = getattr(importlib.import_module(f'ml_investing_wne.tf_models.{config.model}'),'build_model')
except:
    logger.error(f'build_model for {config.model} not implemented!')

def model_factory(input_shape=None):
    """_summary_

    Args:
        X (_type_): _description_

    Returns:
        _type_: _description_
    """
    if len(config.load_model) > 1:
        model = load_pretrained_model()
    elif config.model == 'transformer_learnable_encoding':
        model = transformer(input_shape)
    elif config.model in ['resnet', 'resnet_lstm_regularized', 'inception', 'lstm', 'resnet_lstm_regularized_tunned', 'resnet_lstm_regularized_tunned_small', 'keras_tuner_CNN_LSTM', 'keras_tuner_tsmixer']:
        model = cnn(input_shape)

    return model

def transformer(input_shape):
    """

    Args:
        X (_type_): _description_

    Returns:
        _type_: _description_
    """
    model = build_model(input_shape=(input_shape[0], input_shape[1]), head_size=64, num_heads=4,
                        ff_dim=64, num_transformer_blocks=4, mlp_units=[128], mlp_dropout=0.25,
                        dropout=0.25)
    return model

def cnn(input_shape):
    """_summary_

    Args:
        X (_type_): _description_

    Returns:
        _type_: _description_
    """
    model = build_model(input_shape=(input_shape[0], input_shape[1]), nb_classes=config.nb_classes)
    return model

def load_pretrained_model():
    """_summary_

    Returns:
        _type_: _description_
    """
    if len(config.load_model) > 1:
        try:
            model_name = f'''{config.model}_hist_data_{config.load_model}_{config.freq}_
                             {config.steps_ahead}_{config.seq_len}'''
            # models have to be moved to be in production folder in order to be reused
            model = load_model(os.path.join(config.package_directory, 'models', 'production',
                                            model_name))
            return model
        except FileNotFoundError:
            logger.error('Model that you want to load does not exits')
