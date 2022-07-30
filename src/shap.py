import os
from keras.models import load_model
import ml_investing_wne.config as config
import shap

model = load_model(os.path.join(config.package_directory, 'models',
                                '{}_{}_{}.hdf5'.format(config.model, config.load_model, config.freq)))

explainer = shap.DeepExplainer(model, X)
explainer.shap_values(X)