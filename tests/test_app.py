import os
import sys
import types
import pytest
import flask
from unittest.mock import patch, MagicMock

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Create minimal stub modules for keras and sklearn so app can be imported
keras_mod = types.ModuleType('keras')
models_mod = types.ModuleType('keras.models')
models_mod.load_model = lambda *a, **k: None
keras_mod.models = models_mod
sys.modules['keras'] = keras_mod
sys.modules['keras.models'] = models_mod

sklearn_mod = types.ModuleType('sklearn')
metrics_mod = types.ModuleType('sklearn.metrics')
for name in ['mean_absolute_error','median_absolute_error','mean_squared_log_error','r2_score']:
    setattr(metrics_mod, name, lambda *a, **k: None)
sklearn_mod.metrics = metrics_mod
sys.modules['sklearn'] = sklearn_mod
sys.modules['sklearn.metrics'] = metrics_mod

import app

@pytest.fixture
def client():
    app.app.config['TESTING'] = True
    with app.app.test_client() as client:
        yield client

def test_root(client):
    resp = client.get('/')
    assert resp.status_code == 200
    assert resp.data.decode() == "Welcome to Acea Smart Water Analytics prediction"

def test_predict_lake(client):
    dummy_model = MagicMock()
    dummy_model.predict.return_value = [[1.0, 2.0]]
    with patch('app.load_model', return_value=dummy_model) as mock_load:
        with patch('flask.render_template', return_value='ok'):
            response = client.post('/predict', data={'waterbody': 'lake', 'rainfall': '1', 'temp': '2'})
    assert response.status_code == 200
    assert response.data.decode() == 'ok'
    mock_load.assert_called_once()
