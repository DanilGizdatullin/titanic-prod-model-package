import pytest
from sklearn.model_selection import train_test_split

from classification_model.config.core import config
from classification_model.processing.data_manager import load_raw_dataset


@pytest.fixture
def sample_input_data():
    data = load_raw_dataset(file_name=config.app_conf.training_data_file)

    X_train, X_test, y_train, y_test = train_test_split(
        data,
        data[config.model_conf.target],
        test_size=config.model_conf.test_size,
        random_state=config.model_conf.random_state,
    )

    return X_test
