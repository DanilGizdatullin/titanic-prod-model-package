from sklearn.model_selection import train_test_split

from classification_model.config.core import config
from classification_model.pipeline import titanic_pipe
from classification_model.processing.data_manager import load_dataset, save_pipeline


def run_training() -> None:
    """Train the model."""

    data = load_dataset(file_name=config.app_conf.training_data_file)

    X_train, X_test, y_train, y_test = train_test_split(
        data[config.model_conf.features],
        data[config.model_conf.target],
        test_size=config.model_conf.test_size,
        random_state=config.model_conf.random_state,
    )

    titanic_pipe.fit(X_train, y_train)

    save_pipeline(pipeline_to_persist=titanic_pipe)


if __name__ == "__main__":
    run_training()
