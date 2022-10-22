import os

from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri


def open_mlflow_ui(port=4997):
    os.system(f'mlflow ui --backend-store-uri="{get_tracking_uri()}" --port={port}')


if __name__ == '__main__':
    open_mlflow_ui()
