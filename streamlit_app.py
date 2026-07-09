import os
import sys
import datetime

import s3fs
import numpy as np
import pandas as pd

from PIL import Image, UnidentifiedImageError
import tensorflow as tf

import streamlit as st

from steps.config_loader import load_config
from steps.preprocessing import image_to_model_input
from model.predictor import interpret_prediction

# from zenml.services import load_last_service_from_step

parent = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent)
# from run_deployment_pipeline import run_main

# Largest upload we are willing to open, in bytes. Guards against someone
# uploading an enormous file and exhausting memory.
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB


def validate_upload(uploaded_file) -> Image.Image:
    """Validates an uploaded file and returns it as a PIL image.

    This addresses the ``# TODO input validation`` in the original app: we now
    reject oversized files and anything that is not a decodable image instead
    of letting a malformed upload crash the prediction path.

    Args:
        uploaded_file: A Streamlit ``UploadedFile``.

    Returns:
        (PIL.Image.Image): The decoded image.

    Raises:
        ValueError: If the file is too large or is not a valid image.
    """
    if uploaded_file.size > MAX_UPLOAD_BYTES:
        raise ValueError(
            f"File is too large ({uploaded_file.size} bytes). "
            f"The limit is {MAX_UPLOAD_BYTES} bytes."
        )
    try:
        return Image.open(uploaded_file)
    except UnidentifiedImageError as exc:
        raise ValueError("That file does not look like a valid image.") from exc


def main():
    # Set up
    configs = load_config()
    s3 = s3fs.S3FileSystem(anon=False)

    st.title("Are you a cat?")

    high_level_image = Image.open("_assets/cat.jpg")
    show = st.image(high_level_image, use_column_width=True, caption="Well, are you?")

    whole_pipeline_image = Image.open("_assets/pipeline.png")
    st.markdown(
        """
     #### Problem Statement
     Sometimes it is hard to know if you are a cat or not. \n
     \n
     Upload a selfie and I will try to help.\n
     \n
     This is how I will do it.
     """
    )
    st.image(whole_pipeline_image, caption="Deep learning pipeline with ZenML.")
    st.markdown(
        """
    This is the pipeline. First, we ingest and prepare the training data. Then, \
    we train a convolutional neural network, evaluate its performance on a validation set, \
    and log the model configuration, hyperparameters, and evaluation metrics to MLFlow. \n
    \n
    If the model performance is satisfactory, the model is deployed. \n
    \n
    There is a lot more I plan to do to improve this pipeline. Details are in \
    the [Github repo README](https://github.com/MarinaWyss/are-you-a-cat), \
    so keep an eye out for updates.
    """
    )

    st.sidebar.title("Upload selfie.")
    st.set_option('deprecation.showfileUploaderEncoding', False)
    uploaded_file = st.sidebar.file_uploader(" ", type=['png', 'jpg', 'jpeg'])

    pred_image = None
    if uploaded_file is not None:
        try:
            image = validate_upload(uploaded_file)
        except ValueError as err:
            st.sidebar.error(str(err))
        else:
            u_img = image.convert('L')  # grayscale
            show.image(u_img, caption='This is you.', use_column_width=True)
            # Prepare image for prediction using the same preprocessing the
            # training pipeline uses.
            pred_image = image_to_model_input(image, configs['image_size'])

    st.sidebar.write('\n')
    if st.sidebar.button("Predict"):
        if pred_image is None:
            st.sidebar.write("Upload a valid selfie first.")
        else:
            model = tf.keras.models.load_model(f"saved_model/{configs['best_model']}")
            # There is a problem installing zenml with Streamlit at the moment.
            # This open PR *should* solve the problem:
            # https://github.com/zenml-io/zenml/pull/888

            # I will add this code back in (it works locally) once the PR
            # is done.

            # try:
            #     service = load_last_service_from_step(
            #         pipeline_name="continuous_deployment_pipeline",
            #         step_name="model_deployer",
            #         running=True,
            #     )
            # except KeyError:
            #     st.sidebard.write("No service could be found. The pipeline will run first to create a new service.")
            #     run_main()

            with st.spinner('Classifying...'):
                prediction = model.predict(pred_image)[:, 0].item()
                st.success('Done!')

                # Save image to s3 for monitoring
                time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
                path = f"s3://{configs['s3_bucket']}/{configs['uploads_key']}/{time}"
                Image.open(uploaded_file).convert('L').save(
                    s3.open(f"{path}.png", 'wb'), 'PNG')

                if isinstance(prediction, float):
                    # TODO add SHAP
                    verdict = interpret_prediction(prediction, configs)
                    st.sidebar.write(verdict.message)
                    feedback = _collect_feedback()
                else:  # If something went wrong with the model
                    st.sidebar.write("Something went wrong.")
                    feedback = '-999'

                _write_feedback(s3, path, feedback)


def _collect_feedback() -> str:
    """Renders the feedback widget and returns the user's response.

    Returns:
        (str): '1' for cat, '0' for not-cat, or None if not yet answered.
    """
    st.sidebar.write("Am I right?")
    if 'result' not in st.session_state:
        st.session_state.result = None
    if st.sidebar.button("Yep."):
        st.session_state.result = '1'
    elif st.sidebar.button("Nope."):
        st.session_state.result = '0'
    return st.session_state.result


def _write_feedback(s3, path: str, feedback: str) -> None:
    """Persists a feedback row to S3 for monitoring/future training.

    Args:
        s3 (s3fs.S3FileSystem): Connected filesystem.
        path (str): Base S3 path (without extension) for this prediction.
        feedback (str): Feedback value to store.
    """
    df = pd.DataFrame({'feedback': feedback}, index=[0])
    bytes_to_write = df.to_csv(None, index=False).encode()
    with s3.open(f"{path}.csv", 'wb') as f:
        f.write(bytes_to_write)


if __name__ == "__main__":
    main()
