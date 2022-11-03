import os
import sys
import yaml
import s3fs
import datetime

import numpy as np
import pandas as pd

from PIL import Image
import tensorflow as tf

import streamlit as st

# from zenml.services import load_last_service_from_step

parent = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent)
# from run_deployment_pipeline import run_main


def main():
    # Set up
    with open('steps/config.yaml', 'r') as file:
        configs = yaml.safe_load(file)
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

    if uploaded_file is not None:
        u_img = Image.open(uploaded_file).convert('L')  # grayscale
        show.image(u_img, caption='This is you.', use_column_width=True)
        # Prepare image for prediction
        # TODO input validation
        resized = u_img.resize((configs['image_size'], configs['image_size']))
        pred_image = np.asarray(resized) / 255.
        pred_image = pred_image.reshape(
            -1, configs['image_size'], configs['image_size'], 1)

    st.sidebar.write('\n')
    if st.sidebar.button("Predict"):

        if uploaded_file is None:
            st.sidebar.write("Upload a selfie first.")

        else:
            # There is a problem installing zenml with Streamlit at the moment.
            # This open PR *should* solve the problem:
            # https://github.com/zenml-io/zenml/pull/888

            # I will add this code back in (it works locally) once the PR
            # is done.

            # service = load_last_service_from_step(
            #    pipeline_name="continuous_deployment_pipeline",
            #    step_name="model_deployer",
            #    running=True,
            # )
            # if service is None:
            #    st.write("No service could be found. \
            #        The pipeline will be run first to create a service.")
            #    run_main()

            model = tf.keras.models.load_model(f"saved_model/{configs['best_model']}")

            with st.spinner('Classifying...'):
                # prediction = service.predict(pred_image)[:, 0].item()
                prediction = model.predict(pred_image)[:, 0].item()
                st.success('Done!')

                # Save image to s3 for monitoring
                time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
                path = f"{configs['s3_bucket']}/{configs['uploads_key']}/{time}"
                u_img.save(s3.open(f"{path}.png", 'wb'), 'PNG')

                if isinstance(prediction, float):
                    # TODO add SHAP
                    if prediction < configs['min_unsure']:
                        st.sidebar.write("I'm pretty sure you're not a cat.")
                    if configs['min_unsure'] <= prediction < configs['max_unsure']:
                        st.sidebar.write("I don't think you're a cat, but it's hard to tell.")
                    if prediction >= configs['max_unsure']:
                        st.sidebar.write("I'm pretty sure you are a cat.")

                    # Gather user feedback for monitoring/future training
                    feedback = st.sidebar.radio(
                        "Am I right?",
                        ('Yes', 'No'),
                        horizontal=True
                    )
                    # Save feedback to s3
                    if st.sidebar.button("Submit feedback.")
                        # TODO figure out a slicker way to do this
                        df = pd.DataFrame({'feedback': feedback}, index=[0])
                        with s3.open(f"{path}.csv", 'wb') as f:
                            df.to_csv(f)

                else:
                    st.sidebar.write("Something went wrong.")
                    # Also capture if something went wrong
                    df = pd.DataFrame({'feedback': '-999'}, index=[0])
                    with s3.open(f"{path}.csv", 'wb') as f:
                        df.to_csv(f)


if __name__ == "__main__":
    main()
