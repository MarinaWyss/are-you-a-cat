import yaml
import numpy as np
from PIL import Image
from skimage.transform import resize
import streamlit as st

from zenml.services import load_last_service_from_step

from run_deployment_pipeline import run_main


def main():
    with open('steps/config.yaml', 'r') as file:
        configs = yaml.safe_load(file)

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
        # TODO input validation
        pred_image = np.asarray(u_img) / 255.
        pred_image = resize(pred_image,
                            (configs['image_size'], configs['image_size']))
        pred_image = pred_image.reshape(
            -1, configs['image_size'], configs['image_size'], 1)

    st.sidebar.write('\n')
    if st.sidebar.button("Predict"):

        if uploaded_file is None:
            st.sidebar.write("Upload a selfie first.")

        else:
            service = load_last_service_from_step(
                pipeline_name="continuous_deployment_pipeline",
                step_name="model_deployer",
                running=True,
            )
            if service is None:
                st.write("No service could be found. \
                    The pipeline will be run first to create a service.")
                run_main()

            with st.spinner('Classifying...'):
                prediction = service.predict(pred_image)[:, 0].item()
                st.success('Done!')
                if isinstance(prediction, float):
                    # TODO add SHAP
                    if prediction > configs['classification_cutoff']:
                        st.sidebar.write("You are a cat.")
                        st.sidebar.write(f"Predicted probability of being a cat: \n"
                                         f"{round(prediction * 100, 2)}%")
                    else:
                        st.sidebar.write("You are not a cat.")
                        st.sidebar.write(f"Predicted probability of being a cat: \n"
                                         f"{round(prediction * 100, 2)}%")
                else:
                    st.sidebar.write("Something went wrong.")


if __name__ == "__main__":
    main()
