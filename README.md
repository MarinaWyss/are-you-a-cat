# Are You A Cat?
### An MLOps Project for the [ZenML Month of MLOps Competition](https://blog.zenml.io/mlops-competition/)

![Test status](https://github.com/MarinaWyss/zenml-mlops-competition/workflows/run-tests/badge.svg)

![my cat](https://github.com/MarinaWyss/are-you-a-cat/blob/main/_assets/cat.jpg?raw=true)

#### Problem Statement
Sometimes it is hard to know if you are a cat or not. The goal of this project is to use deep learning to help with that.

J/K. It's really about me practicing some MLOps and deep learning stuff. There's a lot more I want to do to improve this pipeline, so my plans for future iterations are included in the project description below.

#### The Solution

![pipeline](https://github.com/MarinaWyss/are-you-a-cat/blob/main/_assets/pipeline.png?raw=true)

This project uses [ZenML](https://zenml.io/home) to build a simple end-to-end pipeline for a model to identify if a photo (selfie) uploaded to a Streamlit app is of a cat or not.

##### Training Pipeline

The training pipeline is pretty simple: Clean the data, train the model, and evaluate model performance on the test set.

###### Data

Training data for this project comes from the following datasets:
- [Cats and dogs](https://github.com/rpeden/cat-or-not/releases)
- [Selfies](https://www.kaggle.com/datasets/jigrubhatt/selfieimagedetectiondataset)
- [Random images](https://www.kaggle.com/datasets/shamsaddin97/image-captioning-dataset-random-images?resource=download)

I'm using a random sample from each of the above. For training, I used 25% cat images, 25% dogs, 25% selfies, and 25% misc.

Data prep is simple: The images are reshaped and normalized. That's it for now!

###### Model

The model for this project is a 2D CNN, implemented with Tensorflow Keras. The model, hyperparameters, and training metrics are saved using MLflow autologging as experiment tracking artifacts.

The model's hyperparameters were tuned using `keras_tuner`. The best model configuration achieves recall of 0.8 and precision of 0.53 on a hold-out test set. Note that there are several limitations to the current model training, which are noted below, so the performance can certainly be improved on.

###### Evaluation

The trained model predicts on a hold-out validation set, and logs those metrics to MLflow as well.

##### Deployment Pipeline

The deployment pipeline extends the training pipeline and implements a continuous deployment workflow. It preps the input data, trains a model, and (re)deploys the prediction server that serves the model if it meets some evaluation criteria (minimum recall and precision).

###### Deployment Trigger

After the model is trained and evaluated, the deployment trigger step checks whether the newly-trained model meets the criteria set for deployment.

###### Model Deployer

This step deploys the model as a service using MLflow (if deployment criteria is met). 

The MLflow deployment server runs locally as a daemon process that will continue to run in the background after the example execution is complete. When a new pipeline is run which produces a model that passes the evaluation checks, the pipeline automatically updates the currently-running MLflow deployment server to serve the new model instead of the old one.

##### Inference Pipeline

This project primarily uses a Streamlit application for inference, but it contains a separate inference pipeline for testing as well.

##### Streamlit Application

For inference, I have a simple [Streamlit application](https://marinawyss-are-you-a-cat-appstreamlit-app-zbvy1b.streamlitapp.com/) that consumes the latest model service asynchronously from the pipeline logic.

The Streamlit app takes in a photo (selfie), and returns the probability that you are a cat. In the future I will add a SHAP explanation of why the prediction was made for this particular photo.

#### Limitations

This pipeline is a simple first pass, and has some major limitations. Some things I plan to incorporate in the future include:

- Model/Training:
  - Currently, I'm training locally (no GPU!) on a small dataset. In the future, training should be done in the cloud (i.e. Sagemaker) with more data/maybe for longer.
  - Hyperparameter tuning is ad-hoc and manual. ZenML is planning hyperparameter tuning support in the future, so I'll add that once it's available.
  - I haven't done any in-depth model performance work yet (e.g. no error analysis or real experimentation). I haven't even tuned the classification cut-off.
  - Model training is slow. This may be because I'm using a dinky laptop at the moment, but there may be ways to improve the efficiency of model training.
- Deployment:
  - Deployment is currently happening locally via MLflow. In the future I will migrate this to Seldon or KServe for deployment in a more production-friendly setting.
- Monitoring:
  - I don't have anything set up besides basic logging yet >.<
    - And I haven't even set up the logs to save after a run...
  - I have no idea how the model will perform on real data (which may differ significantly from the training set I used). I also have no way to track this/gather feedback or more labels.
  - Similarly, because I don't know what the input data look like, I cannot do any drift detection.
  - I also don't know any performance metrics for the app.
  - There is also no way for me to know about errors users get from the app. For example, I don't know what happens if someone uploads a document instead of a photo. I won't know if something breaks and the app is down. And I've done very little to make the app fail gracefully in the case of user error.
- Orchestration:
  - Running the pipeline is currently ad-hoc and manual. In the future if I was able to store incoming images/build the training dataset in some automated way, I could schedule periodic re-training with Airflow.
    - It would be even cooler if I could do some kind of continual learning approach so that I don't need to store the images at all...
- Misc:
  - Sphinx docs would be nice.
  - Dockerizing might make sense.
  - There are likely opportunities for better testing, since currently the tests I have are quite minimal.
