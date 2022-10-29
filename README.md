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

I'm using a random sample from each of the above. For training I used 25% cat images, 25% dogs, 25% selfies, and 25% misc.

Data prep is simple: The images are reshaped and normalized. That's it for now!

###### Model

The model for this project is a 2D CNN, implemented with Tensorflow Keras. The model and training metrics are saved using MLFlow autologging.

###### Evaluation

The trained model predicts on a hold-out validation set, and logs those metrics to MLFlow as well.

##### Deployment Pipeline

The deployment 


