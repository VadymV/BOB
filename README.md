# BOB (Be aware of your Own Brain 🧠)

Since my recent projects cannot be made publicly available (at least for now), I have decided to create a sample project
that could give some insights regarding my recent work.
In this project, I will write production-ready code - the code that my all other publicly available repositories do not
have.

The goal of the project BOB is to identify valence (positivity 😂 or negativity 😡) from brain signals. Why? Because we
are on the frontier of technological advances and novel applications will help us to better understand human needs,
behaviour, and decision-making. 
Here, I show how
- to train different models. See [README.md](./examples/DEAP/README.md)
- to perform affective ranking of images according to their affective similarity
to the brain response of a source image


In short, the steps are as follows:
1. Create a packaged Python code
2. Deploy containerized code to an endpoint (e.g., Cloud Run, Kubernetes)
3. Let users evaluate affective ranking of images by accessing that endpoint
4. Focus on deployable and manageable ML code. For this, I will use Kubeflow

Stay tuned 🤓 for the updates.

The next update will include the serving of the model
for the affective ranking of the images. The idea is the following. The user can upload some images and trigger
the model to produce affective ranking of the uploaded images according to their similarity to the brain response of a 
source image. Here, the user triggers model's predictions via the web application that runs as a separate service
(container image on Cloud Run, FastApi). The model's predictions are produced by a separate service
using the PyTorch Serve operation that runs on Kubernetes. Such decoupling makes the system maintainable and less
error-prone.

----


### Containerize the code and deploy to Cloud Run to run the prediction application:
- sudo usermod -a -G docker ${USER} 
- export PROJECT=<project-id>
- export LOCATION=europe-west3
- export REPO_NAME=bob-repo
- gcloud artifacts repositories create ${REPO_NAME} --repository-format=docker \
    --location=${LOCATION} --description="Docker repository" \
    --project=${PROJECT} 
- gcloud auth configure-docker ${LOCATION}-docker.pkg.dev 
- docker build --tag bob .
- docker run --detach --publish 3100:3100 bob
- docker tag bob ${LOCATION}-docker.pkg.dev/${PROJECT}/${REPO_NAME}/bob:latest 
- docker push ${LOCATION}-docker.pkg.dev/${PROJECT}/${REPO_NAME}/bob:latest
- gcloud run deploy --image=${LOCATION}-docker.pkg.dev/${PROJECT}/${REPO_NAME}/bob:latest --platform=managed --region=${LOCATION} --port=3100 --max-instances=1


### Deploy the model to GKE autopilot cluster and use PyTorch Serving:
- export MACHINE="cpu"
- export MODEL_NAME="bob"
- export MODEL_VERSION="1.0"
- gcloud container clusters create-auto ml-cluster \
    --release-channel=regular \
    --cluster-version=1.28.7-gke.1026000 \
    --location=${LOCATION}
- gcloud artifacts repositories create models \
    --repository-format=docker \
    --location=${LOCATION} \
    --description="Repo for BOB serving image"
- gcloud artifacts repositories describe models \
    --location=${LOCATION}
- TODO: Build and push the image using Cloud Build
- TODO: Deploy the packaged model to GKE

### Affective ranking of images:
- TODO

This is an open-source project done in my free time. 
I have a full-time job, so I will proceed as I have time.
