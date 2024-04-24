# BOB (Be aware of your Own Brain 🧠)

Since my recent projects cannot be made publicly available (at least for now), I have decided to create a sample project
that could give some insights regarding my recent work.
In this project, I will write production-ready code - the code that my all other publicly available repositories do not
have.

The goal of the project BOB is to identify valence (positivity 😂 or negativity 😡) from brain signals. Why? Because we
are on the frontier of technological advances and novel applications will help us to better understand human needs,
behaviour, and decision making.

Stay tuned 🤓 for the updates. 

Update **24.04.2024**:
After experimenting with Azure, I have decided to use GCP. 
Google Cloud is more intuitive for me. I suppose for those who have background in Linux and software development,
GCP would be a better choice. However, it is my personal opinion and I have previously used GCP.
Furthermore, GCP is probably a preferred choice when it comes to what I want to achieve with this project.

The next update is scheduled on **28.04.2024** ⏲️.
- I will deploy a containerized FastAPI web app on Google Cloud and GKE to predict valence :trophy:.

Next steps:
Kubeflow? Yes, but it requires a more careful planning and is a bit too much, when you do not have to build such an
elaborate end-to-end ML lifecycle. Nevertheless, I intend to do this. 

----

### General setup:
- This project uses Poetry as packaging and dependency management tool
- Please see https://python-poetry.org/ 
- You can run ```poetry build``` to build the source and wheels archives

### Data setup:
- Download data: https://www.eecs.qmul.ac.uk/mmv/datasets/deap/
- Extract ```data_preprocessed_python.zip```
- Create a folder (I will refer to this folder as PROJECT_FOLDER) and place the extracted data into this folder.

### Current state:
- The script ```split_data.py``` preprocesses the data and splits the data into train/validation/test sets.
- The script ```train_ssl.py``` trains a representation model in a self-supervised way without labels.
- The folder ```examples``` contains examples on how to train other models and to evaluate them.
- The training process can be seen on the W&B (https://wandb.ai/site) dashboard (you need to create an account).

### Next update will include:
- Deploying Kubeflow and Ray on Google Kubernetes Engine (GKE) for smooth training and serving of the models.
- Statistical evaluation of the results achieved by linear and baseline models. Here, we want to evaluate which model
  performs better. For this, statistical significance of results will be reported using p-value and effect size.

This is an open-source project done in my free time. 
I have a full time job, so I will proceed as I have time. 
