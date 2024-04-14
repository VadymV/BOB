# BOB (Be aware of your Own Brain 🧠)

Since my recent projects cannot be made publicly available (at least for now), I have decided to create a sample project
that could give some insights regarding my recent work.
In this project, I will write production-ready code - the code that my all other publicly available repositories do not
have.

The goal of the project BOB is to identify valence (positivity 😂 or negativity 😡) from brain signals. Why? Because we
are on the frontier of technological advances and novel applications will help us to better understand human needs,
behaviour, and decision making.

Stay tuned 🤓 for the updates. The next update is scheduled for 14.04.2024 (**postponed to 15.04.2024**) ⏲️.

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
- The script ```run.py``` trains a representation model in a self-supervised way without labels.
- The training process can be seen on the W&B (https://wandb.ai/site) dashboard (you need to create an account).

### Next steps:
- A YAML file containing the hyperparameters (i.e. epochs, batch size, and other) and parameters.
- A linear model that is trained in a supervised way (labels for valence are used) from the features learnt from by the
  representation model. This model will classify brain data into two classes: positivity or negativity.
- A baseline model that is trained in a supervised way (labels for valence are used) from engineered features. This
  model will classify brain data into two classes: positivity or negativity.
- Statistical evaluation of the results achieved by linear and baseline models. Here, we want to evaluate which model
  performs better. For this, statistical significance of results will be reported using p-value and effect size.

#### Afterwards, I will deploy Kubeflow and Ray on Google Kubernetes Engine (GKE) for smooth training and serving of the models.

This is an open-source project done in my free time. 
I have a full time job, so I will proceed as I have time. 
