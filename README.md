# BOB (Be aware of your Own Brain 🧠)
Since my recent projects cannot be made publicly available (at least for now), I have decided to create a sample project that could give some insights regarding my recent work.
In this project, I will write production-ready code - the code that my all other publicly available repositories do not have.

The goal of the project BOB is to identify valence (positivity 😂 or negativity 😡) from brain signals. Why? Because we are on the frontier of technological advances and novel applications will help us to better understand human needs, behaviour, and decision making.  

Stay tuned 🤓 for the updates. The next update is scheduled for 07.04.2024 ⏲️.

----

#### The 07.04.2024 update will contain the following:
- A model that is trained in a contrastive self-supervised way without using the labels for valence. This model will learn features from brain data.
- A linear model that is trained in a supervised way (labels for valence are used) from the features learnt from by the model in a previous step. This model will classify brain data into two classes: positivity or negativity.
- A baseline model that is trained in a supervised way (labels for valence are used) from engineered features. This model will classify brain data into two classes: positivity or negativity.
- Statistical evaluation of the results achieved by linear and baseline models. Here, we want to evaluate which model performs better. For this, statistical significance of results will be reported using p-value and effect size.

#### Afterwards, I will deploy Kubeflow and Ray on Google Kubernetes Engine (GKE) for smooth training and serving of the models.

