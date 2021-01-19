# Spam Classifier Pipeline with Airflow 2.0

In this project, I've used Airflow 2.0 to create a pipeline that creates and deploys a Machine Learning model to classify e-mails as spam or not.

### What’s Airflow?

[Airflow](airflow.apache.org/) is an open-source workflow management platform. It started at Airbnb in October 2014 as a solution to manage the company's increasingly complex workflows. Creating Airflow allowed Airbnb to programmatically author and schedule their workflows and monitor them via the built-in Airflow user interface. From the beginning, the project was made open source, becoming an Apache Incubator project in March 2016 and a Top-Level Apache Software Foundation project in January 2019.

Airflow is written in Python, and workflows are created via Python scripts. Airflow is designed under the principle of "configuration as code".

In Airflow, a DAG — or a Directed Acyclic Graph — is a collection of all the tasks you want to run, organized in a way that reflects their relationships and dependencies.
Airflow uses Python language to create its workflow/DAG file, it’s quite convenient and powerful for the developer ([Wikipedia](https://en.wikipedia.org/wiki/Apache_Airflow)).

## How to run this project

This project has a DAG with 4 tasks that create a pipeline to:
* Download the training set from some URL
* Clean the data
* Train and test a model to classify e-mails as spam or not.
* Deploy a demo web app locally to test the model.


<img width="500" alt="image" src="https://user-images.githubusercontent.com/22003608/104956981-00f16880-59ac-11eb-98f3-73de74ecb2f9.png">

### Docker

The fastest way to install and run Airflow is using [Docker](www.docker.com).

Clone this project:
```
https://github.com/cpatrickalves/airflow2-spam-classifier-pipeline.git
```

Build a docker image with Airflow 2.0 running on Python 3.7:

```
docker build . -t spam-classifier-airflow2:1.0
```

Than run the docker image:

```
docker run --rm -p 8080:8080 -p 80:80 spam-classifier-airflow2:1.0
```

The `-p` flags above are used to open the ports 8080 and 80 of the docker container. These ports are used by Airflow and the demo web app, respectively.

Once you run the above command, just open a browser and go to `localhost:8080`. The Airflow UI should load.

Now, follow these steps:
* login with username `admin` and password `admin`.
* Enable the DAG by clicking on the circle button.

<img width="250" alt="image" src="https://user-images.githubusercontent.com/22003608/104957735-7c074e80-59ad-11eb-8411-bb6531eb2281.png">

* Click on the "Play" button to start the Pipeline (if another page called Trigger DAG appears, just click **Trigger**)

<img width="250" alt="image" src="https://user-images.githubusercontent.com/22003608/104957880-c1c41700-59ad-11eb-8550-80eb4a5f8aa3.png">

* Click on the DAG name and then on the Graph View, you should see this:

<img width="500" alt="image" src="https://user-images.githubusercontent.com/22003608/104958127-3ac36e80-59ae-11eb-9df3-19df5da7b8ff.png">

* When all boxes become green, your pipeline is finished.
* You can open [localhost](http://localhost) in your browser to test the demo web app.
* You can play with some texts to see if they are classified as spam or not.

<img width="600" alt="image" src="https://user-images.githubusercontent.com/22003608/104958512-ed93cc80-59ae-11eb-8f4b-09ef3377e11f.png">

## The Spam Classifier model

This is a baseline project to create a Machine Learning Pipeline using Airflow.
With this in mind, the machine learning model used was very simple.
The model was built using two approaches:
* Based on word presence (whether a word appears in the document or not, which will make the input attributes binary)
* Based on word frequency (frequency of word occurrence in the document, which will make the input attributes continuous)

You can check the implementation details at the source code in `dags/spam/NaiveBayesSolver.py`.

## Possible improvements
There are dozens of possible improvements for this project.

For example:
* You can add your own model implementation using some state-of-the-art solutions.
* You can change the source data to train the model.
* You can set up an e-mail server to send e-mails in case of failures in the pipeline.
* You can update the `deploy_app` step to build another container and deploy it on the Cloud.

---
## Built With
* [Python](https://www.python.org/) - Programming Language.
* [Airflow](airflow.apache.org/) - A platform created by the community to programmatically author, schedule, and monitor workflows.
* [Streamlit](https://www.streamlit.io/) - Open-source app framework for Machine Learning and Data Science teams.
