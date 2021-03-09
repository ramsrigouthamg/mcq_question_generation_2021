# Deployment using Cortex
Know more about cortex here : https://www.cortex.dev/


##Traditional Deployment pipeline (on AWS as an example):

Model inference code -> create Flask API -> Add Gunicorn to be production-ready -> create a Docker container -> deploy Docker on container orchestration service (EKS, ECS, Fargate etc) -> logging, autoscaling, load balancing -> infrastructure for rolling updates (without taking down current API)

##Cortex ML deployment pipeline:
Model inference code -> Cortex

Cortex automates the whole traditional deployment process. If you just provide your AWS credentials, you can use Cortex’s command-line interface to deploy an API to production just starting from model inference code.

Cortex is open-source, easy-to-use, built for scale, and also supports AWS spot instances to save cost.


## Steps for deployment 

1) Download s2v_reddit_2015_md from: https://github.com/explosion/sense2vec
2) Extract the zip folder rename it as "s2v_old" and place it in this same folder.
3) Install Cortex on command line from the Cortex website 
4) On command line run "cortex deploy". This will create a local deployment and serve the api locally.
5) Run "cortex get" to know the status of API deployment. 
6) Once it is live you can run - 'curl http://localhost:8888 -X POST -H "Content-Type: application/json" -d sample.json' to check if the API is working.
7) Once you make sure API is working, you can go ahead with deployment on AWS.
8) To deploy on AWS follow the instructions on Cortex to create cluster and deploy on aws with the command "cortex deploy --env aws"
9) Make sure you update AWS credentials in cluster.yaml 