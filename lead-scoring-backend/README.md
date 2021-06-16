#### Table of content
- [LEAD SCORING BACKEND](#lead-scoring-backend)
  - [Folder Structure](#folder-structure)
    - [app](#app-define-api)
- [API](#api)
  - [Setup for develop](#setup-for-develop)
    - [Docker Local env](#docker-Local-env)
    - [Setup on local machine](#setup-on-local-machine)
  - [API Components](#api-components)
    - [API Docs](#api-docs)
    - [API info](#api-info)
    - [API predict](#api-predict)   
- [Model](#model)



# LEAD SCORING BACKEND
This repository is used to contain all implementation codes related to the lead scoring platform for the B2C Vietnam team.

## Folder Structure
### app Define API
Store API & config base on FastAPI framework:
- config.py: Store config for API, .env interaction.
- routers.py: Define routers into APIs.
- api_key.py: Define security process apply to each apis need access token.
- test: Define test case for API. Base on module `TestClient` and `pytest`
- apis: Defines APIs supported by service.
- models: Models of lead-scoring-backend
- training: Training data and scripts


# API
Provide API for model predict Lead Scoring. Using FastAPI framework for implement.

Home page docs: https://fastapi.tiangolo.com/

## Setup for develop

### Setup env vars from .env_dev template
```
cp .env_dev .env
```

### Docker Local env
- Setup docker.
- Build docker image from Docker files.
    ```
    docker build -t fapi-lead-scoring .
    ```
- Running docker container build from image `fapi-lead-scoring`
    ```
    docker run -d --name fapi-lead-scoring -p 80:80 fapi-lead-scoring
    ```
- Checking status of API:
    - URL: `http://127.0.0.1:80/health`
    - Response: 
        `{"status":"OK","message":"I am healthy"}`

### Setup on local machine
Follow up FastAPI docs: https://fastapi.tiangolo.com/tutorial/#install-fastapi
- Running local server:
    ```
    ./local_server.sh
    ```
- Checking status of API:
    - URL: `http://127.0.0.1:8000/health`
    - Response: 
        `{"status":"OK","message":"I am healthy"}`

## API Components
### API Docs
- URL: `http://<IP>:<PORT>/docs`
- For local running: `http://127.0.0.1:80/docs` or `http://127.0.0.1:8000/docs`

### API info
#### Info
Get info about API Server
- URL: `http://<IP>:<PORT>/info`

#### Health Check
Get info health check
- URL: `http://<IP>:<PORT>/health`


### API predict
#### Predict Lead Score
- URL: `http://<IP>:<PORT>/v1/predict`
- Data Model:
    - FeatureModel: Define data schema on request body.
    - ResponseModel: Define data schema on response body

# Model
Lead Scoring Model detail