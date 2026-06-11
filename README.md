# Face Mask Detection MLOps Project

## Overview

This project demonstrates an end-to-end MLOps workflow for face mask detection using YOLO models exported to ONNX format.

The project includes:

* Model inference using ONNX Runtime
* Command-line prediction module
* FastAPI-based inference service
* Dockerized deployment
* Support for multiple YOLO models

Future work includes deployment on a Linux server, Kubernetes orchestration, and Kubeflow integration.

---

## Project Structure

```text
.
├── api/
├── configs/
├── data/
│   ├── samples/
│   └── uploads/
├── deployment/
├── models/
│   ├── yolo11/
│   └── yolo26/
├── notebooks/
├── reports/
├── src/
│   ├── inference/
│   └── train/
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Available Models

| Model   | Format |
| ------- | ------ |
| YOLO26n | ONNX   |
| YOLO11s | ONNX   |

Models are stored in the `models/` directory.

---

## Local Installation

Create and activate a virtual environment:

```bash
python -m venv venv
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Command Line Inference

Run prediction using the default model:

```bash
python src/inference/predict.py
```

Run prediction with a custom image:

```bash
python src/inference/predict.py --image test.png
```

Run prediction with a specific model:

```bash
python src/inference/predict.py --image test.png --model yolo11s
```

Prediction outputs are stored in:

```text
reports/predictions/
```

---

## FastAPI Inference Service

Start the API server:

```bash
uvicorn api.main:app --reload
```

Open Swagger UI:

```text
http://127.0.0.1:8000/docs
```

### Predict Endpoint

```http
POST /predict
```

Parameters:

| Parameter | Description                   |
| --------- | ----------------------------- |
| file      | Input image                   |
| model     | Model name (default: yolo26n) |

Example:

```http
POST /predict?model=yolo11s
```

Response:

```json
{
  "timestamp": "...",
  "image_name": "...",
  "model_name": "...",
  "num_detections": 3,
  "detections": [...]
}
```

---

## Docker

Build image:

```bash
docker build -t facemask-api:v1 .
```

Run container:

```bash
docker run -p 8000:8000 facemask-api:v1
```

Access API:

```text
http://localhost:8000/docs
```

---

# Docker Hub

Image is published on Docker Hub:

```text
ngdangkhoa/facemask-api:v1
```

Pull image:

```bash
docker pull ngdangkhoa/facemask-api:v1
```

---

## Kubernetes (K3s)

Namespace:

```bash
kubectl create namespace facemask
```

Deploy application:

```bash
kubectl apply -f deployment/deployment.yaml
```

Create service:

```bash
kubectl apply -f deployment/service.yaml
```

Create ingress:

```bash
kubectl apply -f deployment/ingress.yaml
```

Check resources:

```bash
kubectl get all -n facemask
```

---

## Deployment Architecture

```text
Client
   │
   ▼
Traefik Ingress
   │
   ▼
Service
   │
   ▼
FastAPI Pod
   │
   ▼
ONNX Runtime
   │
   ▼
YOLO ONNX Models
```

---






## Technologies

* Python
* YOLO
* ONNX Runtime
* OpenCV
* FastAPI
* Docker
* Docker Hub
* Kubernetes (K3s)
* Traefik Ingress

---

## Roadmap

### Completed

* [x] Inference module
* [x] FastAPI application
* [x] Image upload endpoint
* [x] Multi-model inference
* [x] Docker containerization
* [X] Server deployment
* [x] K3s installation
* [x] Kubernetes Deployment
* [x] Kubernetes Service
* [x] Traefik Ingress

### In Progress

* [ ] Kubeflow integration

### Planned

* [ ] Kubeflow Pipelines
* [ ] Automated training workflow
* [ ] Model versioning
* [ ] Experiment tracking (MLflow)
* [ ] CI/CD pipeline
* [ ] Streamlit demo application
```
```
