#!/bin/bash

PROJECT_ID="gwx-internship-01"
REGION="us-east1"
SERVICE_NAME="keboli-interview-agent" 
GAR_REPO="us-east1-docker.pkg.dev/$PROJECT_ID/gwx-gar-intern-01"
IMAGE="$GAR_REPO/$SERVICE_NAME:latest"

echo "Deploying $SERVICE_NAME..."

docker build -t $IMAGE .
docker push $IMAGE

gcloud run deploy $SERVICE_NAME \
  --image=$IMAGE \
  --region=$REGION \
  --allow-unauthenticated \
  --project=$PROJECT_ID \
  --platform=managed \
  --port=8001 \
  --max-instances=2 \
  --min-instances=0 \
  --min=0 \
  --max=2 \
  --cpu=1 \
  --memory=4Gi \
  --service-account gwx-cloudrun-sa-01@gwx-internship-01.iam.gserviceaccount.com

echo "$SERVICE_NAME is live!"