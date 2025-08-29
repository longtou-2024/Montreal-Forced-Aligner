#docker build -t longtou/mfa:lt .
docker build -t longtou/mfa:lt -f Dockerfile.lite .
docker tag longtou/mfa:lt us-central1-docker.pkg.dev/prod-ai-project/tts/mfa:lt

#docker run -it --runtime=nvidia longtou/mfa:lt /bin/bash
#gcloud auth print-access-token | docker login -u oauth2accesstoken --password-stdin https://us-central1-docker.pkg.dev
#docker push us-central1-docker.pkg.dev/prod-ai-project/tts/mfa:lt
