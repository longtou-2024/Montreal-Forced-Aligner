#docker build -t longtou/mfa:lt .
#docker build -t longtou/mfa:lt -f Dockerfile.lite .
docker build -t longtou/mfa:lt -f Dockerfile.endpoint .

#docker tag longtou/mfa:lt us-central1-docker.pkg.dev/prod-ai-project/tts/mfa:lt
docker tag longtou/mfa:lt asia-northeast3-docker.pkg.dev/dev-ai-project-357507/tts/mfa:lt

#docker run -it --runtime=nvidia longtou/mfa:lt /bin/bash

#gcloud auth print-access-token | docker login -u oauth2accesstoken --password-stdin https://us-central1-docker.pkg.dev
#gcloud auth print-access-token | docker login -u oauth2accesstoken --password-stdin https://asia-northeast3-docker.pkg.dev

#docker push us-central1-docker.pkg.dev/prod-ai-project/tts/mfa:lt
docker push asia-northeast3-docker.pkg.dev/dev-ai-project-357507/tts/mfa:lt
