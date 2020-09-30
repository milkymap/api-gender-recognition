# api-face-gender-recognition
an api for recognizing the gender(man, woman) by analyzing an input image

# build with docker? 
There is two step for lunching the app with docker 
The first step  is to build : 
docker build -f Dockerfile.yml -t recogender .

The second step is to start the app
docker run --rm --name gender_predictor -p 80:8080 recogender:latest
