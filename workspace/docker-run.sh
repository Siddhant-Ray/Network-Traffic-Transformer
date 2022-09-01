# https://stackoverflow.com/questions/30543409/how-to-check-if-a-docker-image-with-a-specific-tag-exist-locally
if [[ "$(docker images -q siddhantray/ntt-docker:latest 2> /dev/null)" == "" ]]; then
    docker run -it ntt-docker:latest
else
    docker pull siddhantray/ntt-docker
    docker run -it siddhantray/ntt-docker
fi