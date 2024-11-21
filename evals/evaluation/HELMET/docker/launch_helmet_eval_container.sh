volume=$WORKDIR
host_ip=$(hostname -I | awk '{print $1}')

docker run -it --name helmet_eval -v $volume:/home/user/ -e WORKDIR=/home/user -e HF_HOME=$HF_CACHE_DIR -e HUGGING_FACE_HUB_TOKEN=$HF_TOKEN -e host_ip=$host_ip -e http_proxy=$http_proxy -e https_proxy=$https_proxy helmet-eval:latest
