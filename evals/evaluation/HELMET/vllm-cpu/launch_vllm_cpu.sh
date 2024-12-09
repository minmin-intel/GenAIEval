model="meta-llama/Meta-Llama-3.1-8B-Instruct"
MAX_INPUT=65536
vllm_port=8000
vllm_volume=$HF_CACHE_DIR
echo "token is ${HF_TOKEN}"
LOG_PATH=$WORKDIR

echo "start vllm cpu service"
echo "**************model is $model**************"
docker run -d --rm --name "vllm-cpu-server" -p $vllm_port:8000 -v $vllm_volume:/data -e VLLM_CPU_KVCACHE_SPACE=80 -e HF_TOKEN=$HF_TOKEN -e HUGGING_FACE_HUB_TOKEN=$HF_TOKEN -e HF_HOME=/data -e http_proxy=$http_proxy -e https_proxy=$https_proxy -e no_proxy=$no_proxy --ipc=host vllm-cpu-env --model ${model} #--max-seq-len-to-capture $MAX_INPUT
sleep 5s
echo "Waiting vllm cpu ready"
n=0
until [[ "$n" -ge 100 ]] || [[ $ready == true ]]; do
    docker logs vllm-cpu-server &> ${LOG_PATH}/vllm-cpu-service.log
    n=$((n+1))
    if grep -q "Uvicorn running on" ${LOG_PATH}/vllm-cpu-service.log; then
        break
    fi
    if grep -q "No such container" ${LOG_PATH}/vllm-cpu-service.log; then
        echo "container vllm-cpu-server not found"
        exit 1
    fi
    sleep 5s
done
sleep 5s
echo "Service started successfully"