model="meta-llama/Meta-Llama-3.1-70B-Instruct"
vllm_port=8085
vllm_volume=${HF_CACHE_DIR}
LOG_PATH=${WORKDIR}

echo "start vllm gaudi service"
echo "**************model is $model**************"
docker run -d --runtime=habana --rm --name "vllm-gaudi-server" -e HABANA_VISIBLE_DEVICES=0,1,2,3 -p $vllm_port:80 -v $vllm_volume:/data -e HF_TOKEN=$HF_TOKEN -e HF_HOME=/data -e OMPI_MCA_btl_vader_single_copy_mechanism=none -e PT_HPU_ENABLE_LAZY_COLLECTIVES=true -e http_proxy=$http_proxy -e https_proxy=$https_proxy -e no_proxy=$no_proxy -e VLLM_SKIP_WARMUP=true --cap-add=sys_nice --ipc=host  vllm-gaudi:latest --model ${model} --host 0.0.0.0 --port 80 --block-size 128 --max-seq-len-to-capture 16384 --tensor-parallel-size 4
sleep 5s
echo "Waiting vllm gaudi ready"
n=0
until [[ "$n" -ge 100 ]] || [[ $ready == true ]]; do
    docker logs vllm-gaudi-server &> ${LOG_PATH}/vllm-gaudi-service.log
    n=$((n+1))
    if grep -q "Uvicorn running on" ${LOG_PATH}/vllm-gaudi-service.log; then
        break
    fi
    if grep -q "No such container" ${LOG_PATH}/vllm-gaudi-service.log; then
        echo "container vllm-gaudi-server not found"
        exit 1
    fi
    sleep 5s
done
sleep 5s
echo "vllm-gaudi-server started successfully"