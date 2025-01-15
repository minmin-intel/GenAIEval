# host_ip=$(hostname -I | awk '{print $1}')
#port=8085 # change it to the actual port that vllm container is running at
#llm_endpoint_url="http://${host_ip}:${port}/v1"
#echo "llm_endpoint_url is $llm_endpoint_url"

python3 eval.py --config configs/rag_test_short_cpu.yaml
