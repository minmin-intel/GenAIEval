# host_ip=$(hostname -I | awk '{print $1}')
port=8085 # change it to the actual port that vllm container is running at
llm_endpoint_url="http://${host_ip}:${port}/v1"
echo "llm_endpoint_url is $llm_endpoint_url"
config_file=configs/rag_test/rag_short_test_8k.yaml

python3 eval.py --config $config_file --endpoint_url $llm_endpoint_url