# llama4 with together api
model="meta-llama/Llama-4-Scout-17B-16E-Instruct"
# model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
api_key=$TOGETHER_API_KEY
llm_endpoint_url="https://api.together.ai"

# # llama3.3 with vllm-gaudi
# model="meta-llama/Llama-3.3-70B-Instruct"
# api_key="EMPTY"
# llm_endpoint_url="http://localhost:8086"

output="${WORKDIR}/datasets/financebench/results/long_context_arceep1.json"

python long_context_generate.py \
--output $output \
--model $model \
--api_key $api_key \
--llm_endpoint_url $llm_endpoint_url \
--temperature 0.0 \
--read_processed