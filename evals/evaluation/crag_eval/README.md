# CRAG Benchmark for Agent QnA systems
## Overview
[Comprehensive RAG (CRAG) benchmark](https://www.aicrowd.com/challenges/meta-comprehensive-rag-benchmark-kdd-cup-2024) was introduced by Meta in 2024 as a challenge in KDD conference. The CRAG benchmark has questions across five domains and eight question types, and provides a practical set-up to evaluate RAG systems. In particular, CRAG includes questions with answers that change from over seconds to over years; it considers entity popularity and covers not only head, but also torso and tail facts; it contains simple-fact questions as well as 7 types of complex questions such as comparison, aggregation and set questions to test the reasoning and synthesis capabilities of RAG solutions. Additionally, CRAG also provides mock APIs to query mock knowledge graphs so that developers can benchmark additional API calling capabilities for agents. Moreover, golden answers were provided in the dataset, which makes auto-evaluation with LLMs more robust. Therefore, CRAG benchmark is a realistic and comprehensive benchmark for agents.

## Getting started
1. Setup a work directory and download this repo into your work directory.
```
export $WORKDIR=<your-work-directory>
cd $WORKDIR
git clone https://github.com/opea-project/GenAIEval.git
```
2. Build docker image
```
cd $WORKDIR/GenAIEval/evals/evaluation/crag_eval/docker/
bash build_image.sh
```
3. Set environment vars for downloading models from Huggingface
```
mkdir $WORKDIR/hf_cache 
export HF_CACHE_DIR=$WORKDIR/hf_cache
export HF_HOME=$HF_CACHE_DIR
export HUGGINGFACEHUB_API_TOKEN=<your-hf-api-token>
```
4. Start docker container
This container will be used to preprocess dataset and run benchmark scripts.
```
bash launch_eval_container.sh
```

## CRAG dataset
1. Download original data and process it with commands below.
You need to create an account on the Meta CRAG challenge [website](https://www.aicrowd.com/challenges/meta-comprehensive-rag-benchmark-kdd-cup-2024). After login, go to this [link](https://www.aicrowd.com/challenges/meta-comprehensive-rag-benchmark-kdd-cup-2024/problems/meta-kdd-cup-24-crag-end-to-end-retrieval-augmented-generation/dataset_files) and download the `crag_task_3_dev_v4.tar.bz2` file. Then make a `datasets` directory in your work directory using the commands below.
```
cd $WORKDIR
mkdir datasets
```
Then put the `crag_task_3_dev_v4.tar.bz2` file in the `datasets` directory, and decompress it by running the command below.
```
cd $WORKDIR/datasets
tar -xf crag_task_3_dev_v4.tar.bz2
```
2. Preprocess the CRAG data
Data preprocessing directly relates to the quality of retrieval corpus and thus can have significant impact on the agent QnA system. Here, we provide one way of preprocessing the data where we simply extracts all the web search snippets as-is from the dataset per domain. We also extract all the query-answer pairs along with other meta data per domain. You can run the command below to use our method. The data processing will take some time to finish.
```
cd $WORKDIR/GenAIEval/evals/evaluation/crag_eval/preprocess_data
bash run_data_preprocess.sh
```
3. Optional - Sample queries for benchmark
The CRAG dataset has more than 4000 queries, and running all of them can be very expensive and time-consuming. You can sample a subset for benchmark. Here we provide a script to sample up to 5 queries per question_type per dynamism in each domain. For example, we were able to get 92 queries from the music domain using the script.
```
bash run_sample_data.sh
```
4. Use the small subset that we have processed for a quick run
```
Small data files in this repo
```
## Launch agent QnA system
Here we showcase a RAG agent in GenAIExample repo. Please refer to the README in the AgentQnA example for more details. You can build your own agent systems using OPEA components, then expose your own systems as an endpoint for this benchmark.
1. Build images
```
git clone
cd GenAIExamples/AgentQnA/tests/
bash 1_build_images.sh
```
2. Start retrieval tool
```
bash 2_start_retrieval_tool.sh
```
3. Ingest data into vector database and validate retrieval tool
```
# Placeholder - may change depending on data
bash 3_ingest_data_and_validate_retrieval.sh
```
3. Launch and validate agent endpoint
```
bash 4_launch_and_validate_agent.sh
```

## Run CRAG benchmark
Once you have your agent system up and running, you can follow the steps below to run the benchmark.
1. Generate answers with agent
Change the variables in the script below and run the script. By default, it will run a sampled set of queries in music domain.
```
cd $WORKDIR/GenAIEval/evals/evaluation/crag_eval/run_benchmark
bash run_generate_answer.sh
```
2. Use LLM-as-judge to grade the answers
First, in another terminal, launch llm endpoint with HF TGI
```
cd llm_judge
bash launch_llm_judge_endpoint.sh
```
Validate that the llm endpoint is working properly.
```
export host_ip=$(hostname -I | awk '{print $1}')
curl ${host_ip}:8085/generate_stream \
    -X POST \
    -d '{"inputs":"What is Deep Learning?","parameters":{"max_new_tokens":20}}' \
    -H 'Content-Type: application/json'
```
Second, back to the interactive crag-eval docker, run command below
```
bash run_grading.sh
```
