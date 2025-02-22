# Adapted from https://github.com/facebookresearch/CRAG/blob/main/local_evaluation.py

import bz2
import json
import os
import re
from datetime import datetime

from openai import APIConnectionError, OpenAI, RateLimitError
from prompts import IN_CONTEXT_EXAMPLES, INSTRUCTIONS
from tqdm.auto import tqdm



def load_json_file(file_path):
    """Load and return the content of a JSON file."""
    logger.info(f"Loading JSON from {file_path}")
    with open(file_path) as f:
        return json.load(f)


def get_system_message():
    """Returns the system message containing instructions and in context examples."""
    return INSTRUCTIONS + "\n" + IN_CONTEXT_EXAMPLES


def attempt_api_call(client, model_name, messages, max_retries=10):
    """Attempt an API call with retries upon encountering specific errors."""
    # todo: add default response when all efforts fail
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                # response_format={"type": "json_object"},
                temperature=0.0,
            )
            return response.choices[0].message.content
        except (APIConnectionError, RateLimitError):
            print(f"API call failed on attempt {attempt + 1}, retrying...")
        except Exception as e:
            print(f"Unexpected error: {e}")
            break
    return None


def log_response(messages, response, output_directory="api_responses"):
    """Save the response from the API to a file."""
    os.makedirs(output_directory, exist_ok=True)
    file_name = datetime.now().strftime("%d-%m-%Y-%H-%M-%S.json")
    file_path = os.path.join(output_directory, file_name)
    with open(file_path, "w") as f:
        json.dump({"messages": messages, "response": response}, f)


def parse_response(response: str):
    """
    Return a tuple of (explanation, score) from the response, 
    where score is 0 if the prediction is wrong, 1 if the prediction is correct.

    Need to handle
    Corner case 1:
        {"explanation": ...}
        Wait, no! I made a mistake. The prediction does not exactly match the ground truth. ...
        {...}

    Corner case 2:
        {"score": 0, "explanation": "The prediction does not contain item, nick "goose" bradshaw, that is in the ground truth."}
        return a tuple of (explanation, score)
    """
    matches = re.findall(r"{([^}]*)}", response)
    text = ""
    for match in matches:
        text = "{" + match + "}"
    try:
        score = -1
        # Pattern to match the score
        score_pattern = r'"score"\s*:\s*(\d+)'
        score_match = re.search(score_pattern, text)
        if score_match:
            score = int(score_match.group(1))
            if score != 0 and score != 1:
                raise Exception("bad score: " + response)
        else:
            return "Parse Err: Score not found", -1

        # Pattern to match the explanation
        explanation_pattern = r'"explanation"\s*:\s*"(.+)"'
        explanation_match = re.search(explanation_pattern, text)
        if explanation_match:
            explanation = explanation_match.group(1)
            return explanation, score
        else:
            return text, score
    except Exception as e:
        print(f"Parsing Error with resp: {response}")
        print(f"Error: {e}")
        return response, -1


# def trim_predictions_to_max_token_length(prediction):
#     """Trims prediction output to 75 tokens using Llama2 tokenizer"""
#     max_token_length = 75
#     tokenized_prediction = tokenizer.encode(prediction)
#     trimmed_tokenized_prediction = tokenized_prediction[1 : max_token_length + 1]
#     trimmed_prediction = tokenizer.decode(trimmed_tokenized_prediction)
#     return trimmed_prediction


def load_data_in_batches(dataset_path, batch_size):
    """
    Generator function that reads data from a compressed file and yields batches of data.
    Each batch is a dictionary containing lists of interaction_ids, queries, search results, query times, and answers.
    
    Args:
    dataset_path (str): Path to the dataset file.
    batch_size (int): Number of data items in each batch.
    
    Yields:
    dict: A batch of data.
    """
    def initialize_batch():
        """ Helper function to create an empty batch. """
        return {"interaction_id": [], "query": [], "search_results": [], "query_time": [], "answer": []}

    try:
        with bz2.open(dataset_path, "rt") as file:
            batch = initialize_batch()
            for line in file:
                try:
                    item = json.loads(line)
                    for key in batch:
                        batch[key].append(item[key])
                    
                    if len(batch["query"]) == batch_size:
                        yield batch
                        batch = initialize_batch()
                except json.JSONDecodeError:
                    logger.warn("Warning: Failed to decode a line.")
            # Yield any remaining data as the last batch
            if batch["query"]:
                yield batch
    except FileNotFoundError as e:
        logger.error(f"Error: The file {dataset_path} was not found.")
        raise e
    except IOError as e:
        logger.error(f"Error: An error occurred while reading the file {dataset_path}.")
        raise e



def generate_predictions(dataset_path, participant_model):
    """
    Processes batches of data from a dataset to generate predictions using a model.
    
    Args:
    dataset_path (str): Path to the dataset.
    participant_model (object): UserModel that provides `get_batch_size()` and `batch_generate_answer()` interfaces.
    
    Returns:
    tuple: A tuple containing lists of queries, ground truths, and predictions.
    """
    queries, ground_truths, predictions = [], [], []
    batch_size = participant_model.get_batch_size()

    for batch in tqdm(load_data_in_batches(dataset_path, batch_size), desc="Generating predictions"):
        batch_ground_truths = batch.pop("answer")  # Remove answers from batch and store them
        batch_predictions = participant_model.batch_generate_answer(batch)
        
        queries.extend(batch["query"])
        ground_truths.extend(batch_ground_truths)
        predictions.extend(batch_predictions)
    
    return queries, ground_truths, predictions


def evaluate_predictions(queries, ground_truths_list, predictions, args):
    """
    Evaluates the predictions generated by a model against ground truth answers.
    
    Args:
    queries (List[str]): List of queries.
    ground_truths_list (List[str]): List of ground truth answers. 
    predictions (list): List of predictions generated by the model.
    
    Returns:
    dict: A dictionary containing evaluation results.
    """

    # vllm client
    client = OpenAI(
        base_url=f"{args.llm_endpoint_url}/v1",
        api_key="token-abc123",
    )
    
    n_miss, n_correct = 0, 0
    system_message = get_system_message()

    eval_response = []
    score_list = []

    for _idx, prediction in enumerate(tqdm(
        predictions, total=len(predictions), desc="Evaluating Predictions"
    )):
        query = str(queries[_idx])
        ground_truth = str(ground_truths_list[_idx]) #.strip()
        # trim prediction to 75 tokens using Llama2 tokenizer
        # prediction = trim_predictions_to_max_token_length(prediction)
        prediction = str(prediction) #.strip()

        ground_truth_lowercase = ground_truth.lower()
        prediction_lowercase = prediction.lower()

        if "i don't know" in prediction_lowercase:
            n_miss += 1
            accuracy = 0
            eval_response.append("I don't know in response")
            score_list.append(accuracy)
            continue

        accuracy = -1

        messages = [
            {"role": "system", "content": system_message},
            {
                "role": "user",
                "content": f"Question: {query}\n Ground truth: {ground_truth}\n Prediction: {prediction}\n",
            },
        ]
        if prediction_lowercase == ground_truth_lowercase:
            # exact correct
            print("Exact match!")
            accuracy = 1
            eval_response.append("exact match")
            # break
        # elif "invalid" in prediction_lowercase and "invalid" in ground_truth_lowercase:
        #     accuracy = 1
        #     break
        # elif "invalid" in prediction_lowercase and "invalid" not in ground_truth_lowercase:
        #     # hallucination
        #     accuracy = 0
        #     continue
        # elif "invalid" not in prediction_lowercase and "invalid" in ground_truth_lowercase:
        #     # hallucination
        #     accuracy = 0
        #     continue
        else:
            # need to use the OpenAI evaluation model to get the accuracy result (0 means wrong, 1 means correct)
            print("Using LLM as judge...")
            response = attempt_api_call(client, args.model, messages)
            print(f"LLM judge Response: {response}")
            eval_response.append(response)
            if response:
                # log_response(messages, response)
                _, accuracy = parse_response(response)
                print(f"Parsed Accuracy: {accuracy}")
                # if accuracy == 1:
                #     # no need to check other ground truth(s)
                #     break
        print(f"**Score: {accuracy}\n**Reason: {response}")
        score_list.append(accuracy)

        if accuracy == 1:
            n_correct += 1
        elif accuracy == 0:
            n_miss += 1

    n = len(predictions)
    results = {
        "score": (2 * n_correct + n_miss) / n - 1,
        "accuracy": n_correct / n,
        "hallucination": (n - n_correct - n_miss) / n,
        "missing": n_miss / n,
        "n_miss": n_miss,
        "n_correct": n_correct,
        "n_hallucination": n - n_correct - n_miss,
        "total": n,
    }
    # logger.info(results)
    return results, eval_response, score_list
    


if __name__ == "__main__":
    import argparse
    import os
    import pandas as pd

    parser = argparse.ArgumentParser()
    parser.add_argument("--llm_endpoint_url", type=str, default="http://localhost:8085")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.3-70B-Instruct")
    parser.add_argument("--filename", type=str, default="react_agent_baseline_all_t0p5.csv", help="Path to the dataset file.")

    args = parser.parse_args()

    WORKDIR=os.getenv('WORKDIR')
    DATASET_PATH = os.path.join(WORKDIR, 'datasets/financebench/results/')

    # Load the dataset
    dataset_path = os.path.join(DATASET_PATH, args.filename)
    df = pd.read_csv(dataset_path)
    # df = df.head(2)

    queries = df["question"].tolist()
    ground_truths = df["answer"].tolist()
    predictions = df["agent_response"].tolist()
    
    # Evaluate Predictions
    evaluation_results, eval_response, score_list = evaluate_predictions(
        queries, ground_truths, predictions, args,
    )

    print(f"Eval Results:\n{evaluation_results}")
    logfile = args.filename.replace(".csv", "_eval_results.json")
    with open(os.path.join(DATASET_PATH, logfile), "w") as f:
        json.dump(evaluation_results, f)

    df["score"] = score_list
    df["eval_response"] = eval_response

    output_filename = args.filename.replace(".csv", "_graded.csv")
    df.to_csv(os.path.join(DATASET_PATH, output_filename), index=False)

    