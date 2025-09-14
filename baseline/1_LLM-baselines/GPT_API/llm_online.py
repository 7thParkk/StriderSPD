import json
import os
import jsonlines
import openai
import argparse
import tiktoken
from transformers import AutoTokenizer
import concurrent.futures
from tqdm import tqdm


DEFAULT_MAX_INPUT_TOKEN = 16384
CONCURRENT_REQUESTS = 16


def prompt_limit_token(prompt, max_token, model):
    if 'gpt' in model.lower():
        encoder = tiktoken.encoding_for_model(model)
        token = encoder.encode(prompt)
        token = token[:max_token]
        prompt = encoder.decode(token)
    elif 'deepseek' in model.lower():
        tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1")
        token = tokenizer.encode(prompt, max_length=max_token, truncation=True, add_special_tokens=False)
        prompt = tokenizer.decode(token, skip_special_tokens=True)
    
    return prompt
    


def main(args):
    max_input_token = int(args.token) - 100
    
    with open(f"./test_data/{args.dataset}.json", "r") as f:
        test_set = json.load(f)
    for i, data in enumerate(test_set):
        data["num"] = i
        
    generated_predictions = []
    
    os.makedirs(f"./saves/{args.dataset}/{args.model}", exist_ok=True)
    pred_file = f"./saves/{args.dataset}/{args.model}/generated_predictions.jsonl"
    progress_file = f"./saves/{args.dataset}/{args.model}/generated_predictions.jsonl.progress"
    
    if os.path.exists(progress_file):
        with jsonlines.open(progress_file, "r") as reader:
            generated_predictions = list(reader)
            predicted_list = [p['num'] for p in generated_predictions]
            test_set = [data for data in test_set if data['num'] not in predicted_list]

    
    def single_predict(data):
        client = openai.Client(api_key=args.key, base_url=args.url)
        system_prompt = data["system"]
        user_prompt = data["instruction"]
        
        user_prompt = prompt_limit_token(user_prompt, max_input_token, args.model)
        
        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ]
        
        try:
            response = client.chat.completions.create(
                model=args.model,
                messages=messages,
            )
        except Exception as e:
            print(e)
            return None
        
        predict = response.choices[0].message.content
    
        result = {
            "prompt": messages,
            "predict": predict,
            "label": data["output"],
            "num": data["num"]
        }
        
        return result
    
    # for data in tqdm(test_set, total=len(test_set)):
    #     result = single_predict(data)
    #     generated_predictions.append(result)
    
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=CONCURRENT_REQUESTS) as executor, jsonlines.open(pred_file + ".progress", "a") as progress_file:
            futures = {executor.submit(single_predict, data): data for data in test_set}
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                result = future.result()
                if result is not None:
                    generated_predictions.append(result)
                    progress_file.write(result)
    except KeyboardInterrupt:
        print("KeyboardInterrupt")
    
    print("Stopped! Saving generated predictions, do not interrupt!")
    generated_predictions = sorted(generated_predictions, key=lambda x: x["num"])
    with open(pred_file, "w", encoding="utf-8", errors="ignore") as f:
        for pred in generated_predictions:
            f.write(json.dumps(pred) + "\n")
    print("Saved!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-u", "--url", help="OpenAI API URL")
    parser.add_argument("-k", "--key", help="OpenAI API Key")
    parser.add_argument("-m", "--model", help="OpenAI Model")
    parser.add_argument("-d", "--dataset", help="Dataset")
    parser.add_argument("-t", "--token", help="Max input token", default=DEFAULT_MAX_INPUT_TOKEN)
    args = parser.parse_args()
    main(args)
    