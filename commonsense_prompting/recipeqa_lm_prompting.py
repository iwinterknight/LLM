import openai
import json
import re
import time
import pandas as pd
import pickle

from data_preparation.recipe_commonsense_data_augmentation import fetch_data_df

# openai.api_key = "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"
openai.api_key = "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"


def prune_instruction(step_text, num_sentences=3):
    return step_text.split(".")[:num_sentences]


def construct_prompt(row, trim=False):
    prompt = "Generate 30 useful commonsense cooking questions and answers with reasoning, for recipe " \
             "text below. Answers should be descriptive and have reasoning. Create answers if answers are not in the recipe text.\n "
    prompt += '\n"""\n\n'

    title = "Title : " + row[1] + "\n"
    prompt += title
    prompt += "Recipe :\n"
    prompt += "1. Ingredients : " + ", ".join(row[2].split("\n")) + "\n"
    prompt += "Instructions :\n"
    i = 3
    while (row[i] != "stop" and pd.isna(row[i])):
        if trim:
            trimmed_instruction = prune_instruction(row[i])
            prompt += str(i - 1) + ". " + " ".join(trimmed_instruction) + "\n"
        else:
            prompt += str(i - 1) + ". " + row[i] + "\n"
        i += 1

    prompt += ' """ '
    return prompt


def generate_commonsense_qa():
    start_index = 20570
    skip_inex = None

    if start_index == 0:
        fn_json = "./data/recipe_qa_train_data.json"
        with open(fn_json, "r", encoding='utf-8') as infile:
            items = json.load(infile)
        data_df = fetch_data_df(items)
    else:
        data_df = pd.read_pickle("latest_saved_df.pickle")

    generated_qa = {}
    start = time.time()
    start_flag = False
    for index, row in data_df.iterrows():
        if index and index == skip_inex:
            continue

        if index >= start_index:
            if index > start_index and index % 10 == 0:
                print(f"Processed {index} queries.")
            if index > start_index and index % 50 == 0:
                end = time.time()
                print("Generated QA for {} recipes.\nTime elapsed : {} seconds".format(index, end - start))
                fn = f"recipe_commonsense_qa_{index - 50}_{index}.pickle"
                with open(fn, "wb") as f:
                    pickle.dump(generated_qa, f)

                data_df.to_pickle("latest_saved_df.pickle")
                start = time.time()

            if not start_flag:
                prompt = row['prompt']
                start_flag = True
            else:
                prompt = "Generate 30 useful commonsense cooking questions and long answers with detailed reasoning, for the recipe text below. Questions should not be from the text.\n"

            prompt = construct_prompt(row, trim=False)
            prompt_len = len(re.findall(r'\w+', prompt))
            prompt_buffer_len = 2 * prompt_len
            if prompt_buffer_len > 4097:
                prompt = construct_prompt(row, trim=True)
                prompt_len = len(re.findall(r'\w+', prompt))
                prompt_buffer_len = 2 * prompt_len
            generation_max_token = 4097 - prompt_buffer_len

            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.4,
                    max_tokens=generation_max_token,
                    top_p=1,
                    frequency_penalty=0.2,
                    presence_penalty=0.2,
                    stop="stop",
                    timeout=1500
                )
            except Exception as e:
                data_df.to_pickle("latest_saved_df.pickle")
                print("Exception {} at index {}".format(e, index))

                try:
                    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.4,
                        max_tokens=generation_max_token - 500,
                        top_p=1,
                        frequency_penalty=0.2,
                        presence_penalty=0.2,
                        stop="stop",
                        timeout=1500
                    )
                except Exception as e1:
                    data_df.to_pickle("latest_saved_df.pickle")
                    print("Repeat Exception {} at index {}".format(e1, index))
                    continue

            prompt_response = response["choices"][0]["message"]["content"]
            data_df.at[index, 'prompt_response'] = prompt_response
            generated_qa[index] = (data_df.iloc[index]["title"], prompt_response)
            data_df.to_pickle("latest_saved_df.pickle")

            if prompt_response is None or len(prompt_response) == 0:
                print("Response not generated!! : {}".format(index))

    return data_df


df = generate_commonsense_qa()
print(df)

# import pickle
# with open("recipe_commonsense_qa_3750_3800.pickle", "rb") as f:
#     sample_data = pickle.load(f)
#
# sample_data
