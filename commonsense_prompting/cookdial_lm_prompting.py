import os.path

import openai
import json
import re
import time
import pandas as pd
import pickle
from os import listdir
from os.path import isfile, join

from data_preparation.recipe_prompt_instructions import *

# openai.api_key = "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"
openai.api_key = "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"


def read_recipes():
    recipe_dir = "data/cookdial_recipes"
    recipe_files = [f for f in listdir(recipe_dir) if isfile(join(recipe_dir, f))]

    recipes = []
    for i, file in enumerate(recipe_files):
        f = open(os.path.join(recipe_dir, file))
        data = json.load(f)
        f.close()

        first_instruction = True
        recipe_title = data["title"]
        recipe = "Recipe : " + recipe_title + "\n" + "Ingredients :\n"
        recipe_steps = data["content"]
        for step in recipe_steps:
            if step["type"] == "ingredient":
                recipe += "- " + step["text"] + "\n"
            elif step["type"] == "instruction":
                if first_instruction:
                    recipe += "Instructions :\n"
                    first_instruction = False
                step_text = step["text"]
                step_num = re.findall(r"(\d+)\)", step_text)[0]
                step_span = re.match(r"(\d+)\)", step_text)
                step_start = step_span.span()[1]
                filtered_step_text = step_text[step_start:].strip()
                recipe += step_num + ". " + filtered_step_text + "\n"
        recipes.append((recipe_title, recipe))

    with open('data/cookdial_recipes.pickle', 'wb') as handle:
        pickle.dump(recipes, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return recipes


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


def create_input_dict(recipes):
    input_recipes_list = []
    for recipe_title, recipe in recipes:
        data_dict = {}
        # data_dict["instruction"] = prompt_5["instruction"] + "\n\nExpected Output :" + prompt_5["expected_output"]
        data_dict["recipe_title"] = recipe_title
        data_dict["instruction"] = \
            "Generate 30 useful commonsense cooking instruction based questions and answers with detailed reasoning for the recipe given in triple backticks.\n" + "```" + recipe + "```"
        data_dict["recipe"] = recipe
        input_recipes_list.append(data_dict)
    return input_recipes_list


def generate_commonsense_qa():
    recipes = read_recipes()
    start_index = 0
    skip_inex = None

    if start_index == 0:
        input_recipes_list = create_input_dict(recipes)
    else:
        with open('latest_saved_data_dict.pickle', 'rb') as handle:
            input_recipes_list = pickle.load(handle)

    generated_qa = {}
    start = time.time()
    start_flag = False
    for index, data_dict in enumerate(input_recipes_list):
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

                with open('latest_saved_data_dict.pickle', 'wb') as handle:
                    pickle.dump(generated_qa, handle, protocol=pickle.HIGHEST_PROTOCOL)
                start = time.time()

            try:
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": data_dict["instruction"]}],
                    temperature=0.4,
                    max_tokens=7500,
                    top_p=1,
                    frequency_penalty=0.2,
                    presence_penalty=0.2,
                    timeout=6000
                )
            except Exception as e:
                with open('latest_saved_data_dict.pickle', 'rb') as handle:
                    input_recipes_list = pickle.load(handle)
                print("Exception {} at index {}".format(e, index))

                try:
                    response = openai.ChatCompletion.create(
                        model="gpt-4",
                        messages=[{"role": "user", "content": data_dict["instruction"]}],
                        temperature=0.4,
                        max_tokens=7000,
                        top_p=1,
                        frequency_penalty=0.2,
                        presence_penalty=0.2,
                        timeout=4000
                    )
                except Exception as e1:
                    print("Latest Index : {}".format(index))
                    with open('latest_saved_data_dict.pickle', 'wb') as handle:
                        pickle.dump(generated_qa, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    print("Repeat Exception {} at index {}".format(e1, index))
                    continue

            prompt_response = response["choices"][0]["message"]["content"]
            data_dict['prompt_response'] = prompt_response
            generated_qa[index] = (data_dict["recipe_title"], prompt_response)
            with open('latest_saved_data_dict.pickle', 'wb') as handle:
                pickle.dump(generated_qa, handle, protocol=pickle.HIGHEST_PROTOCOL)

            if prompt_response is None or len(prompt_response) == 0:
                print("Response not generated!! : {}".format(index))

    return generated_qa



generated_qa = generate_commonsense_qa()
print(generated_qa)

# import pickle
# with open("recipe_commonsense_qa_3750_3800.pickle", "rb") as f:
#     sample_data = pickle.load(f)
#
# sample_data
