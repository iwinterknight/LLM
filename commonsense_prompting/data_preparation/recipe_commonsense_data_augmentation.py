import os
import json
import sys
import pandas as pd
from collections import OrderedDict

from .recipe_prompt_instructions import *


def read_data(dataset, writefile):
    f = open(dataset)
    data = json.loads(f.read())["data"]
    f.close()
    items = []
    for d in data:
        recipe = OrderedDict()
        title = d["recipe_id"].replace("-", " ")
        recipe["title"] = title
        context = d["context"]
        steps = OrderedDict()
        for i, step in enumerate(context):
            if i == 0:
                steps["Ingredients"] = step["body"]
            else:
                steps[str(i)] = step["body"]
        recipe["steps"] = steps
        items.append(recipe)
    json_obj = json.dumps(items, indent=4)
    with open(writefile, "w", encoding='utf-8') as outfile:
        outfile.write(json_obj)

    return items


# def write_data(items, writefile):
#     f = open(writefile, "w", encoding='utf-8')
#     for item in items:
#         f.write("\n\n\"\"\"\n")
#         title = item["title"]
#         f.write("Title : {}\nRecipe :\n".format(title))
#         steps = item["steps"]
#         for i, step in enumerate(steps):
#             if i == 0:
#                 f.write("{}. Ingredients : {}\n".format(i+1, step))
#             else:
#                 f.write("{}. {}\n".format(i+1, step))
#         f.write("\n\"\"\"\n\n")
#     f.close()


def write_data(items, writefile):
    f = open(writefile, "w", encoding='utf-8')
    for item in items:
        title = item["title"]
        steps = item["steps"]
        for i, step in enumerate(steps):
            if i == 0:
                f.write("{}. Ingredients : {}\n".format(i+1, step))
            else:
                f.write("{}. {}\n".format(i+1, step))
        f.write("\n\"\"\"\n\n")
    f.close()


def fetch_data_df(items):
    data_df = pd.json_normalize(items)
    data_df = data_df.fillna("stop")

    data_df["prompt"] = prompt_4["instruction"] + "\n\nExpected Output :" + prompt_4["expected_output"]

    cols = data_df.columns.tolist()
    cols = [cols[-1]] + cols[:-1]
    data_df = data_df[cols]
    print(data_df.head())

    return data_df


# # items = read_data(sys.argv[1])
# datapath = "../data/recipe_qa_train.json"
# fn_json = "../data/recipe_qa_train_data.json"
# fn_pd = "../data/recipe_qa_train_data.pickle"
#
# # items = read_data(datapath, fn_json)
# # print(items[0])
# # print(len(items))
#
# fetch_data_df(fn_json)
