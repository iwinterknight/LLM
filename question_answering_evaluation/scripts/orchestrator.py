import requests
import pandas as pd
from rouge import Rouge
import faiss

from collections import Counter

rouge = Rouge()

from flask import Flask, render_template, request, url_for, jsonify

app = Flask(__name__)

from sentence_transformers import SentenceTransformer, util

similarity_model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')

SKIP_FLAG = False
ROUGE_SKIP_FLAG = True


def get_question_type(context, question, additional_data=None):
    url = "http://192.168.0.6:2081/"
    r = requests.post(url, json={'current_step_details': [context], 'text': [question]})
    response = r.json()
    return response["question_types"], response["performance"][0]


def get_faq_response(question, additional_data=None):
    url = "http://192.168.0.6:2089/"
    r = requests.post(url, json={'text': [question]})
    response = r.json()
    return response["response"], response["performance"][0]


def get_mrc_response(context, question, additional_data=None):
    url = "http://192.168.0.6:2090/"
    if additional_data and "is_wikihow" in additional_data:
        r = requests.post(url, json={'current_step_details': [context], 'text': [question],
                                     'is_wikihow': [additional_data["is_wikihow"]]})
    else:
        r = requests.post(url, json={'current_step_details': [context], 'text': [question]})
    response = r.json()
    return response["response"], response["performance"][0]


def fetch_legacy_response(question, context, additional_data):
    question_types, question_type_score = get_question_type(context, question)
    question_type = question_types[0]

    # if question_type == "EVI":
    #     question_type = question_types[1]

    response = None
    score = None
    response_type = None
    if question_type == "FAQ":
        response, score = get_faq_response(question)
        response_type = "FAQ"
    elif question_type == "MRC":
        response, score = get_mrc_response(context, question, additional_data)
        response_type = "MRC"
    else:
        response_type = "EVI"

    return response_type, response, score


def retrieve_legacy_responses(df):
    responses = {}
    contexts, questions, answers, legacy_answers, response_types, generated_answers = [], [], [], [], [], []
    for index, row in df.iterrows():
        if index > 0 and index % 100 == 0:
            print(f"{index} retrieved of {len(df)}")

        context, question, answer, generated_answer = row["passage"], row["question"], row["answer"], row[
            "generated_answer"]
        additional_data = None
        response_type, response, score = fetch_legacy_response(question, context, additional_data)

        if SKIP_FLAG:
            if response_type == "FAQ":
                contexts.append(context)
                questions.append(question)
                answers.append(answer)
                legacy_answers.append(response)
                response_types.append(response_type)
                generated_answers.append(generated_answer)
        else:
            contexts.append(context)
            questions.append(question)
            answers.append(answer)
            legacy_answers.append(response)
            response_types.append(response_type)
            generated_answers.append(generated_answer)

    assert (len(contexts) == len(questions) == len(answers) == len(legacy_answers) == len(
        response_types) == len(generated_answers))

    responses["contexts"] = contexts
    responses["questions"] = questions
    responses["answers"] = answers
    responses["legacy_answers"] = legacy_answers
    responses["response_types"] = response_types
    responses["generated_answers"] = generated_answers

    print("{}".format(Counter(responses["response_types"])))

    return responses


def topk_measure(responses, k=5):
    answers, __legacy_answers, generated_answers, response_types = responses["answers"], responses["legacy_answers"], responses[
        "generated_answers"], responses["response_types"]

    correct_legacy, correct_generated = 0, 0
    total = 0
    answer_encodings = similarity_model.encode(answers)
    index = faiss.IndexFlatL2(answer_encodings.shape[1])
    index.add(answer_encodings)

    legacy_answers = []
    for legacy_answer in __legacy_answers:
        if not legacy_answer:
            legacy_answers.append("")
        else:
            legacy_answers.append(legacy_answer)

    legacy_encodings = similarity_model.encode(legacy_answers)
    generated_encodings = similarity_model.encode(generated_answers)

    for i, (legacy_encoding, generated_encoding, response_type) in enumerate(zip(legacy_encodings, generated_encodings, response_types)):
        if response_type == "EVI" and SKIP_FLAG:
            continue

        D, I_legacy = index.search(legacy_encoding.reshape(1, legacy_encoding.shape[0]), k)
        D, I_generated = index.search(generated_encoding.reshape(1, generated_encoding.shape[0]), k)

        if i in I_legacy[0]:
            correct_legacy += 1
        if i in I_generated[0]:
            correct_generated += 1

        total += 1

    topk_acc_legacy = correct_legacy / total
    topk_acc_generated = correct_generated / total

    print(f"top_{k}_acc for legacy QA : {topk_acc_legacy}")
    print(f"top_{k}_acc for generated QA : {topk_acc_generated}")

    return topk_acc_legacy, topk_acc_generated


def sentence_similarity_measure(responses):
    answers, legacy_answers, generated_answers, response_types = responses["answers"], responses["legacy_answers"], responses[
        "generated_answers"], responses["response_types"]

    legacy_sim_cum, generated_sim_cum = 0.0, 0.0
    total = 0
    answer_encodings = similarity_model.encode(answers)
    index = faiss.IndexFlatL2(answer_encodings.shape[1])
    index.add(answer_encodings)

    for i, (answer, legacy_answer, generated_answer, response_type) in enumerate(zip(answers, legacy_answers, generated_answers, response_types)):
        if response_type == "EVI" and SKIP_FLAG:
            continue

        if not legacy_answer:
            legacy_answer = ""

        answer_encoding = similarity_model.encode(answer)
        legacy_encoding = similarity_model.encode(legacy_answer)
        generated_encoding = similarity_model.encode(generated_answer)

        legacy_sim = util.dot_score(answer_encoding, legacy_encoding)
        generated_sim = util.dot_score(answer_encoding, generated_encoding)

        legacy_sim_cum += legacy_sim[0][0].numpy()
        generated_sim_cum += generated_sim[0][0].numpy()
        total += 1

    legacy_sim_cum /= total
    generated_sim_cum /= total

    print(f"Sentence similarity scores for legacy QA : {legacy_sim_cum}")
    print(f"Sentence similarity scores for generated QA : {generated_sim_cum}")

    return legacy_sim_cum, generated_sim_cum


def rouge_measure(responses):
    answers, __legacy_answers, generated_answers, response_types = responses["answers"], responses["legacy_answers"], responses[
        "generated_answers"], responses["response_types"]

    legacy_rouge_scores, generated_rouge_scores = 0.0, 0.0
    if ROUGE_SKIP_FLAG:
        check_answers, check_legacy_answers, check_generated_answers = [], [], []
        for i, (answer, legacy_answer, generated_answer, response_type) in enumerate(
                zip(answers, __legacy_answers, generated_answers, response_types)):

            check_answers.append(answer)
            check_generated_answers.append(generated_answer)

            if response_type != "EVI" and len(legacy_answer) > 0:
                check_legacy_answers.append(legacy_answer)


        generated_rouge_scores = rouge.get_scores(check_answers, check_generated_answers, avg=True)
    else:
        legacy_answers = []
        for legacy_answer in __legacy_answers:
            if not legacy_answer:
                legacy_answers.append("")
            else:
                legacy_answers.append(legacy_answer)

        legacy_rouge_scores = rouge.get_scores(answers, legacy_answers, avg=True)
        generated_rouge_scores = rouge.get_scores(answers, generated_answers, avg=True)

    print(f"Rogue scores for legacy QA : {legacy_rouge_scores}")
    print(f"Rogue scores for generated QA : {generated_rouge_scores}")

    return legacy_rouge_scores, generated_rouge_scores


def main():
    df = pd.read_pickle("../data/validation_set_generated_responses_xl_df.pickle")

    responses = retrieve_legacy_responses(df)
    topk_measure(responses)
    sentence_similarity_measure(responses)
    rouge_measure(responses)

    # context = [
    #     "stir the contents of the container thoroughly. keep stirring until the texture of the oats is consistent from top to bottom. otherwise, youll end up with unappetizing dry patches. you can also add other dry ingredients at this stage, such as chia seeds, flax, and ground spices."]
    # question = ["How much should I stir?"]
    #
    # get_question_type(context, question)


if __name__ == "__main__":
    main()
