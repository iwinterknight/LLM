import os
import pickle
import json
import openai
import numpy as np
import torch
import random
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from scripts.llm_prompts import REASONING_CLASSIFICATION_PROMPT

entailment_tokenizer = AutoTokenizer.from_pretrained("chromeNLP/textattack_bert_base_MNLI_fixed")
entailment_model = AutoModelForSequenceClassification.from_pretrained("chromeNLP/textattack_bert_base_MNLI_fixed")

similarity_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

openai.api_key = "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"
# openai.api_key = "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"


def get_llm_response(sent1, sent2):
    prompt = """
    For the 2 utterances given below, output :
    1. Entailment Score - Value ranges between 0 and 1. Entailment score is 0 when utterances contradict each other. Entailment score is 1 when either one of the utterances entails the other utterance completely. Entailment score is 0.5 if the utterances are unrelated. Score should be graded depending on the degree of entailment between the 2 utterances. Do not give a high score unless the entailment is obvious and complete.
    2. Similarity Score - Value ranges between 0 and 1. Similarity score is 0 when utterances are unrelated. Similarity score is 1 when the utterances are semantically very similar.  Score should be graded depending on the degree of similarity between the 2 utterances. Do not give a high score unless the similarity is obvious and covers the text discussed in the utterances.
    Response should be a json string in the following format :\n{\"entailment_score\": NUMBER, \"similarity_score\": NUMBER}\n
    """

    prompt_input = f"\nutterance 1 : {sent1}\nutterance 2 : {sent2}"
    prompt += prompt_input
    try:
        response = openai.ChatCompletion.create(
            # model="gpt-3.5-turbo",
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=500,
            top_p=1,
            frequency_penalty=0.2,
            presence_penalty=0.2,
            timeout=5
            # stop="stop"
        )
    except Exception as e:
        print("Exception {}".format(e))
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            # model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=500,
            top_p=1,
            frequency_penalty=0.2,
            presence_penalty=0.2,
            timeout=5
            # stop="stop"
        )
    response_json = response["choices"][0]["message"]["content"]
    response_dict = json.loads(response_json)
    if 'entailment_score' not in response_dict:
        response_dict['entailment_score'] = 0.0
    if 'similarity_score' not in response_dict:
        response_dict['similarity_score'] = 0.0
    return response_dict['entailment_score'], response_dict['similarity_score']


def cosine_similarity(embed1, embed2):
    return np.dot(embed1, embed2) / (np.linalg.norm(embed1) * np.linalg.norm(embed2))


def compute_entailment_score(sent1, sent2):
    encoding = entailment_tokenizer(sent1, sent2, return_tensors="pt")

    outputs = entailment_model(**encoding, labels=torch.LongTensor([1]))
    logits = outputs.logits

    entailment_score = logits.tolist()[0][0]
    neutral_score = logits.tolist()[0][1]
    contradiction_score = logits.tolist()[0][2]
    return entailment_score, neutral_score, contradiction_score


def compute_similarity_score(sent1, sent2):
    sent1_embeds = similarity_model.encode(sent1)
    sent2_embeds = similarity_model.encode(sent2)
    return cosine_similarity(sent1_embeds, sent2_embeds)


def compute_entailment_similarity_scores(s1, s2):
    similarity_score = compute_similarity_score(s1, s2)
    entailment_score, neutral_score, contradiction_score = compute_entailment_score(s1, s2)
    llm_entailment_score, llm_similarity_score = get_llm_response(s1, s2)

    return {"entailment_score": entailment_score, "similarity_score": similarity_score.tolist(),
            "llm_entailment_score": llm_entailment_score, "llm_similarity_score": llm_similarity_score}


def remove_out_of_context_qa(response):
    anchor = "The recipe does not mention"
    similarity_score = compute_similarity_score(anchor, response)
    entailment_score, neutral_score, contradiction_score = compute_entailment_score(anchor, response)
    if similarity_score > 0.5 or entailment_score > 0.5:
        return False
    return True


def entailment_similarity_analysis(mrc):
    mrc_scores = {}
    faulty_indices = list()
    for k, v in mrc.items():

        # if k < 45:
        #     continue

        scores = {}
        if k > 0 and k % 10 == 0:
            print(f"{k} QA generated.")
        context, question, answer, extracted_answers, flan_answer, alpaca_answer = v["context"], v["question"], \
                                                                                   v["answer"], v["extracted_answers"], \
                                                                                   v["generated_answer"], v[
                                                                                       "alpaca_answer"]

        try:
            flan_scores = compute_entailment_similarity_scores(answer, flan_answer)
            alpaca_scores = compute_entailment_similarity_scores(answer, alpaca_answer)
            extractive_qa_scores = {}
            max_entailment_score, max_similarity_score, max_llm_entailment_score, max_llm_similarity_score = -float('inf'), -float('inf'), -float('inf'), -float('inf')
            for extractive_model in extracted_answers:
                for ans in extractive_model:
                    if len(ans) < 3:
                        continue
                    extractive_qa_score = compute_entailment_similarity_scores(answer, ans)
                    max_entailment_score = max(extractive_qa_score["entailment_score"],
                                               max_entailment_score)
                    max_similarity_score = max(extractive_qa_score["similarity_score"],
                                               max_similarity_score)
                    max_llm_entailment_score = max(extractive_qa_score["llm_entailment_score"],
                                                   max_llm_entailment_score)
                    max_llm_similarity_score = max(extractive_qa_score["llm_similarity_score"],
                                                   max_llm_similarity_score)
                extractive_qa_scores["entailment_score"] = max_entailment_score
                extractive_qa_scores["similarity_score"] = max_similarity_score
                extractive_qa_scores["llm_entailment_score"] = max_llm_entailment_score
                extractive_qa_scores["llm_similarity_score"] = max_llm_similarity_score
        except Exception as e:
            print(f"Exception {e} at QA index {k}")
            faulty_indices.append(k)

        scores["flan_scores"] = flan_scores
        scores["alpaca_scores"] = alpaca_scores
        scores["extractive_qa_scores"] = extractive_qa_scores
        mrc_scores[k] = scores

    with open(os.path.join("data", "output", 'mrc_scores.pickle'), 'wb') as handle:
        pickle.dump(mrc_scores, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return mrc_scores


def count_mrc():
    with open(os.path.join("data", "universal_mrc.pickle"), 'rb') as handle:
        mrc = pickle.load(handle)

    entailment_similarity_analysis(mrc)


def apply_cutoffs(context, question, answer, scores, cutoffs, pruned_qa_dict, pruned_qa_stats):
    for k, score in scores.items():
        cutoff = cutoffs[k]
        if score > cutoff:
            pruned_qa_stats[k] += 1
            pruned_qa_dict[k] = {"context": context, "question": question, "answer": answer}


def analyze_mrc(mrc_scores=None):
    c = 0.8
    flan_cutoffs = {"entailment_score": c, "similarity_score": c, "llm_entailment_score": c,
                          "llm_similarity_score": c}
    alpaca_cutoffs = {"entailment_score": c, "similarity_score": c, "llm_entailment_score": c,
                          "llm_similarity_score": c}
    extractiveqa_cutoffs = {"entailment_score": c, "similarity_score": c, "llm_entailment_score": c,
                          "llm_similarity_score": c}

    pruned_qa = {}

    with open(os.path.join("data", "universal_mrc.pickle"), 'rb') as handle:
        mrc = pickle.load(handle)

    if not mrc_scores:
        with open(os.path.join("data", "output", "mrc_scores.pickle"), 'rb') as handle:
            mrc_scores = pickle.load(handle)

    for k, v in mrc_scores.items():
        flan_scores, alpaca_scores, extractiveqa_scores = v["flan_scores"], v["alpaca_scores"], v["extractive_qa_scores"]
        mrc_item = mrc[k]
        context, question, answer, extracted_answers, flan_answer, alpaca_answer = mrc_item["context"], mrc_item["question"], \
                                                                                   mrc_item["answer"], mrc_item["extracted_answers"], \
                                                                                   mrc_item["generated_answer"], \
                                                                                   mrc_item["alpaca_answer"]

        include = remove_out_of_context_qa(answer)
        if include:
            try:
                score_list = ["entailment_score", "similarity_score", "llm_entailment_score", "llm_similarity_score"]
                include_ctr = 0
                for (score, cutoff) in [(extractiveqa_scores, extractiveqa_cutoffs)]:
                    for s in score_list:
                        if score[s] > cutoff[s]:
                            include_ctr += 1
                if include_ctr == len(score_list):
                    include = False

                include_ctr = 0
                for (score, cutoff) in [(flan_scores, flan_cutoffs)]:
                    for s in score_list:
                        if score[s] > cutoff[s]:
                            include_ctr += 1
                if include_ctr == len(score_list):
                    include = False

                include_ctr = 0
                for (score, cutoff) in [(alpaca_scores, alpaca_cutoffs)]:
                    for s in score_list:
                        if score[s] > cutoff[s]:
                            include_ctr += 1
                if include_ctr == len(score_list):
                    include = False
            except Exception as e:
                print(f"Exception {e} at item {k}")

        if include:
            pruned_qa[k] = {"context": context, "question": question, "answer": answer, "flan_answer": flan_answer,
                            "alpaca_answer": alpaca_answer, "extracted_answers": extracted_answers}

    with open(os.path.join("data", "output", 'pruned_qa.pickle'), 'wb') as handle:
        pickle.dump(pruned_qa, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return pruned_qa


def extract_reasoning_token_based(mrc_scores=None):
    with open(os.path.join("data", "universal_mrc.pickle"), 'rb') as handle:
        mrc = pickle.load(handle)

    reasoning_tokens = {"how": 0, "why": 0, "what": 0, "substitute": 0}
    for k, mrc_item in mrc.items():
        context, question, answer, extracted_answers, flan_answer, alpaca_answer = mrc_item["context"], mrc_item["question"], \
                                                                                   mrc_item["answer"], mrc_item["extracted_answers"], \
                                                                                   mrc_item["generated_answer"], \
                                                                                   mrc_item["alpaca_answer"]

        question_tokens = question.split(" ")
        for token_index, token in enumerate(question_tokens):
            if token_index < 3:
                if token.lower() == "why":
                    reasoning_tokens["why"] += 1
                    break
                elif token.lower() == "how":
                    reasoning_tokens["how"] += 1
                    break
                elif token.lower() == "what":
                    reasoning_tokens["what"] += 1
                    break

        if "instead of" in question or "substitute" in question:
            reasoning_tokens["substitute"] += 1

    return reasoning_tokens


def categorize_reasoning(pruned_qa, num_samples=50):
    prompt = REASONING_CLASSIFICATION_PROMPT

    sample_list = [True] * num_samples + [False] * (len(pruned_qa) - num_samples)
    random.shuffle(sample_list)

    assert len(sample_list) == len(pruned_qa)

    qa_reasoning_types = {}
    for ctr, (k, v) in enumerate(pruned_qa.items()):
        if not sample_list[ctr]:
            continue

        if ctr > 0 and ctr % 10 == 0:
            print(f"{ctr} items processed.")

            if ctr % 100 == 0:
                with open(os.path.join("data", "output", f"qa_reasoning_types_{ctr-100}_{ctr}.pickle"), 'wb') as handle:
                    pickle.dump(qa_reasoning_types, handle, protocol=pickle.HIGHEST_PROTOCOL)

        reasoning_type = {}
        question = v["question"]
        answer = v["answer"]
        alternate_answer = v["alpaca_answer"]
        prompt_input = prompt + f"\nQuestion : {question}\nAnswer : {answer}\nAlternate Answer : {alternate_answer}"

        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                # model="gpt-4",
                messages=[{"role": "user", "content": prompt_input}],
                temperature=0.2,
                max_tokens=500,
                top_p=1,
                frequency_penalty=0.2,
                presence_penalty=0.2,
                timeout=5
                # stop="stop"
            )
        except Exception as e:
            print("Exception {}".format(e))
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt_input}],
                temperature=0.2,
                max_tokens=500,
                top_p=1,
                frequency_penalty=0.2,
                presence_penalty=0.2,
                timeout=5
                # stop="stop"
            )

        response_json = response["choices"][0]["message"]["content"]
        output = response_json.split("Reasoning:")[1].split("Category:")

        try:
            reasoning = output[0].strip()
            category = output[1].strip()
            reasoning_type["reasoning"] = reasoning
            reasoning_type["category"] = category
        except Exception as e:
            print("Exception {}".format(e))

        reasoning_type["question"] = question
        reasoning_type["answer"] = answer
        reasoning_type["alpaca_answer"] = alternate_answer
        reasoning_type['category'] = output
        qa_reasoning_types[k] = reasoning_type

    with open(os.path.join("data", "output", 'qa_reasoning_types.pickle'), 'wb') as handle:
        pickle.dump(qa_reasoning_types, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return qa_reasoning_types


if __name__ == '__main__':
    # count_mrc()
    # pruned_qa = analyze_mrc()

    with open(os.path.join("data", "output", 'pruned_qa.pickle'), 'rb') as handle:
        pruned_qa = pickle.load(handle)

    qa_reasoning_types = categorize_reasoning(pruned_qa)

    with open(os.path.join("data", "output", 'qa_reasoning_types.pickle'), 'rb') as handle:
        qa_reasoning_types = pickle.load(handle)

    print(len(qa_reasoning_types))

    extract_reasoning_token_based()


