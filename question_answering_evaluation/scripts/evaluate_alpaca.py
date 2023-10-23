import json
import struct
from rouge_score import rouge_scorer
import faiss
import os
import pandas as pd
from sentence_transformers import SentenceTransformer, util

similarity_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)


def load_data():
    alpaca_df = pd.read_pickle(os.path.join("../data", "alpaca_output_128_df.pickle"))
    actual_answers, alpaca_answers = alpaca_df["answers"].tolist(), alpaca_df["predictions"].tolist()
    data = {"answers": actual_answers, "predictions": alpaca_answers}
    return data


def sentence_similarity_measure(data):
    actual_answers = data["answers"]
    alpaca_answers = data["predictions"]

    alpaca_sim_cum = 0.0
    total = 0
    answer_encodings = similarity_model.encode(actual_answers)
    index = faiss.IndexFlatL2(answer_encodings.shape[1])
    index.add(answer_encodings)

    for i, (actual_answer, alpaca_answer) in enumerate(
            zip(actual_answers, alpaca_answers)):
        answer_encoding = similarity_model.encode(actual_answer)
        alpaca_encoding = similarity_model.encode(alpaca_answer)
        alpaca_sim = util.dot_score(answer_encoding, alpaca_encoding)

        alpaca_sim_cum += alpaca_sim[0][0].numpy()
        total += 1

    alpaca_sim_cum /= total

    print(f"Sentence similarity scores for alpaca : {alpaca_sim_cum}")
    return alpaca_sim_cum


def adaptive_sentence_similarity_measure(data):
    actual_answers = data["answers"]
    alpaca_answers = data["predictions"]

    alpaca_sim_cum = 0.0
    total_alpaca = 0
    unanswered_alpaca = 0
    answer_encodings = similarity_model.encode(actual_answers)
    index = faiss.IndexFlatL2(answer_encodings.shape[1])
    index.add(answer_encodings)

    for i, (actual_answer, alpaca_answer) in enumerate(
            zip(actual_answers, alpaca_answers)):
        answer_encoding = similarity_model.encode(actual_answer)
        alpaca_encoding = similarity_model.encode(alpaca_answer)

        alpaca_sim = util.dot_score(answer_encoding, alpaca_encoding)

        if len(alpaca_answer) > 0:
            alpaca_sim_cum += alpaca_sim[0][0].numpy()
            total_alpaca += 1
        else:
            unanswered_alpaca += 1

    alpaca_sim_cum /= total_alpaca

    print(f"Adaptive sentence similarity scores for alpaca : {alpaca_sim_cum}\t Num unanswered = {unanswered_alpaca}")
    return alpaca_sim_cum


def rouge_measure(data):
    actual_answers = data["answers"]
    alpaca_answers = data["predictions"]

    corrected_alpaca_answers = []
    for i, alpaca_answer in enumerate(alpaca_answers):
        if not alpaca_answer:
            alpaca_answer = ""
        corrected_alpaca_answers.append(alpaca_answer)

    alpaca_rouge_scores = {"rouge1": {"fmeasure": 0.0, "precision": 0.0, "recall": 0.0},
                           "rougeL": {"fmeasure": 0.0, "precision": 0.0, "recall": 0.0}}
    total = 0
    for i, (actual_answer, corrected_alpaca_answer) in enumerate(
            zip(actual_answers, corrected_alpaca_answers)):
        score = scorer.score(actual_answer, corrected_alpaca_answer)
        alpaca_rouge_scores["rouge1"]["fmeasure"] += score["rouge1"].fmeasure
        alpaca_rouge_scores["rouge1"]["precision"] += score["rouge1"].precision
        alpaca_rouge_scores["rouge1"]["recall"] += score["rouge1"].recall
        alpaca_rouge_scores["rougeL"]["fmeasure"] += score["rougeL"].fmeasure
        alpaca_rouge_scores["rougeL"]["precision"] += score["rougeL"].precision
        alpaca_rouge_scores["rougeL"]["recall"] += score["rougeL"].recall

        total += 1

    alpaca_rouge_scores["rouge1"]["fmeasure"] /= total
    alpaca_rouge_scores["rouge1"]["precision"] /= total
    alpaca_rouge_scores["rouge1"]["recall"] /= total
    alpaca_rouge_scores["rougeL"]["fmeasure"] /= total
    alpaca_rouge_scores["rougeL"]["precision"] /= total
    alpaca_rouge_scores["rougeL"]["recall"] /= total

    print(f"Rogue scores for alpaca : {alpaca_rouge_scores}")
    return alpaca_rouge_scores


def adaptive_rouge_measure(data):
    actual_answers = data["answers"]
    alpaca_answers = data["predictions"]

    alpaca_rouge_scores = {"rouge1": {"fmeasure": 0.0, "precision": 0.0, "recall": 0.0},
                           "rougeL": {"fmeasure": 0.0, "precision": 0.0, "recall": 0.0}}
    total_alpaca, unanswered_alpaca = 0, 0
    for i, (actual_answer, alpaca_answer) in enumerate(zip(actual_answers, alpaca_answers)):
        if not alpaca_answer:
            alpaca_answer = ""
        if len(alpaca_answer) > 0:
            score = scorer.score(actual_answer, alpaca_answer)
            alpaca_rouge_scores["rouge1"]["fmeasure"] += score["rouge1"].fmeasure
            alpaca_rouge_scores["rouge1"]["precision"] += score["rouge1"].precision
            alpaca_rouge_scores["rouge1"]["recall"] += score["rouge1"].recall
            alpaca_rouge_scores["rougeL"]["fmeasure"] += score["rougeL"].fmeasure
            alpaca_rouge_scores["rougeL"]["precision"] += score["rougeL"].precision
            alpaca_rouge_scores["rougeL"]["recall"] += score["rougeL"].recall
            total_alpaca += 1
        else:
            unanswered_alpaca += 1
    alpaca_rouge_scores["rouge1"]["fmeasure"] /= total_alpaca
    alpaca_rouge_scores["rouge1"]["precision"] /= total_alpaca
    alpaca_rouge_scores["rouge1"]["recall"] /= total_alpaca
    alpaca_rouge_scores["rougeL"]["fmeasure"] /= total_alpaca
    alpaca_rouge_scores["rougeL"]["precision"] /= total_alpaca
    alpaca_rouge_scores["rougeL"]["recall"] /= total_alpaca

    print(f"Adaptive rogue scores for alpaca : {alpaca_rouge_scores}\t Num unanswered = {unanswered_alpaca}")
    return alpaca_rouge_scores


def create_bleurt_reference_files(data):
    actual_answers = data["answers"]
    alpaca_answers = data["predictions"]

    for i, (actual_answer, alpaca_answer) in enumerate(zip(actual_answers, alpaca_answers)):
        if not actual_answer:
            actual_answer = ""
        with open(os.path.join("../data", "output", "all_candidates"), "a", encoding="utf-8") as f:
            f.write(actual_answer + "\n")

        # if not alpaca_answer:
        #     alpaca_answer = ""

        with open(os.path.join("../data", "output", "all_candidates.jsonl"), 'a') as outfile:
            json.dump(actual_answer, outfile)
            outfile.write('\n')

        with open(os.path.join("../data", "output", "references_alpaca.jsonl"), 'a') as outfile:
            json.dump(alpaca_answer, outfile)
            outfile.write('\n')

        # with open(os.path.join("data", "output", "references_alpaca"), "a", encoding="utf-8") as f:
        #     f.write(alpaca_answer + "\n")


def create_bleurt_adaptive_reference_files(data):
    actual_answers = data["answers"]
    alpaca_answers = data["predictions"]

    for i, (actual_answer, alpaca_answer) in enumerate(zip(actual_answers, alpaca_answers)):
        if alpaca_answer and len(alpaca_answer) > 0:
            with open(os.path.join("../data", "output", "adaptive_candidates_alpaca"), "a", encoding="utf-8") as f:
                f.write(actual_answer + "\n")

            with open(os.path.join("../data", "output", "adaptive_references_alpaca"), "a", encoding="utf-8") as f:
                f.write(alpaca_answer + "\n")


data = load_data()
sentence_similarity_measure(data)
adaptive_sentence_similarity_measure(data)
rouge_measure(data)
adaptive_rouge_measure(data)

create_bleurt_reference_files(data)
create_bleurt_adaptive_reference_files(data)


# python -m  bleurt.score_files -candidate_file="bleurt/bleurt/test_data/output/all_candidates.jsonl" -reference_file="bleurt/bleurt/test_data/output/references_alpaca.jsonl" -bleurt_checkpoint=bleurt/BLEURT-20 2>&1 | tee output_alpaca.txt