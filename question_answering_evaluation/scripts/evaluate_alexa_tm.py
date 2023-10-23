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
    atm_df = pd.read_pickle(os.path.join("../data", "alexa_tm_df.pickle"))
    atm_df.predictions_conversational.fillna(value="", inplace=True)
    actual_answers, atm_answers = atm_df["answers"].tolist(), atm_df["predictions_conversational"].tolist()
    data = {"answers": actual_answers, "predictions": atm_answers}
    return data


def sentence_similarity_measure(data):
    actual_answers = data["answers"]
    atm_answers = data["predictions"]

    atm_sim_cum = 0.0
    total = 0
    answer_encodings = similarity_model.encode(actual_answers)
    index = faiss.IndexFlatL2(answer_encodings.shape[1])
    index.add(answer_encodings)

    for i, (actual_answer, atm_answer) in enumerate(
            zip(actual_answers, atm_answers)):
        answer_encoding = similarity_model.encode(actual_answer)
        atm_encoding = similarity_model.encode(atm_answer)
        atm_sim = util.dot_score(answer_encoding, atm_encoding)

        atm_sim_cum += atm_sim[0][0].numpy()
        total += 1

    atm_sim_cum /= total

    print(f"Sentence similarity scores for atm : {atm_sim_cum}")
    return atm_sim_cum


def adaptive_sentence_similarity_measure(data):
    actual_answers = data["answers"]
    atm_answers = data["predictions"]

    atm_sim_cum = 0.0
    total_atm = 0
    unanswered_atm = 0
    answer_encodings = similarity_model.encode(actual_answers)
    index = faiss.IndexFlatL2(answer_encodings.shape[1])
    index.add(answer_encodings)

    for i, (actual_answer, atm_answer) in enumerate(
            zip(actual_answers, atm_answers)):
        answer_encoding = similarity_model.encode(actual_answer)
        atm_encoding = similarity_model.encode(atm_answer)

        atm_sim = util.dot_score(answer_encoding, atm_encoding)

        if len(atm_answer) > 0:
            atm_sim_cum += atm_sim[0][0].numpy()
            total_atm += 1
        else:
            unanswered_atm += 1

    atm_sim_cum /= total_atm

    print(f"Adaptive sentence similarity scores for atm : {atm_sim_cum}\t Num unanswered = {unanswered_atm}")
    return atm_sim_cum


def rouge_measure(data):
    actual_answers = data["answers"]
    atm_answers = data["predictions"]

    corrected_atm_answers = []
    for i, atm_answer in enumerate(atm_answers):
        if not atm_answer:
            atm_answer = ""
        corrected_atm_answers.append(atm_answer)

    atm_rouge_scores = {"rouge1": {"fmeasure": 0.0, "precision": 0.0, "recall": 0.0},
                           "rougeL": {"fmeasure": 0.0, "precision": 0.0, "recall": 0.0}}
    total = 0
    for i, (actual_answer, corrected_atm_answer) in enumerate(
            zip(actual_answers, corrected_atm_answers)):
        score = scorer.score(actual_answer, corrected_atm_answer)
        atm_rouge_scores["rouge1"]["fmeasure"] += score["rouge1"].fmeasure
        atm_rouge_scores["rouge1"]["precision"] += score["rouge1"].precision
        atm_rouge_scores["rouge1"]["recall"] += score["rouge1"].recall
        atm_rouge_scores["rougeL"]["fmeasure"] += score["rougeL"].fmeasure
        atm_rouge_scores["rougeL"]["precision"] += score["rougeL"].precision
        atm_rouge_scores["rougeL"]["recall"] += score["rougeL"].recall

        total += 1

    atm_rouge_scores["rouge1"]["fmeasure"] /= total
    atm_rouge_scores["rouge1"]["precision"] /= total
    atm_rouge_scores["rouge1"]["recall"] /= total
    atm_rouge_scores["rougeL"]["fmeasure"] /= total
    atm_rouge_scores["rougeL"]["precision"] /= total
    atm_rouge_scores["rougeL"]["recall"] /= total

    print(f"Rogue scores for atm : {atm_rouge_scores}")
    return atm_rouge_scores


def adaptive_rouge_measure(data):
    actual_answers = data["answers"]
    atm_answers = data["predictions"]

    atm_rouge_scores = {"rouge1": {"fmeasure": 0.0, "precision": 0.0, "recall": 0.0},
                           "rougeL": {"fmeasure": 0.0, "precision": 0.0, "recall": 0.0}}
    total_atm, unanswered_atm = 0, 0
    for i, (actual_answer, atm_answer) in enumerate(zip(actual_answers, atm_answers)):
        if not atm_answer:
            atm_answer = ""
        if len(atm_answer) > 0:
            score = scorer.score(actual_answer, atm_answer)
            atm_rouge_scores["rouge1"]["fmeasure"] += score["rouge1"].fmeasure
            atm_rouge_scores["rouge1"]["precision"] += score["rouge1"].precision
            atm_rouge_scores["rouge1"]["recall"] += score["rouge1"].recall
            atm_rouge_scores["rougeL"]["fmeasure"] += score["rougeL"].fmeasure
            atm_rouge_scores["rougeL"]["precision"] += score["rougeL"].precision
            atm_rouge_scores["rougeL"]["recall"] += score["rougeL"].recall
            total_atm += 1
        else:
            unanswered_atm += 1
    atm_rouge_scores["rouge1"]["fmeasure"] /= total_atm
    atm_rouge_scores["rouge1"]["precision"] /= total_atm
    atm_rouge_scores["rouge1"]["recall"] /= total_atm
    atm_rouge_scores["rougeL"]["fmeasure"] /= total_atm
    atm_rouge_scores["rougeL"]["precision"] /= total_atm
    atm_rouge_scores["rougeL"]["recall"] /= total_atm

    print(f"Adaptive rogue scores for atm : {atm_rouge_scores}\t Num unanswered = {unanswered_atm}")
    return atm_rouge_scores


def create_bleurt_reference_files(data):
    actual_answers = data["answers"]
    atm_answers = data["predictions"]

    for i, (actual_answer, atm_answer) in enumerate(zip(actual_answers, atm_answers)):
        if not actual_answer:
            actual_answer = ""
        with open(os.path.join("../data", "output", "all_candidates"), "a", encoding="utf-8") as f:
            f.write(actual_answer + "\n")

        # if not atm_answer:
        #     atm_answer = ""

        with open(os.path.join("../data", "output", "all_candidates.jsonl"), 'a') as outfile:
            json.dump(actual_answer, outfile)
            outfile.write('\n')

        with open(os.path.join("../data", "output", "references_atm.jsonl"), 'a') as outfile:
            json.dump(atm_answer, outfile)
            outfile.write('\n')

        # with open(os.path.join("data", "output", "references_atm"), "a", encoding="utf-8") as f:
        #     f.write(atm_answer + "\n")


def create_bleurt_adaptive_reference_files(data):
    actual_answers = data["answers"]
    atm_answers = data["predictions"]

    for i, (actual_answer, atm_answer) in enumerate(zip(actual_answers, atm_answers)):
        if atm_answer and len(atm_answer) > 0:
            with open(os.path.join("../data", "output", "adaptive_candidates_atm"), "a", encoding="utf-8") as f:
                f.write(actual_answer + "\n")

            with open(os.path.join("../data", "output", "adaptive_references_atm"), "a", encoding="utf-8") as f:
                f.write(atm_answer + "\n")


data = load_data()
sentence_similarity_measure(data)
adaptive_sentence_similarity_measure(data)
rouge_measure(data)
adaptive_rouge_measure(data)

create_bleurt_reference_files(data)
create_bleurt_adaptive_reference_files(data)


# python -m  bleurt.score_files -candidate_file="bleurt/bleurt/test_data/output/all_candidates.jsonl" -reference_file="bleurt/bleurt/test_data/output/references_atm.jsonl" -bleurt_checkpoint=bleurt/BLEURT-20 2>&1 | tee output_atm.txt