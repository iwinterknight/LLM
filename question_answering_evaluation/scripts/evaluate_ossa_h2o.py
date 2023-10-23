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
    ossa_df = pd.read_pickle(os.path.join("../data", "ossa_h20_output_128_df.pickle"))
    actual_answers, ossa_answers = ossa_df["answers"].tolist(), ossa_df["predictions"].tolist()
    data = {"answers": actual_answers, "predictions": ossa_answers}
    return data


def sentence_similarity_measure(data):
    actual_answers = data["answers"]
    ossa_answers = data["predictions"]

    ossa_sim_cum = 0.0
    total = 0
    answer_encodings = similarity_model.encode(actual_answers)
    index = faiss.IndexFlatL2(answer_encodings.shape[1])
    index.add(answer_encodings)

    for i, (actual_answer, ossa_answer) in enumerate(
            zip(actual_answers, ossa_answers)):
        answer_encoding = similarity_model.encode(actual_answer)
        ossa_encoding = similarity_model.encode(ossa_answer)
        ossa_sim = util.dot_score(answer_encoding, ossa_encoding)

        ossa_sim_cum += ossa_sim[0][0].numpy()
        total += 1

    ossa_sim_cum /= total

    print(f"Sentence similarity scores for ossa : {ossa_sim_cum}")
    return ossa_sim_cum


def adaptive_sentence_similarity_measure(data):
    actual_answers = data["answers"]
    ossa_answers = data["predictions"]

    ossa_sim_cum = 0.0
    total_ossa = 0
    unanswered_ossa = 0
    answer_encodings = similarity_model.encode(actual_answers)
    index = faiss.IndexFlatL2(answer_encodings.shape[1])
    index.add(answer_encodings)

    for i, (actual_answer, ossa_answer) in enumerate(
            zip(actual_answers, ossa_answers)):
        answer_encoding = similarity_model.encode(actual_answer)
        ossa_encoding = similarity_model.encode(ossa_answer)

        ossa_sim = util.dot_score(answer_encoding, ossa_encoding)

        if len(ossa_answer) > 0:
            ossa_sim_cum += ossa_sim[0][0].numpy()
            total_ossa += 1
        else:
            unanswered_ossa += 1

    ossa_sim_cum /= total_ossa

    print(f"Adaptive sentence similarity scores for ossa : {ossa_sim_cum}\t Num unanswered = {unanswered_ossa}")
    return ossa_sim_cum


def rouge_measure(data):
    actual_answers = data["answers"]
    ossa_answers = data["predictions"]

    corrected_ossa_answers = []
    for i, ossa_answer in enumerate(ossa_answers):
        if not ossa_answer:
            ossa_answer = ""
        corrected_ossa_answers.append(ossa_answer)

    ossa_rouge_scores = {"rouge1": {"fmeasure": 0.0, "precision": 0.0, "recall": 0.0},
                           "rougeL": {"fmeasure": 0.0, "precision": 0.0, "recall": 0.0}}
    total = 0
    for i, (actual_answer, corrected_ossa_answer) in enumerate(
            zip(actual_answers, corrected_ossa_answers)):
        score = scorer.score(actual_answer, corrected_ossa_answer)
        ossa_rouge_scores["rouge1"]["fmeasure"] += score["rouge1"].fmeasure
        ossa_rouge_scores["rouge1"]["precision"] += score["rouge1"].precision
        ossa_rouge_scores["rouge1"]["recall"] += score["rouge1"].recall
        ossa_rouge_scores["rougeL"]["fmeasure"] += score["rougeL"].fmeasure
        ossa_rouge_scores["rougeL"]["precision"] += score["rougeL"].precision
        ossa_rouge_scores["rougeL"]["recall"] += score["rougeL"].recall

        total += 1

    ossa_rouge_scores["rouge1"]["fmeasure"] /= total
    ossa_rouge_scores["rouge1"]["precision"] /= total
    ossa_rouge_scores["rouge1"]["recall"] /= total
    ossa_rouge_scores["rougeL"]["fmeasure"] /= total
    ossa_rouge_scores["rougeL"]["precision"] /= total
    ossa_rouge_scores["rougeL"]["recall"] /= total

    print(f"Rogue scores for ossa : {ossa_rouge_scores}")
    return ossa_rouge_scores


def adaptive_rouge_measure(data):
    actual_answers = data["answers"]
    ossa_answers = data["predictions"]

    ossa_rouge_scores = {"rouge1": {"fmeasure": 0.0, "precision": 0.0, "recall": 0.0},
                           "rougeL": {"fmeasure": 0.0, "precision": 0.0, "recall": 0.0}}
    total_ossa, unanswered_ossa = 0, 0
    for i, (actual_answer, ossa_answer) in enumerate(zip(actual_answers, ossa_answers)):
        if not ossa_answer:
            ossa_answer = ""
        if len(ossa_answer) > 0:
            score = scorer.score(actual_answer, ossa_answer)
            ossa_rouge_scores["rouge1"]["fmeasure"] += score["rouge1"].fmeasure
            ossa_rouge_scores["rouge1"]["precision"] += score["rouge1"].precision
            ossa_rouge_scores["rouge1"]["recall"] += score["rouge1"].recall
            ossa_rouge_scores["rougeL"]["fmeasure"] += score["rougeL"].fmeasure
            ossa_rouge_scores["rougeL"]["precision"] += score["rougeL"].precision
            ossa_rouge_scores["rougeL"]["recall"] += score["rougeL"].recall
            total_ossa += 1
        else:
            unanswered_ossa += 1
    ossa_rouge_scores["rouge1"]["fmeasure"] /= total_ossa
    ossa_rouge_scores["rouge1"]["precision"] /= total_ossa
    ossa_rouge_scores["rouge1"]["recall"] /= total_ossa
    ossa_rouge_scores["rougeL"]["fmeasure"] /= total_ossa
    ossa_rouge_scores["rougeL"]["precision"] /= total_ossa
    ossa_rouge_scores["rougeL"]["recall"] /= total_ossa

    print(f"Adaptive rogue scores for ossa : {ossa_rouge_scores}\t Num unanswered = {unanswered_ossa}")
    return ossa_rouge_scores


def create_bleurt_reference_files(data):
    actual_answers = data["answers"]
    ossa_answers = data["predictions"]

    for i, (actual_answer, ossa_answer) in enumerate(zip(actual_answers, ossa_answers)):
        if not actual_answer:
            actual_answer = ""
        with open(os.path.join("../data", "output", "all_candidates"), "a", encoding="utf-8") as f:
            f.write(actual_answer + "\n")

        # if not ossa_answer:
        #     ossa_answer = ""

        with open(os.path.join("../data", "output", "all_candidates.jsonl"), 'a') as outfile:
            json.dump(actual_answer, outfile)
            outfile.write('\n')

        with open(os.path.join("../data", "output", "references_ossa.jsonl"), 'a') as outfile:
            json.dump(ossa_answer, outfile)
            outfile.write('\n')

        # with open(os.path.join("data", "output", "references_ossa"), "a", encoding="utf-8") as f:
        #     f.write(ossa_answer + "\n")


def create_bleurt_adaptive_reference_files(data):
    actual_answers = data["answers"]
    ossa_answers = data["predictions"]

    for i, (actual_answer, ossa_answer) in enumerate(zip(actual_answers, ossa_answers)):
        if ossa_answer and len(ossa_answer) > 0:
            with open(os.path.join("../data", "output", "adaptive_candidates_ossa"), "a", encoding="utf-8") as f:
                f.write(actual_answer + "\n")

            with open(os.path.join("../data", "output", "adaptive_references_ossa"), "a", encoding="utf-8") as f:
                f.write(ossa_answer + "\n")


data = load_data()
sentence_similarity_measure(data)
adaptive_sentence_similarity_measure(data)
rouge_measure(data)
adaptive_rouge_measure(data)

create_bleurt_reference_files(data)
create_bleurt_adaptive_reference_files(data)


# python -m  bleurt.score_files -candidate_file="bleurt/bleurt/test_data/output/all_candidates.jsonl" -reference_file="bleurt/bleurt/test_data/output/references_ossa.jsonl" -bleurt_checkpoint=bleurt/BLEURT-20 2>&1 | tee output_ossa.txt