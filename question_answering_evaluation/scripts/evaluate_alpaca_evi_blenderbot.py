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
    alpaca_df = pd.read_pickle(os.path.join("../data", "commonsense_qa_alpaca_output_df.pickle"))
    evi_df = pd.read_pickle(os.path.join("../data", "commonsense_qa_evi_output_df.pickle"))
    blenderbot_df = pd.read_pickle(os.path.join("../data", "commonsense_qa_blenderbot_output_df.pickle"))

    actual_answers, alpaca_answers, evi_answers, blenderbot_answers = alpaca_df["answers"].tolist(), alpaca_df[
        "alpaca_responses"].tolist(), evi_df["evi_responses"].tolist(), blenderbot_df["blenderbot_responses"].tolist()

    data = {"actual_answers": actual_answers, "alpaca_answers": alpaca_answers, "evi_answers": evi_answers,
            "blenderbot_answers": blenderbot_answers}

    return data


def sentence_similarity_measure(data):
    actual_answers = data["actual_answers"]
    alpaca_answers = data["alpaca_answers"]
    evi_answers = data["evi_answers"]
    blenderbot_answers = data["blenderbot_answers"]

    alpaca_sim_cum, evi_sim_cum, blenderbot_sim_cum = 0.0, 0.0, 0.0
    total = 0
    answer_encodings = similarity_model.encode(actual_answers)
    index = faiss.IndexFlatL2(answer_encodings.shape[1])
    index.add(answer_encodings)

    for i, (actual_answer, alpaca_answer, evi_answer, blenderbot_answer) in enumerate(
            zip(actual_answers, alpaca_answers, evi_answers, blenderbot_answers)):
        answer_encoding = similarity_model.encode(actual_answer)
        alpaca_encoding = similarity_model.encode(alpaca_answer)
        evi_encoding = similarity_model.encode(evi_answer)
        blenderbot_encoding = similarity_model.encode(blenderbot_answer)

        alpaca_sim = util.dot_score(answer_encoding, alpaca_encoding)
        evi_sim = util.dot_score(answer_encoding, evi_encoding)
        blenderbot_sim = util.dot_score(answer_encoding, blenderbot_encoding)

        alpaca_sim_cum += alpaca_sim[0][0].numpy()
        evi_sim_cum += evi_sim[0][0].numpy()
        blenderbot_sim_cum += blenderbot_sim[0][0].numpy()
        total += 1

    alpaca_sim_cum /= total
    evi_sim_cum /= total
    blenderbot_sim_cum /= total

    print(f"Sentence similarity scores for alpaca : {alpaca_sim_cum}")
    print(f"Sentence similarity scores for evi : {evi_sim_cum}")
    print(f"Sentence similarity scores for blenderbot : {blenderbot_sim_cum}")

    return alpaca_sim_cum, evi_sim_cum, blenderbot_sim_cum


def adaptive_sentence_similarity_measure(data):
    actual_answers = data["actual_answers"]
    alpaca_answers = data["alpaca_answers"]
    evi_answers = data["evi_answers"]
    blenderbot_answers = data["blenderbot_answers"]

    alpaca_sim_cum, evi_sim_cum, blenderbot_sim_cum = 0.0, 0.0, 0.0
    total_alpaca, total_evi, total_blenderbot = 0, 0, 0
    unanswered_alpaca, unanswered_evi, unanswered_blenderbot = 0, 0, 0
    answer_encodings = similarity_model.encode(actual_answers)
    index = faiss.IndexFlatL2(answer_encodings.shape[1])
    index.add(answer_encodings)

    for i, (actual_answer, alpaca_answer, evi_answer, blenderbot_answer) in enumerate(
            zip(actual_answers, alpaca_answers, evi_answers, blenderbot_answers)):
        answer_encoding = similarity_model.encode(actual_answer)
        alpaca_encoding = similarity_model.encode(alpaca_answer)
        evi_encoding = similarity_model.encode(evi_answer)
        blenderbot_encoding = similarity_model.encode(blenderbot_answer)

        alpaca_sim = util.dot_score(answer_encoding, alpaca_encoding)
        evi_sim = util.dot_score(answer_encoding, evi_encoding)
        blenderbot_sim = util.dot_score(answer_encoding, blenderbot_encoding)

        if len(alpaca_answer) > 0:
            alpaca_sim_cum += alpaca_sim[0][0].numpy()
            total_alpaca += 1
        else:
            unanswered_alpaca += 1

        if len(evi_answer) > 0:
            evi_sim_cum += evi_sim[0][0].numpy()
            total_evi += 1
        else:
            unanswered_evi += 1

        if blenderbot_answer and len(blenderbot_answer) > 0:
            blenderbot_sim_cum += blenderbot_sim[0][0].numpy()
            total_blenderbot += 1
        else:
            unanswered_blenderbot += 1

    alpaca_sim_cum /= total_alpaca
    evi_sim_cum /= total_evi
    blenderbot_sim_cum /= total_blenderbot

    print(f"Adaptive sentence similarity scores for alpaca : {alpaca_sim_cum}\t Num unanswered = {unanswered_alpaca}")
    print(f"Adaptive sentence similarity scores for evi : {evi_sim_cum}\t Num unanswered = {unanswered_evi}")
    print(
        f"Adaptive sentence similarity scores for blenderbot : {blenderbot_sim_cum}\t Num unanswered = {unanswered_blenderbot}")

    return alpaca_sim_cum, evi_sim_cum, blenderbot_sim_cum


def rouge_measure(data):
    actual_answers = data["actual_answers"]
    alpaca_answers = data["alpaca_answers"]
    evi_answers = data["evi_answers"]
    blenderbot_answers = data["blenderbot_answers"]

    corrected_alpaca_answers = []
    for i, alpaca_answer in enumerate(alpaca_answers):
        if not alpaca_answer:
            alpaca_answer = ""
        corrected_alpaca_answers.append(alpaca_answer)

    corrected_evi_answers = []
    for i, evi_answer in enumerate(evi_answers):
        if not evi_answer:
            evi_answer = ""
        corrected_evi_answers.append(evi_answer)

    corrected_blenderbot_answers = []
    for i, blenderbot_answer in enumerate(blenderbot_answers):
        if not blenderbot_answer:
            blenderbot_answer = ""
        corrected_blenderbot_answers.append(blenderbot_answer)

    alpaca_rouge_scores = {"rouge1": {"fmeasure": 0.0, "precision": 0.0, "recall": 0.0},
                           "rougeL": {"fmeasure": 0.0, "precision": 0.0, "recall": 0.0}}
    evi_rouge_scores = {"rouge1": {"fmeasure": 0.0, "precision": 0.0, "recall": 0.0},
                        "rougeL": {"fmeasure": 0.0, "precision": 0.0, "recall": 0.0}}
    blenderbot_rouge_scores = {"rouge1": {"fmeasure": 0.0, "precision": 0.0, "recall": 0.0},
                               "rougeL": {"fmeasure": 0.0, "precision": 0.0, "recall": 0.0}}
    total = 0
    for i, (actual_answer, corrected_alpaca_answer, corrected_evi_answer, corrected_blenderbot_answer) in enumerate(
            zip(actual_answers, corrected_alpaca_answers, corrected_evi_answers, corrected_blenderbot_answers)):
        score = scorer.score(actual_answer, corrected_alpaca_answer)
        alpaca_rouge_scores["rouge1"]["fmeasure"] += score["rouge1"].fmeasure
        alpaca_rouge_scores["rouge1"]["precision"] += score["rouge1"].precision
        alpaca_rouge_scores["rouge1"]["recall"] += score["rouge1"].recall
        alpaca_rouge_scores["rougeL"]["fmeasure"] += score["rougeL"].fmeasure
        alpaca_rouge_scores["rougeL"]["precision"] += score["rougeL"].precision
        alpaca_rouge_scores["rougeL"]["recall"] += score["rougeL"].recall

        score = scorer.score(actual_answer, corrected_evi_answer)
        evi_rouge_scores["rouge1"]["fmeasure"] += score["rouge1"].fmeasure
        evi_rouge_scores["rouge1"]["precision"] += score["rouge1"].precision
        evi_rouge_scores["rouge1"]["recall"] += score["rouge1"].recall
        evi_rouge_scores["rougeL"]["fmeasure"] += score["rougeL"].fmeasure
        evi_rouge_scores["rougeL"]["precision"] += score["rougeL"].precision
        evi_rouge_scores["rougeL"]["recall"] += score["rougeL"].recall

        score = scorer.score(actual_answer, corrected_blenderbot_answer)
        blenderbot_rouge_scores["rouge1"]["fmeasure"] += score["rouge1"].fmeasure
        blenderbot_rouge_scores["rouge1"]["precision"] += score["rouge1"].precision
        blenderbot_rouge_scores["rouge1"]["recall"] += score["rouge1"].recall
        blenderbot_rouge_scores["rougeL"]["fmeasure"] += score["rougeL"].fmeasure
        blenderbot_rouge_scores["rougeL"]["precision"] += score["rougeL"].precision
        blenderbot_rouge_scores["rougeL"]["recall"] += score["rougeL"].recall
        total += 1

    alpaca_rouge_scores["rouge1"]["fmeasure"] /= total
    alpaca_rouge_scores["rouge1"]["precision"] /= total
    alpaca_rouge_scores["rouge1"]["recall"] /= total
    alpaca_rouge_scores["rougeL"]["fmeasure"] /= total
    alpaca_rouge_scores["rougeL"]["precision"] /= total
    alpaca_rouge_scores["rougeL"]["recall"] /= total

    evi_rouge_scores["rouge1"]["fmeasure"] /= total
    evi_rouge_scores["rouge1"]["precision"] /= total
    evi_rouge_scores["rouge1"]["recall"] /= total
    evi_rouge_scores["rougeL"]["fmeasure"] /= total
    evi_rouge_scores["rougeL"]["precision"] /= total
    evi_rouge_scores["rougeL"]["recall"] /= total

    blenderbot_rouge_scores["rouge1"]["fmeasure"] /= total
    blenderbot_rouge_scores["rouge1"]["precision"] /= total
    blenderbot_rouge_scores["rouge1"]["recall"] /= total
    blenderbot_rouge_scores["rougeL"]["fmeasure"] /= total
    blenderbot_rouge_scores["rougeL"]["precision"] /= total
    blenderbot_rouge_scores["rougeL"]["recall"] /= total

    print(f"Rogue scores for alpaca : {alpaca_rouge_scores}")
    print(f"Rogue scores for evi : {evi_rouge_scores}")
    print(f"Rogue scores for blenderbot : {blenderbot_rouge_scores}")

    return alpaca_rouge_scores, evi_rouge_scores, blenderbot_rouge_scores


def adaptive_rouge_measure(data):
    actual_answers = data["actual_answers"]
    alpaca_answers = data["alpaca_answers"]
    evi_answers = data["evi_answers"]
    blenderbot_answers = data["blenderbot_answers"]

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

    evi_rouge_scores = {"rouge1": {"fmeasure": 0.0, "precision": 0.0, "recall": 0.0},
                        "rougeL": {"fmeasure": 0.0, "precision": 0.0, "recall": 0.0}}
    total_evi, unanswered_evi = 0, 0
    for i, (actual_answer, evi_answer) in enumerate(zip(actual_answers, evi_answers)):
        if not evi_answer:
            evi_answer = ""
        if len(evi_answer) > 0:
            score = scorer.score(actual_answer, evi_answer)
            evi_rouge_scores["rouge1"]["fmeasure"] += score["rouge1"].fmeasure
            evi_rouge_scores["rouge1"]["precision"] += score["rouge1"].precision
            evi_rouge_scores["rouge1"]["recall"] += score["rouge1"].recall
            evi_rouge_scores["rougeL"]["fmeasure"] += score["rougeL"].fmeasure
            evi_rouge_scores["rougeL"]["precision"] += score["rougeL"].precision
            evi_rouge_scores["rougeL"]["recall"] += score["rougeL"].recall
            total_evi += 1
        else:
            unanswered_evi += 1
    evi_rouge_scores["rouge1"]["fmeasure"] /= total_evi
    evi_rouge_scores["rouge1"]["precision"] /= total_evi
    evi_rouge_scores["rouge1"]["recall"] /= total_evi
    evi_rouge_scores["rougeL"]["fmeasure"] /= total_evi
    evi_rouge_scores["rougeL"]["precision"] /= total_evi
    evi_rouge_scores["rougeL"]["recall"] /= total_evi

    blenderbot_rouge_scores = {"rouge1": {"fmeasure": 0.0, "precision": 0.0, "recall": 0.0},
                               "rougeL": {"fmeasure": 0.0, "precision": 0.0, "recall": 0.0}}
    total_blenderbot, unanswered_blenderbot = 0, 0
    for i, (actual_answer, blenderbot_answer) in enumerate(zip(actual_answers, blenderbot_answers)):
        if not blenderbot_answer:
            blenderbot_answer = ""
        if len(blenderbot_answer) > 0:
            score = scorer.score(actual_answer, blenderbot_answer)
            blenderbot_rouge_scores["rouge1"]["fmeasure"] += score["rouge1"].fmeasure
            blenderbot_rouge_scores["rouge1"]["precision"] += score["rouge1"].precision
            blenderbot_rouge_scores["rouge1"]["recall"] += score["rouge1"].recall
            blenderbot_rouge_scores["rougeL"]["fmeasure"] += score["rougeL"].fmeasure
            blenderbot_rouge_scores["rougeL"]["precision"] += score["rougeL"].precision
            blenderbot_rouge_scores["rougeL"]["recall"] += score["rougeL"].recall
            total_blenderbot += 1
        else:
            unanswered_blenderbot += 1
    blenderbot_rouge_scores["rouge1"]["fmeasure"] /= total_blenderbot
    blenderbot_rouge_scores["rouge1"]["precision"] /= total_blenderbot
    blenderbot_rouge_scores["rouge1"]["recall"] /= total_blenderbot
    blenderbot_rouge_scores["rougeL"]["fmeasure"] /= total_blenderbot
    blenderbot_rouge_scores["rougeL"]["precision"] /= total_blenderbot
    blenderbot_rouge_scores["rougeL"]["recall"] /= total_blenderbot

    print(f"Adaptive rogue scores for alpaca : {alpaca_rouge_scores}\t Num unanswered = {unanswered_alpaca}")
    print(f"Adaptive rogue scores for evi : {evi_rouge_scores}\t Num unanswered = {unanswered_evi}")
    print(
        f"Adaptive rogue scores for blenderbot : {blenderbot_rouge_scores}\t Num unanswered = {unanswered_blenderbot}")

    return alpaca_rouge_scores, evi_rouge_scores, blenderbot_rouge_scores


def create_bleurt_reference_files(data):
    actual_answers = data["actual_answers"]
    alpaca_answers = data["alpaca_answers"]
    evi_answers = data["evi_answers"]
    blenderbot_answers = data["blenderbot_answers"]

    for i, (actual_answer, alpaca_answer, evi_answer, blenderbot_answer) in enumerate(zip(actual_answers, alpaca_answers, evi_answers, blenderbot_answers)):
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

        if not evi_answer:
            evi_answer = ""
        with open(os.path.join("../data", "output", "references_evi"), "a", encoding="utf-8") as f:
            f.write(evi_answer + "\n")

        if not blenderbot_answer:
            blenderbot_answer = ""
        with open(os.path.join("../data", "output", "references_blenderbot"), "a", encoding="utf-8") as f:
            f.write(blenderbot_answer + "\n")


def create_bleurt_adaptive_reference_files(data):
    actual_answers = data["actual_answers"]
    alpaca_answers = data["alpaca_answers"]
    evi_answers = data["evi_answers"]
    blenderbot_answers = data["blenderbot_answers"]

    for i, (actual_answer, alpaca_answer, evi_answer, blenderbot_answer) in enumerate(zip(actual_answers, alpaca_answers, evi_answers, blenderbot_answers)):
        if alpaca_answer and len(alpaca_answer) > 0:
            with open(os.path.join("../data", "output", "adaptive_candidates_alpaca"), "a", encoding="utf-8") as f:
                f.write(actual_answer + "\n")

            with open(os.path.join("../data", "output", "adaptive_references_alpaca"), "a", encoding="utf-8") as f:
                f.write(alpaca_answer + "\n")

        if evi_answer and len(evi_answer) > 0:
            with open(os.path.join("../data", "output", "adaptive_candidates_evi"), "a", encoding="utf-8") as f:
                f.write(actual_answer + "\n")

            with open(os.path.join("../data", "output", "adaptive_references_evi"), "a", encoding="utf-8") as f:
                f.write(evi_answer + "\n")

        if not blenderbot_answer:
            with open(os.path.join("../data", "output", "adaptive_candidates_blenderbot"), "a", encoding="utf-8") as f:
                f.write(actual_answer + "\n")

            with open(os.path.join("../data", "output", "adaptive_references_blenderbot"), "a", encoding="utf-8") as f:
                f.write(blenderbot_answer + "\n")


data = load_data()
sentence_similarity_measure(data)
adaptive_sentence_similarity_measure(data)
rouge_measure(data)
adaptive_rouge_measure(data)

create_bleurt_reference_files(data)
create_bleurt_adaptive_reference_files(data)



# python -m  bleurt.score_files -candidate_file="bleurt/test_data/output/all_candidates.jsonl" -reference_file="bleurt/test_data/output/references_alpaca.jsonl" -bleurt_checkpoint=BLEURT-20 2>&1 | tee output_alpaca.txt
# python -m  bleurt.score_files -candidate_file=bleurt/test_data/output/all_candidates -reference_file=bleurt/test_data/output/references_evi -bleurt_checkpoint=BLEURT-20 2>&1 | tee output_evi.txt
# python -m  bleurt.score_files -candidate_file=bleurt/test_data/output/all_candidates -reference_file=bleurt/test_data/output/references_blenderbot -bleurt_checkpoint=BLEURT-20 2>&1 | tee output_blenderbot.txt
# python -m  bleurt.score_files -candidate_file=bleurt/test_data/output/adaptive_candidates_evi -reference_file=bleurt/test_data/output/adaptive_references_evi -bleurt_checkpoint=BLEURT-20 2>&1 | tee output_adaptive_evi.txt