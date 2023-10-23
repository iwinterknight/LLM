import os
import pickle

with open(os.path.join("../data", "output", "faiss-index", "similar_questions.pickle"), 'rb') as handle:
    similar_questions = pickle.load(handle)

SCORE_THRESHOLD = 0.5

NUM_SIMILAR_PER_ANCHOR = 100
total_anchors = len(similar_questions)
total_lq_anchors, total_hq_anchors = 0.0, 0.0
avg_task_similarity_ratio = 0.0
avg_lq_task_similarity_ratio, avg_hq_task_similarity_ratio = 0.0, 0.0
for k, v in similar_questions.items():
    anchor_question, anchor_question_title = k
    same_task_ratio = 0.0
    for ques, title, score in v:
        if ques != anchor_question and title == anchor_question_title and score >= SCORE_THRESHOLD:
            same_task_ratio += 1
    same_task_ratio /= NUM_SIMILAR_PER_ANCHOR
    avg_task_similarity_ratio += same_task_ratio

    if same_task_ratio < 0.25:
        avg_lq_task_similarity_ratio += same_task_ratio
        total_lq_anchors += 1

    if same_task_ratio > 0.25:
        avg_hq_task_similarity_ratio += same_task_ratio
        total_hq_anchors += 1

avg_task_similarity_ratio /= total_anchors
avg_lq_task_similarity_ratio /= total_lq_anchors
avg_hq_task_similarity_ratio /= total_hq_anchors

print(f"Average Task Similarity : {avg_task_similarity_ratio}\nAverage Lower Quartile Task Similarity : "
      f"{avg_lq_task_similarity_ratio}\nAverage Higher Quartile Task Similarity : {avg_hq_task_similarity_ratio}")
