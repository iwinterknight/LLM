import os
import pickle
import re
from collections import OrderedDict

import random
from random import seed

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

similarity_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')


def prepare_qa():
    recipeqa = {}

    with open(os.path.join("../data", "latest_saved_df.pickle"), "rb") as f:
        df = pickle.load(f)

    for i, row in enumerate(df.iterrows()):
        # if i > 3:
        #   break
        title = row[1]["title"]
        text = row[1].tolist()

        context = list()
        for t in text[1:]:
            if t == 'stop':
                break
            context.append(t)

        recipeqa[title] = {}
        qa = row[1]["prompt_response"]

        if type(qa) != str:
            continue

        splts = qa.split("\n\n")
        questions, answers = list(), list()
        for splt in splts[1:]:
            qa_splts = splt.split("\nAnswer:")
            if len(qa_splts) < 2:
                qa_splts = splt.split("\n-")
                if len(qa_splts) < 2:
                    qa_splts = splt.split("\n")
                    if len(qa_splts) < 2:
                        continue
            q = qa_splts[0]
            a = qa_splts[1].strip()
            if q and len(q) > 0:
                q_ = re.sub("(?<!\d)[1-9][0-9](?!\d)\.", '', q).strip()
                q_ = re.sub("[0-9]\.", '', q_).strip()
                q_ = q_.replace("Q:", "")
                q_.strip()
                questions.append(q_)
                answers.append(a)
        recipeqa[title]["context"] = context
        recipeqa[title]["questions"] = questions
        recipeqa[title]["answers"] = answers
    return recipeqa


def embed_qa(recipeqa):
    tot_questions, tot_answers, tot_titles = list(), list(), list()
    tot_ques_embeds, tot_ans_embeds = None, None
    i = 0
    for title, qa in recipeqa.items():
        num_questions = len(tot_questions)
        if num_questions > 0:
            print(f"{num_questions} recipe qas embedded...")

        questions = qa["questions"]
        tot_questions.extend(questions)
        answers = qa["answers"]
        tot_answers.extend(answers)
        titles = [title] * len(questions)
        tot_titles.extend(titles)

        ques_embeds = similarity_model.encode(questions)
        ans_embeds = similarity_model.encode(answers)

        with open(os.path.join("../data", "output", 'ques_embeds.pickle'), 'wb') as handle:
            pickle.dump(ques_embeds, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(os.path.join("../data", "output", 'ans_embeds.pickle'), 'wb') as handle:
            pickle.dump(ans_embeds, handle, protocol=pickle.HIGHEST_PROTOCOL)

        if i == 0:
            tot_ques_embeds = ques_embeds
            tot_ans_embeds = ans_embeds
        else:
            try:
                if ques_embeds.shape[1] == tot_ques_embeds.shape[1] and ans_embeds.shape[1] == tot_ans_embeds.shape[1]:
                    tot_ques_embeds = np.vstack((tot_ques_embeds, ques_embeds))
                    tot_ans_embeds = np.vstack((tot_ans_embeds, ans_embeds))
            except Exception as e:
                print(f"{e}\n{ques_embeds}\n{ans_embeds}")

        i += 1
    print(len(tot_questions), len(tot_answers))
    return tot_ques_embeds, tot_ans_embeds, tot_questions, tot_answers, tot_titles


def create_vectors_index(vector_ids, vector_dimension, vector_embeddings, stack_embedding_arrays=False):
    index = faiss.IndexIDMap(faiss.IndexFlatIP(vector_dimension))
    if stack_embedding_arrays:
        vector_embeddings = np.vstack(vector_embeddings)
    faiss.normalize_L2(vector_embeddings)
    index.add_with_ids(vector_embeddings, np.asarray(vector_ids))
    return index


def create_faiss_index(recipeqa, num_samples=100):
    all_recipe_questions, all_recipe_answers = [], []
    recipe_title_encodings, recipe_context_encodings = [], []
    questionids2text, questionids2answers, questionids2titles = OrderedDict(), OrderedDict(), OrderedDict()
    recipe_question_encodings, recipe_answer_encodings = None, None

    seed(0)
    sample_indices_list = []
    for i in range(num_samples):
        sample_indices_list.append(random.randint(0, len(recipeqa)))

    k = 0
    sample_qa = {}
    recipeqa_list = list()
    for title, qa in recipeqa.items():
        recipeqa_list.append((title, qa))

    for i, item in enumerate(recipeqa_list):
        if i in sample_indices_list:
            sample_qa[item[0]] = item[1]

    qa_ctr = 0
    for i, (title, qa) in enumerate(sample_qa.items()):
        recipe_title_encoding = similarity_model.encode(title)
        recipe_title_encoding = recipe_title_encoding.reshape((1, recipe_title_encoding.shape[0]))
        recipe_title_encodings.append(recipe_title_encoding)

        recipe_questions = qa["questions"]
        all_recipe_questions.extend(recipe_questions)
        recipe_answers = qa["answers"]
        all_recipe_answers.extend(recipe_answers)
        for j, (question, answer) in enumerate(zip(recipe_questions, recipe_answers)):
            question_encoding = similarity_model.encode(question)
            question_encoding = np.expand_dims(question_encoding, axis=0)

            answer_encoding = similarity_model.encode(answer)
            answer_encoding = np.expand_dims(answer_encoding, axis=0)

            try:
                if question_encoding.shape == answer_encoding.shape == (1, 768):
                    if qa_ctr == 0:
                        recipe_question_encodings = question_encoding
                        recipe_answer_encodings = answer_encoding
                    else:
                        recipe_question_encodings = np.vstack((recipe_question_encodings, question_encoding))
                        recipe_answer_encodings = np.vstack((recipe_answer_encodings, answer_encoding))

                questionids2text[qa_ctr] = question
                questionids2answers[qa_ctr] = answer
                questionids2titles[qa_ctr] = title
                qa_ctr += 1
            except Exception as e:
                print(f"{e}\n{question_encoding}\n{answer_encoding}")

    recipe_question_index = create_vectors_index(np.asarray(list(questionids2text.keys())), 768,
                                                 recipe_question_encodings, stack_embedding_arrays=False)
    faiss.write_index(recipe_question_index, os.path.join("../data", "output", "faiss-index", "recipe_questions.index"))
    with open(os.path.join("../data", "output", "faiss-index", "questionids2text.pickle"), 'wb') as handle:
        pickle.dump(questionids2text, handle, protocol=pickle.HIGHEST_PROTOCOL)

    recipe_answer_index = create_vectors_index(np.asarray(list(questionids2answers.keys())), 768,
                                                 recipe_answer_encodings, stack_embedding_arrays=False)
    faiss.write_index(recipe_answer_index, os.path.join("../data", "output", "faiss-index", "recipe_answers.index"))
    with open(os.path.join("../data", "output", "faiss-index", "questionids2answers.pickle"), 'wb') as handle:
        pickle.dump(questionids2answers, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(os.path.join("../data", "output", "faiss-index", "questionids2titles.pickle"), 'wb') as handle:
        pickle.dump(questionids2titles, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Data Indexed as Recipe Questions & Answers...\n")
    return


def search_faiss_index(snippet, index, topn):
    snippet_embedding = similarity_model.encode(snippet)
    snippet_embedding = snippet_embedding.reshape((1, snippet_embedding.shape[0]))
    faiss.normalize_L2(snippet_embedding)
    D, I = index.search(snippet_embedding, topn)
    return I[0], D[0]


def read_index():
    recipe_question_index = faiss.read_index(os.path.join("../data", "output", "faiss-index", "recipe_questions.index"))
    with open(os.path.join("../data", "output", "faiss-index", "questionids2text.pickle"), 'rb') as handle:
        questionids2text = pickle.load(handle)

    recipe_answer_index = faiss.read_index(os.path.join("../data", "output", "faiss-index", "recipe_answers.index"))
    with open(os.path.join("../data", "output", "faiss-index", "questionids2answers.pickle"), 'rb') as handle:
        questionids2answers = pickle.load(handle)

    with open(os.path.join("../data", "output", "faiss-index", "questionids2titles.pickle"), 'rb') as handle:
        questionids2titles = pickle.load(handle)

    return recipe_question_index, recipe_answer_index, questionids2text, questionids2answers, questionids2titles


def test():
    recipeqa = prepare_qa()

    TOPN = 100
    create_faiss_index(recipeqa, num_samples=len(recipeqa))

    recipe_question_index, recipe_answer_index, questionids2text, questionids2answers, questionids2titles = read_index()

    similar_questions = {}
    total_repeated_questions = 0
    for id, sample_question in questionids2text.items():
        sample_recipe = questionids2titles[id]
        I, D = search_faiss_index(snippet=sample_question, index=recipe_question_index, topn=TOPN)

        qt = list()
        for iter, index in enumerate(I):
            fetched_question = questionids2text[index]
            fetched_title = questionids2titles[index]
            fetch_score = D[iter]

            if fetched_question is not sample_question:
                qt.append((fetched_question, fetched_title, fetch_score))

            if fetched_question == sample_question:
                total_repeated_questions += 1

        similar_questions[(sample_question, sample_recipe)] = qt
    print(similar_questions, total_repeated_questions)


test()
