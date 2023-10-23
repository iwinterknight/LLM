import os
import pickle
import re
import torch

import random
from random import seed

import faiss
from collections import OrderedDict
import numpy as np
from sentence_transformers import SentenceTransformer

from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')


from transformers import AutoTokenizer, AutoModelForQuestionAnswering, AutoModelForSeq2SeqLM

similarity_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

tokenizer_1 = AutoTokenizer.from_pretrained("deepset/roberta-large-squad2")
model_1 = AutoModelForQuestionAnswering.from_pretrained("deepset/roberta-large-squad2")


tokenizer_2 = AutoTokenizer.from_pretrained("deepset/deberta-v3-large-squad2")
model_2 = AutoModelForQuestionAnswering.from_pretrained("deepset/deberta-v3-large-squad2")

tokenizer_3 = AutoTokenizer.from_pretrained("distilbert-base-cased-distilled-squad")
model_3 = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-cased-distilled-squad")

tokenizer_4 = AutoTokenizer.from_pretrained("MaRiOrOsSi/t5-base-finetuned-question-answering")
model_4 = AutoModelForSeq2SeqLM.from_pretrained("MaRiOrOsSi/t5-base-finetuned-question-answering")


VEC_DIM = 384


def mrc_abstractive(tokenizer, model, context, question):
    input_ids = tokenizer("question : " + question + " context : " + context, return_tensors="pt").input_ids
    outputs = model.generate(input_ids)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer


def mrc_extractive(tokenizer, model, context, question):
    inputs = tokenizer(question, context, return_tensors="pt", max_length=512, truncation=True)

    with torch.no_grad():
        outputs = model(**inputs)

    answer_start_index = outputs.start_logits.argmax()
    answer_end_index = outputs.end_logits.argmax()

    predict_answer_tokens = inputs.input_ids[0, answer_start_index: answer_end_index + 1]
    answer = tokenizer.decode(predict_answer_tokens, skip_special_tokens=True)

    return answer


def word_count(text):
    tokens = tokenizer.tokenize(text)
    return len(tokens)


def slice_context(context, max_len=512):
    wc = word_count(context)
    num_slices = int(wc / max_len)
    if wc % max_len > 0:
        num_slices += 1

    context_slices = []
    ptr = 0
    segmented_context = tokenizer.tokenize(context)
    for slice in range(num_slices):
        context_slice = " ".join(segmented_context[ptr : ptr + max_len])
        ptr += (max_len + 2)
        context_slices.append(context_slice)

    return context_slices


def mrc(context, question):
    context_slices = slice_context(context, max_len=512)
    answer_slices_1, answer_slices_2, answer_slices_3, answer_slices_4 = [], [], [], []
    for context_slice in context_slices:
        answer_slice = mrc_extractive(tokenizer_1, model_1, context_slice, question)
        answer_slices_1.append(answer_slice)
    answer_slices_1.append(mrc_extractive(tokenizer_1, model_1, context, question))

    for context_slice in context_slices:
        answer_slice = mrc_extractive(tokenizer_2, model_2, context_slice, question)
        answer_slices_2.append(answer_slice)
    answer_slices_2.append(mrc_extractive(tokenizer_2, model_2, context, question))

    for context_slice in context_slices:
        answer_slice = mrc_extractive(tokenizer_3, model_3, context_slice, question)
        answer_slices_3.append(answer_slice)
    answer_slices_3.append(mrc_extractive(tokenizer_3, model_3, context, question))

    for context_slice in context_slices:
        answer_slice = mrc_abstractive(tokenizer_4, model_4, context_slice, question)
        answer_slices_4.append(answer_slice)
    answer_slices_4.append(mrc_abstractive(tokenizer_4, model_4, context, question))

    return answer_slices_1, answer_slices_2, answer_slices_3, answer_slices_4


def prepare_qa():
    recipeqa = {}

    with open(os.path.join("data", "latest_saved_df.pickle"), "rb") as f:
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

        with open(os.path.join("data", "output", 'ques_embeds.pickle'), 'wb') as handle:
            pickle.dump(ques_embeds, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(os.path.join("data", "output", 'ans_embeds.pickle'), 'wb') as handle:
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
    questionids2context, questionids2text, questionids2answers, questionids2titles, questionids2outputs = \
        OrderedDict(), OrderedDict(), OrderedDict(), OrderedDict(), OrderedDict()
    recipe_question_encodings, recipe_answer_encodings = None, None

    output_1_encodings, output_2_encodings, output_3_encodings, output_4_encodings = None, None, None, None

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

    with open(os.path.join("data", "output", "faiss-index" 'sample_recipeqa.pickle'), 'wb') as handle:
        pickle.dump(sample_qa, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Total samples = {len(sample_qa)}")
    qa_ctr = 0
    for i, (title, qa) in enumerate(sample_qa.items()):
        if i > 0 and i % 10 == 0:
            print(f"{i} samples processed.")

        recipe_title_encoding = similarity_model.encode(title)
        recipe_title_encoding = recipe_title_encoding.reshape((1, recipe_title_encoding.shape[0]))
        recipe_title_encodings.append(recipe_title_encoding)

        recipe_context = qa["context"]
        recipe_context = " ".join(recipe_context)
        recipe_questions = qa["questions"]
        all_recipe_questions.extend(recipe_questions)
        recipe_answers = qa["answers"]
        all_recipe_answers.extend(recipe_answers)
        for j, (question, answer) in enumerate(zip(recipe_questions, recipe_answers)):
            # question_encoding = similarity_model.encode(question)
            # question_encoding = np.expand_dims(question_encoding, axis=0)
            #
            # answer_encoding = similarity_model.encode(answer)
            # answer_encoding = np.expand_dims(answer_encoding, axis=0)


            output_1, output_2, output_3, output_4 = mrc(recipe_context, question)

            # output_1_encoding = similarity_model.encode(output_1['answer'])
            # output_1_encoding = np.expand_dims(output_1_encoding, axis=0)
            #
            # output_2_encoding = similarity_model.encode(output_2['answer'])
            # output_2_encoding = np.expand_dims(output_2_encoding, axis=0)
            #
            # output_3_encoding = similarity_model.encode(output_3['answer'])
            # output_3_encoding = np.expand_dims(output_3_encoding, axis=0)
            #
            # output_4_encoding = similarity_model.encode(output_4[0]['generated_text'])
            # output_4_encoding = np.expand_dims(output_4_encoding, axis=0)

            try:
            #     if question_encoding.shape == answer_encoding.shape == (1, VEC_DIM):
            #         if qa_ctr == 0:
            #             recipe_question_encodings = question_encoding
            #             recipe_answer_encodings = answer_encoding
            #             # output_1_encodings = output_1_encoding
            #             # output_2_encodings = output_2_encoding
            #             # output_3_encodings = output_3_encoding
            #             # output_4_encodings = output_4_encoding
            #         else:
            #             recipe_question_encodings = np.vstack((recipe_question_encodings, question_encoding))
            #             recipe_answer_encodings = np.vstack((recipe_answer_encodings, answer_encoding))
            #             # output_1_encodings = np.vstack((output_1_encodings, output_1_encoding))
            #             # output_2_encodings = np.vstack((output_2_encodings, output_2_encoding))
            #             # output_3_encodings = np.vstack((output_3_encodings, output_3_encoding))
            #             # output_4_encodings = np.vstack((output_4_encodings, output_4_encoding))

                questionids2context[qa_ctr] = recipe_context
                questionids2text[qa_ctr] = question
                questionids2answers[qa_ctr] = answer
                questionids2outputs[qa_ctr] = (output_1, output_2, output_3, output_4)
                questionids2titles[qa_ctr] = title

                qa_ctr += 1
            except Exception as e:
                # print(f"{e}\n{question_encoding}\n{answer_encoding}")
                print(f"{e}")

    # recipe_question_index = create_vectors_index(np.asarray(list(questionids2text.keys())), VEC_DIM,
    #                                              recipe_question_encodings, stack_embedding_arrays=False)
    # faiss.write_index(recipe_question_index, os.path.join("data", "output", "faiss-index", "recipe_questions.index"))

    # recipe_answer_index = create_vectors_index(np.asarray(list(questionids2answers.keys())), VEC_DIM,
    #                                              recipe_answer_encodings, stack_embedding_arrays=False)
    # faiss.write_index(recipe_answer_index, os.path.join("data", "output", "faiss-index", "recipe_answers.index"))

    with open(os.path.join("data", "output", "faiss-index", "questionids2text.pickle"), 'wb') as handle:
        pickle.dump(questionids2text, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(os.path.join("data", "output", "faiss-index", "questionids2answers.pickle"), 'wb') as handle:
        pickle.dump(questionids2answers, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(os.path.join("data", "output", "faiss-index", "questionids2titles.pickle"), 'wb') as handle:
        pickle.dump(questionids2titles, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(os.path.join("data", "output", "faiss-index", "questionids2outputs.pickle"), 'wb') as handle:
        pickle.dump(questionids2outputs, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(os.path.join("data", "output", "faiss-index", "questionids2context.pickle"), 'wb') as handle:
        pickle.dump(questionids2context, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # with open(os.path.join("data", "output", "faiss-index", "output_1_encodings.pickle"), 'wb') as handle:
    #     pickle.dump(output_1_encodings, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #
    # with open(os.path.join("data", "output", "faiss-index", "output_2_encodings.pickle"), 'wb') as handle:
    #     pickle.dump(output_2_encodings, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #
    # with open(os.path.join("data", "output", "faiss-index", "output_3_encodings.pickle"), 'wb') as handle:
    #     pickle.dump(output_3_encodings, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #
    # with open(os.path.join("data", "output", "faiss-index", "output_4_encodings.pickle"), 'wb') as handle:
    #     pickle.dump(output_4_encodings, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Data Indexed as Recipe Questions & Answers...\n")
    return


def search_faiss_index(snippet, index, topn):
    snippet_embedding = similarity_model.encode(snippet)
    snippet_embedding = snippet_embedding.reshape((1, snippet_embedding.shape[0]))
    faiss.normalize_L2(snippet_embedding)
    D, I = index.search(snippet_embedding, topn)
    return I[0], D[0]


def read_index():
    with open(os.path.join("data", "output", "faiss-index", "questionids2context.pickle"), 'rb') as handle:
        questionids2context = pickle.load(handle)

    # recipe_question_index = faiss.read_index(os.path.join("data", "output", "faiss-index", "recipe_questions.index"))
    with open(os.path.join("data", "output", "faiss-index", "questionids2text.pickle"), 'rb') as handle:
        questionids2text = pickle.load(handle)

    # recipe_answer_index = faiss.read_index(os.path.join("data", "output", "faiss-index", "recipe_answers.index"))
    with open(os.path.join("data", "output", "faiss-index", "questionids2answers.pickle"), 'rb') as handle:
        questionids2answers = pickle.load(handle)

    with open(os.path.join("data", "output", "faiss-index", "questionids2titles.pickle"), 'rb') as handle:
        questionids2titles = pickle.load(handle)

    # return recipe_question_index, recipe_answer_index, questionids2text, questionids2answers, questionids2titles
    return questionids2text, questionids2answers, questionids2titles


def test():
    recipeqa = prepare_qa()

    TOPN = 100
    # create_faiss_index(recipeqa, num_samples=50)

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
