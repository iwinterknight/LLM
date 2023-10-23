# import json
# import pickle
# import numpy as np
# import faiss
# import os
# import pandas as pd
# from collections import OrderedDict
# from sentence_transformers import SentenceTransformer, util
#
# similarity_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
#
#
# def create_vectors_index(vector_ids, vector_dimension, vector_embeddings, stack_embedding_arrays=False):
#     index = faiss.IndexIDMap(faiss.IndexFlatIP(vector_dimension))
#     if stack_embedding_arrays:
#         vector_embeddings = np.vstack(vector_embeddings)
#     faiss.normalize_L2(vector_embeddings)
#     index.add_with_ids(vector_embeddings, np.asarray(vector_ids))
#     return index
#
#
# def create_faiss_index():
#     data = pd.read_pickle(os.path.join("data", "recipe_qa_augmented_dataset.pickle"))
#     recipe_questions, recipe_answers = [], []
#     recipe_title_encodings, recipe_context_encodings = [], []
#     titleids2text, contextids2text, questionids2text = OrderedDict(), OrderedDict(), OrderedDict()
#     recipeids2qa, questionids2answers = OrderedDict(), OrderedDict()
#     for i, recipe_item in enumerate(data):
#         recipe_context = recipe_item["context"]
#         contextids2text[i] = recipe_context
#         recipe_context_encoding = similarity_model.encode(recipe_context)
#         recipe_context_encoding = recipe_context_encoding.reshape((1, recipe_context_encoding.shape[0]))
#         recipe_context_encodings.append(recipe_context_encoding)
#
#         recipe_title = recipe_context.split("\n")[0].split(":")[1].strip().lower()
#         titleids2text[i] = recipe_title
#         recipe_title_encoding = similarity_model.encode(recipe_title)
#         recipe_title_encoding = recipe_title_encoding.reshape((1, recipe_title_encoding.shape[0]))
#         recipe_title_encodings.append(recipe_title_encoding)
#
#         qa_list = recipe_item["qa"]
#         recipeids2qa[i] = qa_list
#         for qa in qa_list:
#             recipe_questions.append(qa["question"])
#             recipe_answers.append(qa["answer"])
#     for i, (question, answer) in enumerate(zip(recipe_questions, recipe_answers)):
#         questionids2text[i] = question
#         questionids2answers[i] = answer
#     recipe_question_encodings = similarity_model.encode(recipe_questions)
#
#     recipe_title_index = create_vectors_index(np.asarray(list(titleids2text.keys())), 384, recipe_title_encodings,
#                                               stack_embedding_arrays=True)
#     faiss.write_index(recipe_title_index, os.path.join("data", "output", "faiss-index", "recipe_titles.index"))
#     with open(os.path.join("data", "output", "faiss-index", "titleids2text.pickle"), 'wb') as handle:
#         pickle.dump(titleids2text, handle, protocol=pickle.HIGHEST_PROTOCOL)
#
#     recipe_context_index = create_vectors_index(np.asarray(list(contextids2text.keys())), 384, recipe_context_encodings,
#                                                 stack_embedding_arrays=True)
#     faiss.write_index(recipe_context_index, os.path.join("data", "output", "faiss-index", "recipe_contexts.index"))
#     with open(os.path.join("data", "output", "faiss-index", "contextids2text.pickle"), 'wb') as handle:
#         pickle.dump(contextids2text, handle, protocol=pickle.HIGHEST_PROTOCOL)
#
#     with open(os.path.join("data", "output", "faiss-index", "recipeids2qa.pickle"), 'wb') as handle:
#         pickle.dump(recipeids2qa, handle, protocol=pickle.HIGHEST_PROTOCOL)
#
#     recipe_question_index = create_vectors_index(np.asarray(list(questionids2text.keys())), 384,
#                                                  recipe_question_encodings, stack_embedding_arrays=False)
#     faiss.write_index(recipe_question_index, os.path.join("data", "output", "faiss-index", "recipe_questions.index"))
#     with open(os.path.join("data", "output", "faiss-index", "questionids2text.pickle"), 'wb') as handle:
#         pickle.dump(questionids2text, handle, protocol=pickle.HIGHEST_PROTOCOL)
#
#     with open(os.path.join("data", "output", "faiss-index", "questionids2answers.pickle"), 'wb') as handle:
#         pickle.dump(questionids2answers, handle, protocol=pickle.HIGHEST_PROTOCOL)
#
#     print("Data Indexed by Recipe Titles, Recipe Contexts, All Questions...\n")
#     return
#
#
# def search_faiss_index(snippet, index, topn):
#     snippet_embedding = similarity_model.encode(snippet)
#     snippet_embedding = snippet_embedding.reshape((1, snippet_embedding.shape[0]))
#     faiss.normalize_L2(snippet_embedding)
#     D, I = index.search(snippet_embedding, topn)
#     return I[0], D[0]
#
#
# def read_index():
#     recipe_title_index = faiss.read_index(os.path.join("data", "output", "faiss-index", "recipe_titles.index"))
#     with open(os.path.join("data", "output", "faiss-index", "titleids2text.pickle"), 'rb') as handle:
#         titleids2text = pickle.load(handle)
#
#     recipe_context_index = faiss.read_index(os.path.join("data", "output", "faiss-index", "recipe_contexts.index"))
#     with open(os.path.join("data", "output", "faiss-index", "contextids2text.pickle"), 'rb') as handle:
#         contextids2text = pickle.load(handle)
#
#     with open(os.path.join("data", "output", "faiss-index", "recipeids2qa.pickle"), 'rb') as handle:
#         recipeids2qa = pickle.load(handle)
#
#     recipe_question_index = faiss.read_index(os.path.join("data", "output", "faiss-index", "recipe_questions.index"))
#     with open(os.path.join("data", "output", "faiss-index", "questionids2text.pickle"), 'rb') as handle:
#         questionids2text = pickle.load(handle)
#
#     with open(os.path.join("data", "output", "faiss-index", "questionids2answers.pickle"), 'rb') as handle:
#         questionids2answers = pickle.load(handle)
#
#     return recipe_title_index, recipe_context_index, recipe_question_index, titleids2text, contextids2text, recipeids2qa, \
#            questionids2text, questionids2answers
#
#
# def test():
#     TOPN = 10
#     # create_faiss_index()
#
#     sample_task = "How to make Apple Pie"
#
#     sample_uttr = '''
#         In large bowl, gently mix filling ingredients; spoon into crust-lined pie plate.
#     '''
#     # "Top with second crust.Wrap excess top crust under bottom crust edge, pressing edges together to seal; flute.Cut slits or shapes in several places in top crust."
#
#     recipe_title_index, recipe_context_index, recipe_question_index, titleids2text, contextids2text, recipeids2qa, \
#     questionids2text, questionids2answers = read_index()
#
#     I, D = search_faiss_index(snippet=sample_task, index=recipe_title_index, topn=1000)
#     questions = set()
#     answers = set()
#     recipe_titles = set()
#     for iter, index in enumerate(I):
#         if len(recipe_titles) > 0:
#             break
#         recipe_titles.add((titleids2text[index]))
#         recipeqa = recipeids2qa[index]
#         for qa in recipeqa:
#             questions.add((qa["question"], D[iter]))
#             answers.add(qa["answer"])
#     print(recipe_titles)
#     print(questions, answers)
#
#     I, D = search_faiss_index(snippet=sample_uttr, index=recipe_question_index, topn=1000)
#     questions = set()
#     answers = set()
#     for iter, index in enumerate(I):
#         if len(questions) > TOPN:
#             break
#         questions.add((questionids2text[index], D[iter]))
#         answers.add(questionids2answers[index])
#     print(questions, answers)
#
#     I, D = search_faiss_index(snippet=sample_uttr, index=recipe_context_index, topn=1000)
#     contexts = set()
#     for iter, index in enumerate(I):
#         if len(contexts) > TOPN:
#             break
#         contexts.add((contextids2text[index], D[iter]))
#     print(contexts)
#
#     I, D = search_faiss_index(snippet=sample_uttr, index=recipe_context_index, topn=1000)
#     qa = list()
#     for iter, index in enumerate(I):
#         if len(qa) > TOPN:
#             break
#         qa.append((recipeids2qa[index], D[iter]))
#     print(qa)
#
#
# test()
