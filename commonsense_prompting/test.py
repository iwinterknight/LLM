import pickle
import pandas as pd
import collections

# check_index = 26230
# df = pd.read_pickle("latest_saved_df.pickle")
# check_counts = collections.Counter(df["prompt_response"][:check_index].isna())
# print(check_counts)

with open("latest_saved_data_dict.pickle", "rb") as f:
    sample_data = pickle.load(f)

sample_data