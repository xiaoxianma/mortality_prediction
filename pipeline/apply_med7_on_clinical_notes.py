import pandas as pd
import spacy

med7 = spacy.load("en_core_med7_lg")

intermediate_data_path = "../data/intermediate"

preprocessed_df = pd.read_pickle(f"{intermediate_data_path}/preprocessed_notes.p")
preprocessed_df["ner"] = None
count = 0
preprocessed_index = {}
for i in preprocessed_df.itertuples():

    if count % 1000 == 0:
        print(count)

    count += 1
    ind = i.Index
    text = i.preprocessed_text

    all_pred = []
    for each_sent in text:
        try:
            doc = med7(each_sent)
            result = [(ent.text, ent.label_) for ent in doc.ents]
            if len(result) == 0:
                continue
            all_pred.append(result)
        except:
            print("error..")
            continue
    preprocessed_df.at[ind, "ner"] = all_pred


pd.to_pickle(preprocessed_df, f"{intermediate_data_path}/ner_df.p")
