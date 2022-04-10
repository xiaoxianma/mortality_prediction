import pandas as pd
import os
import numpy as np
import re
import preprocess

PREPROCESS = "../data/intermediate"


clinical_notes = pd.read_pickle(os.path.join(PREPROCESS, "sub_notes.p"))
clinical_notes.shape
sub_notes = clinical_notes[clinical_notes.SUBJECT_ID.notnull()]
sub_notes = sub_notes[sub_notes.CHARTTIME.notnull()]
sub_notes = sub_notes[sub_notes.TEXT.notnull()]
sub_notes = sub_notes[['SUBJECT_ID', 'HADM_ID_y', 'CHARTTIME', 'TEXT']]
sub_notes['preprocessed_text'] = None
for each_note in sub_notes.itertuples():
    text = each_note.TEXT
    sub_notes.at[each_note.Index, 'preprocessed_text'] = preprocess.getSentences(text)


# Save notes
pd.to_pickle(sub_notes, os.path.join(PREPROCESS, "preprocessed_notes.p"))

