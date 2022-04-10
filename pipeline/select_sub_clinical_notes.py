import pandas as pd
import os
import numpy as np

DATAPATH = "../data/raw/"
intermediate_data_path = "../data/intermediate/"

# Check the timeseries data and collect patient ID
lvl2_train_imputer = pd.read_pickle(
    os.path.join(intermediate_data_path, "lvl2_imputer_train.pkl")
)
lvl2_dev_imputer = pd.read_pickle(
    os.path.join(intermediate_data_path, "lvl2_imputer_dev.pkl")
)
lvl2_test_imputer = pd.read_pickle(
    os.path.join(intermediate_data_path, "lvl2_imputer_test.pkl")
)
Ys = pd.read_pickle(os.path.join(intermediate_data_path, "Ys.pkl"))

print(
    "Shape of train, dev, test {}, {}, {}.".format(
        (lvl2_train_imputer.shape), (lvl2_dev_imputer.shape), (lvl2_test_imputer.shape)
    )
)
print(402120 / 24, 57432 / 24, 114936 / 24)
patient_ids = []  # store all patient ids
for each_entry in Ys.index:
    patient_ids.append(each_entry[0])
print("Number of total patient {}".format(len(patient_ids)))


# select clinical notes
admission_df = pd.read_csv(os.path.join(DATAPATH, "ADMISSIONS.csv"))
noteevents_df = pd.read_csv(os.path.join(DATAPATH, "NOTEEVENTS.csv"))
icustays_df = pd.read_csv(os.path.join(DATAPATH, "ICUSTAYS.csv"))

noteevents_df.groupby(noteevents_df.CATEGORY).agg(["count"])

note_categories = noteevents_df.groupby(noteevents_df.CATEGORY).agg(["count"]).index

# selected_note_types = ['Nursing/other', 'Radiology', 'Nursing', 'ECG', 'Physician', 'Echo', 'Respiratory', 'Nutrition']
selected_note_types = []
for each_cat in list(note_categories):
    if each_cat != "Discharge summary":
        selected_note_types.append(each_cat)

# select based on note category
sub_notes = noteevents_df[noteevents_df.CATEGORY.isin(selected_note_types)]

# Drop no chart notes
missing_chardate_index = []
for each_note in sub_notes.itertuples():
    if isinstance(each_note.CHARTTIME, str):
        continue
    if np.isnan(each_note.CHARTTIME):
        missing_chardate_index.append(each_note.Index)
print("{} of notes does not charttime.".format(len(missing_chardate_index)))
print(sub_notes.shape)
sub_notes.drop(missing_chardate_index, inplace=True)

# Select based on patient id
sub_notes = sub_notes[sub_notes.SUBJECT_ID.isin(patient_ids)]

# Select based on time limit (24 hours)
MIMIC_EXTRACT_DATA = "../data/raw/all_hourly_data.h5"
stats = pd.read_hdf(MIMIC_EXTRACT_DATA, "patients")
TIMELIMIT = 1  # 1day
new_stats = stats.reset_index()
new_stats.rename(
    columns={"subject_id": "SUBJECT_ID", "hadm_id": "HADM_ID"}, inplace=True
)
df_adm_notes = pd.merge(
    sub_notes[["ROW_ID", "SUBJECT_ID", "HADM_ID", "CHARTTIME", "CATEGORY", "TEXT"]],
    new_stats[
        [
            "SUBJECT_ID",
            "HADM_ID",
            "icustay_id",
            "age",
            "admittime",
            "dischtime",
            "deathtime",
            "intime",
            "outtime",
            "los_icu",
            "mort_icu",
            "mort_hosp",
            "hospital_expire_flag",
            "hospstay_seq",
            "max_hours",
        ]
    ],
    on=["SUBJECT_ID"],
    how="left",
)
df_adm_notes["CHARTTIME"] = pd.to_datetime(df_adm_notes["CHARTTIME"])
df_less_n = df_adm_notes[
    (
        (df_adm_notes["CHARTTIME"] - df_adm_notes["intime"]).dt.total_seconds()
        / (24 * 60 * 60)
    )
    < TIMELIMIT
]

# Save clinical notes
pd.to_pickle(df_less_n, f"{intermediate_data_path}sub_notes.p")
