# Mortality Prediction
This repo implements all three methodologies that the paper ["Improving Clinical Outcome Predictions Using Convolution over Medical Entities with Multimodal Learning"](https://arxiv.org/abs/2011.12349) proposed. The three models are basic RNN model, multimodal with average medical entities, and multimodal with Doc2Vec medical entities.

# Citation
[Improving Clinical Outcome Predictions Using Convolution over Medical Entities with Multimodal Learning](https://arxiv.org/abs/2011.12349), 	arXiv:2011.12349

# Runbook
### Data Download Instruction
Follow `data/README.md` file
1. Download MIMIC-III data
2. Install med7 library
3. Download pre-trained embeddings
### Data Preparation/Cleaning
Go to `pipeline` folder
1. Run `extract_timeseries_features.py`
2. Run `select_sub_clinical_notes.py`
3. Run `preprocess_clinical_notes.py`
4. Run `apply_med7_on_clinical_notes.py`
5. Run `represent_entities_with_different_embedding.py`
6. Run `create_timeseries_data.py`
### Model Training & Running
Go to `models` folder
1. Run `time_series_baseline.py` for basic RNN model
2. Run `multimodal_baseline.py` for RNN model + averaged medical entities
3. Run `proposed_model.py` for multimodal with Doc2Vec medical entities
### Results
All models' results will be generated under its model directory as a `hdf5` file after running models.
#### Time series baseline model result
| Task        | AUROC | AUPRC | F1    |
|-------------|-------|-------|-------|
| In-hospital | 86.95 | 54.89 | 44.51 |
| In-ICU      | 88.44 | 50.66 | 41.50 |
| LOS > 3     | 68.81 | 63.40 | 53.26 |
| LOS > 7     | 72.83 | 19.35 | 4.24  |
