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
All models results will be generated under its model directory as a `hdf5` file after running models. List result tables below.
#### Time series baseline model result (reproduced original paper baseline model)
| Task        | AUROC | AUPRC | F1    |
|-------------|-------|-------|-------|
| In-hospital | 86.95 | 54.89 | 44.51 |
| In-ICU      | 88.44 | 50.66 | 41.50 |
| LOS > 3     | 68.81 | 63.40 | 53.26 |
| LOS > 7     | 72.83 | 19.35 | 4.24  |

#### Multimodal with medical entities using GRU (reproduced original paper multimodal model)
| Task        | Embedding | AUROC | AUPRC | F1    |
|-------------|-----------|-------|-------|-------|
| In-hospital | Word2Vec  | 87.98 | 58.16 | 46.96 |
|             | FastText  | 86.09 | 54.47 | 45.50 |
|             | Concat    | 87.74 | 58.55 | 45.30 |
| In-ICU      | Word2Vec  | 87.98 | 58.16 | 46.96 |
|             | FastText  | 88.48 | 51.66 | 45.49 |
|             | Concat    | 89.29 | 52.74 | 40.83 |
| LOS > 3     | Word2Vec  | 70.71 | 65.26 | 56.51 |
|             | FastText  | 69.65 | 63.96 | 55.83 |
|             | Concat    | 70.52 | 64.49 | 56.46 |
| LOS > 7     | Word2Vec  | 72.79 | 21.73 | 3.30  |
|             | FastText  | 72.41 | 22.92 | 2.23  |
|             | Concat    | 72.65 | 20.90 | 4.36  |

#### Multimodal with medical entities using LSTM (additional experiments)
| Task        | Embedding | AUROC | AUPRC | F1    |
|-------------|-----------|-------|-------|-------|
| In-hospital | Word2Vec  | 88.60 | 57.86 | 44.31 |
|             | FastText  | 87.51 | 56.34 | 48.75 |
|             | Concat    | 88.15 | 58.31 | 51.51 |
| In-ICU      | Word2Vec  | 88.22 | 58.69 | 49.34 |
|             | FastText  | 88.76 | 53.53 | 48.67 |
|             | Concat    | 88.39 | 52.58 | 44.30 |
| LOS > 3     | Word2Vec  | 69.86 | 65.27 | 54.49 |
|             | FastText  | 69.28 | 63.82 | 55.98 |
|             | Concat    | 69.57 | 63.92 | 56.14 |
| LOS > 7     | Word2Vec  | 72.69 | 21.35 | 0.016 |
|             | FastText  | 71.74 | 20.86 | 0.027 |
|             | Concat    | 73.27 | 21.60 | 0.028 |
