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
### Model Training
Go to `models` folder
1. Run `time_series_baseline.py` for basic RNN model
2. Run `multimodal_baseline.py` for RNN model + averaged medical entities
3. Run `proposed_model.py` for multimodal with Doc2Vec medical entities
### Results
All models' results will be saved under its model directory as a `hdf5` file.
#### Time series baseline model result
	Task	AUROC	AUPRC	F1
Original Paper	In-hospital	85.04	52.15	42.29
	In-ICU	86.32	46.51	36.30
	LOS > 3	67.40	60.17	53.36
	LOS > 7	70.54	16.35	2.33
Our experiment	In-hospital	86.95	54.89	44.51
	In-ICU	88.44	50.66	41.50
	LOS > 3	68.81	63.40	53.26
	LOS > 7	72.83	19.35	4.24
![image](https://user-images.githubusercontent.com/3086064/166121227-4117c778-a82f-4d7d-a1ae-2f49ff4c9156.png)
