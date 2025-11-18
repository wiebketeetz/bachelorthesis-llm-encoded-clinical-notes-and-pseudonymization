
**single_notes_readmission_info.ipynb**
--> computes 30-day readmission for every note
- loads: 
	- MIMIC data
- creates: 
	- single_notes_readmission_info.csv


**subgroups.ipynb**
--> samples the 4100 notes used in this work
- loads:
	- MIMIC data,
	- single_notes_readmission_info.csv
	- demographics/demographics_random
- creates:
	- grouped
	- demographics
		- sample_demographics 
		- demographics_by_subgroup (from sampled_data_og)
		- random_subgroups (from sampled_data_og)
		- demographics_check (from sampled_data)
		- sampled_data_demographics.csv
	- **mimic_notes_pseudonymization**/mimic_readmission.csv (from sampled_data)


**note_preparation.ipynb**
-->handles labels and personal attributes in each notes (removal and swapping)
- loads:
	- demographics/sampled_data_demographics.csv
- creates:
	- prevalence


**results.ipynb**
--> computes and visualizes results  


**one_hot_prediction.ipynb**
--> "Logistic Regression on Patient Attributes" in the thesis
- loads:
	- demographics/sampled_data_demographics.csv
- AUROC and feature importance of readmission prediction with one-hot encoded age, race, gender (no embeddings)


(**mimic_attributes.ipynb**)
--> value_counts of full patients and admissions datasets 



**==grouped==**
	- unlabelled 4100 note-files (`<abbrev>_<hadm_id>.annotated.txt`)
**==annotated==**
	- labelled version of 4100 note-files (`<abbrev>_<hadm_id>.annotated.txt`)
--> annotation created locally by run_locally/annotate_phi.py

**==demographics==**
	**==demographics_by_subgroup==**
		- hadm_id, gender, actual_age, race (no name, no address) as .csv
		- for each subgroup 
	**==random_subgroups==**
		- hadm_id, gender, actual_age, race (no name, no address) as .csv
		- in 41 randomized groups of 100 rows each
	**==demographics_random==**
		-  hadm_id, gender, actual_age, race, name, address as .csv
		- created from random_subgroups by **run_locally/generate_demographics.py (locally)**
	**==demographics_check==**
		- hadm_id, gender, actual_age, race, name, address as .csv
		- for each subgroup
	**sampled_data_demographics.csv**
	- sampled_data with name and address
	**==sample_demographics==**
	- hadm_id, gender, actual_age, race (no name, no address) as .csv for all notes


**==mimic_notes_pseudonymization==**
	- evaluate readmission prediction model with original MIMIC notes

**==auroc_eval==**
	- find best size for test split in range `[0.05, 0.1, ..., 0.95]` 
	- trained on unlabelled, original notes

**==attribute_prediction==**
--> see "Attribute Inference from Note Embeddings" in the thesis

**==readmission_prediction==**
--> see "Readmission Prediction at Different Anonymization Levels "
--> see "Manipulated Personal Information" in the thesis


**==note_template==**
	**note_start_template_F.txt**
	**note_start_template_M.txt**


==**prevalence**== 
prevalence_by_group.csv (Readmission Rate in each gender/race/age group)
prevalence_by_hadm_id.csv (Readmission label for every note and what split it belongs to)



**==run_locally==**
--> gpt-oss scripts for labelling the notes and creating the names and addresses



