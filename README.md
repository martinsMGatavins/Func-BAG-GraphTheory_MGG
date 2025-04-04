# BrainAgeGapsHCP

This is a repository for a project assessing the accuracy and utility of resting-state functional connectivity measures and operationalizations in predicting psychopathology, cognition, and age in a developing cohort (HCP-D).

## Data creation
1. Preprocessing of T1-weighted and fMRI images and extracting parcel-wise (in this case, Gordon333) measures of BOLD signal (including, a functional connectivity matrix, regional homogeneity, amplitude of low-frequency fluctuations): code not located here (R.J. has the code, using xcpD scripts)
2. Filtering of participants on the basis of scan quality: [Part0_Dataprep.R](code/Part0_Dataprep.R)
3. Calculating graph-theoretic measures (for optimal performance use on interactive node on cluster): [Part1_GraphCalculation.py](code/Part1_GraphCalculation.py)
4. Partition of train & test samples using anticlustering: [Part2_Anticlustering.R](code/Part2_Anticlustering.R)

### Data availability
Psychopathology & cognitive data available in WUSTL Box: master_1645x17_20220718.csv
Functional connectivity matrices & other rsfMRI data (in Gordon 333 parcellation) available in WUSTL Box:
* Functional connectivity matrices (nb! large, 1GB, 55278): func_fcon-parc-1644x55278_20220719.csv
* Regional homogeneity (ReHo, 333 parcels): func_reho-1644x333_20220718.csv
* Amplitude of low-frequency fluctuations (ALFF, 333 parcels): func_alff_1635x333_20220718.csv

Additionally, brain age predictions from BIDS 2022 is also available in WUSTL Box: brainage_allmodels.csv
Split data and most recent data versions available in "data" folder on WUSTL Box

## Machine-learning predictions
5. Run model fitting and predictions, extract difference between actual and predicted values: [Part3_Prediction](code/Part3_Prediction/) has code pertaining to running the model (Python script), *no bash script though*
6. *(NOT COMPLETED)* Calculate correlation, coefficients of determination (R-squared), MAEs, etc. (no corresponding code, at the moment)

## *Statistical analysis (INCOMPLETE)*
7. Comparison testing of models, 
8. Haufe transformation of values to assess model performance
9. Visualizations

## Previous code that may be of relevance
* Old analyses from BIDS with brain age gaps: [utility_vis_reg.R](code/utility_vis_reg.R)
* Code (python and bash scripts) from models run during BIDS 2022 (the results are analyzed in utility_vis_reg.R): [GTB](GTB)
* Older analyses including internalizing & externalizing symptoms (from BIDS 2022) & other older code: [oldcodelink](code/old_code/)
* MATLAB version of graph extraction code: [graph_extraction.m](code/graph_extraction.m)
* Environment set up for cluster extraction on the cluster: [graph_env.yml](code/graph_env.yml)
* Previous model attempts using alternative packages, such as _julearn_: [codelink_obsolete](code/obsolete_code/), [codelink_pred](code/obsolete_code/prediction_code)