# OpenML EHR datasets

## 43759:  Indian-Liver-Patient-Dataset
Description
Data Set Information This data set contains 416 liver patient records and 167 non liver patient records.The data set was collected from test samples in North East of Andhra Pradesh, India. 'is_patient' is a class label used to divide into groups(liver patient or not). This data set contains 441 male patient records and 142 female patient records. Any patient whose age exceeded 89 is listed as being of age "90". Attribute Information

age Age of the patient gender Gender of the patient tot_bilirubin Total Bilirubin direct_bilirubin Direct Bilirubin alkphos Alkaline Phosphotase sgpt Alamine Aminotransferase sgot Aspartate Aminotransferase tot_proteins Total Protiens albumin Albumin ag_ratio Albumin and Globulin Ratio is_patient Selector field used to split the data into two sets (labeled by the experts)

Acknowledgements The data set has been elicit from UCI Machine Learning Repository. My sincere thanks to them.

## 945: kidney
Description
Author: McGilchrist and Aisbett
Source: StatLib - 1999
Please cite:

Data on the recurrence times to infection, at the point of insertion of the catheter, for kidney patients using portable dialysis equipment. Catheters may be removed for reasons other than infection, in which case the observation is censored. Each patient has exactly 2 observations.

The data set has been used by several authors to illustrate random effects ("frailty") models for survival data. However, any non-zero estimate of the random effect is almost entirely due to one outlier, subject 21.

Variables: patient, time, status, age, sex (1=male, 2=female), disease type (0=Glomerulo Nephritis, 1=Acute Nephritis, 2=Polycystic Kidney Disease, 3=Other), author's estimate of the frailty

References:
McGilchrist and Aisbett, Biometrics 47, 461-66, 1991

## 46750: Dataset-Mental-Disorders

Description
About Dataset

A Collection of 120 Psychology Patients with 17 Essential Symptoms to Diagnose Mania Bipolar Disorder, Depressive Bipolar Disorder, Major Depressive Disorder, and Normal Individuals. The dataset contains the 17 essential symptoms psychiatrists use to diagnose the described disorders. The behavioral symptoms are considered the levels of patients Sadness, Exhaustness, Euphoric, Sleep disorder, Mood swings, Suicidal thoughts, Anorexia, Anxiety, Try-explaining, Nervous breakdown, Ignore & Move-on, Admitting mistakes, Overthinking, Aggressive response, Optimism, Sexual activity, and Concentration in a Comma Separated Value (CSV) format. The Normal category refer to the individuals using therapy time for specialized counseling, personal development, and life skill enrichments. While such individuals may also have minor mental problems, they differ from those suffering from Major Depressive Disorder and Bipolar Disorder.

https://www.kaggle.com/datasets/cid007/mental-disorder-classification

## 1480: ilpd

Description
Author: Bendi Venkata Ramana, M. Surendra Prasad Babu, N. B. Venkateswarlu
Source: UCI - 2012
Please cite: UCI

Indian Liver Patient Dataset
This data set contains 416 liver patient records and 167 non liver patient records.The data set was collected from north east of Andhra Pradesh, India. The class label divides the patients into 2 groups (liver patient or not). This data set contains 441 male patient records and 142 female patient records.

Attribute Information
V1. Age of the patient. Any patient whose age exceeded 89 is listed as being of age "90".
V2. Gender of the patient
V3. Total Bilirubin
V4. Direct Bilirubin
V5. Alkphos Alkaline Phosphatase
V6. Sgpt Alanine Aminotransferase
V7. Sgot Aspartate Aminotransferase
V8. Total Proteins
V9. Albumin
V10. A/G Ratio Albumin and Globulin Ratio

A feature indicating a train-test split has been removed.

Relevant Papers
Bendi Venkata Ramana, Prof. M. S. Prasad Babu and Prof. N. B. Venkateswarlu, A Critical Comparative Study of Liver Patients from USA and INDIA: An Exploratory Analysis, International Journal of Computer Science Issues, ISSN:1694-0784, May 2012.
Bendi Venkata Ramana, Prof. M. S. Prasad Babu and Prof. N. B. Venkateswarlu, A Critical Study of Selected Classification Algorithms for Liver Disease Diagnosis, International Journal of Database Management Systems (IJDMS), Vol.3, No.2, ISSN : 0975-5705, PP 101-114, May 2011.

## 46860: Support2

Description
This dataset comprises 9105 individual critically ill patients across 5 United States medical centers, accessioned throughout 1989-1991 and 1992-1994. Each row concerns hospitalized patient records who met the inclusion and exclusion criteria for nine disease categories: acute respiratory failure, chronic obstructive pulmonary disease, congestive heart failure, liver disease, coma, colon cancer, lung cancer, multiple organ system failure with malignancy, and multiple organ system failure with sepsis. The goal is to determine these patients' 2- and 6-month survival rates based on several physiologic, demographics, and disease severity information. It is an important problem because it addresses the growing national concern over patients' loss of control near the end of life. It enables earlier decisions and planning to reduce the frequency of a mechanical, painful, and prolonged dying process.

For what purpose was the dataset created?

To develop and validate a prognostic model that estimates survival over a 180-day period for seriously ill hospitalized adults (phase I of SUPPORT) and to compare this model's predictions with those of an existing prognostic system and with physicians' independent estimates (SUPPORT phase II).

Who funded the creation of the dataset?

Funded by the Robert Wood Johnson Foundation

What do the instances in this dataset represent?

The instances represent records of critically ill patients admitted to United States hospitals with advanced stages of serious illness.

Are there recommended data splits?

No recommendation, standard train-test split could be used. Can use three-way holdout split (i.e., train-validation-test) when doing model selection.

Does the dataset contain data that might be considered sensitive in any way?

Yes. There is information about race, gender, income, and education level.

Was there any data preprocessing performed?

No. Due to the high percentage of missing values, there are a couple of recommended imputation values: According to the HBiostat Repository (https://hbiostat.org/data/repo/supportdesc, Professor Frank Harrell) the following default values have been found to be useful in imputing missing baseline physiologic data: Baseline Variable Normal Fill-in Value

Serum albumin (alb) 3.5 PaO2/FiO2 ratio (pafi) 333.3 Bilirubin (bili) 1.01 Creatinine (crea) 1.01 bun 6.51 White blood count (wblc) 9 (thousands) Urine output (urine) 2502 There are 159 patients surviving 2 months for whom there were no patient or surrogate interviews. These patients have missing sfdm2. Additional Information

Data sources are medical records, personal interviews, and the National Death Index (NDI). For each patient administrative records data, clinical data and survey data were collected. The objective of the SUPPORT project was to improve decision-making in order to address the growing national concern over the loss of control that patients have near the end of life and to reduce the frequency of a mechanical, painful, and prolonged process of dying. SUPPORT comprised a two-year prospective observational study (Phase I) followed by a two-year controlled clinical trial (Phase II). Phase I of SUPPORT collected data from patients accessioned during 1989-1991 to characterize the care, treatment preferences, and patterns of decision-making among critically ill patients. It also served as a preliminary step for devising an intervention strategy for improving critically-ill patients' care and for the construction of statistical models for predicting patient prognosis and functional status. An intervention was implemented in Phase II of SUPPORT, which accessioned patients during 1992-1994. The Phase II intervention provided physicians with accurate predictive information on future functional ability, survival probability to six months, and patients' preferences for end-of-life care. Additionally, a skilled nurse was provided as part of the intervention to elicit patient preferences, provide prognoses, enhance understanding, enable palliative care, and facilitate advance planning. The intervention was expected to increase communication, resulting in earlier decisions to have orders against resuscitation, decrease time that patients spent in undesirable states (e.g., in the Intensive Care Unit, on a ventilator, and in a coma), increase physician understanding of patients' preferences for care, decrease patient pain, and decrease hospital resource use. Data collection in both phases of SUPPORT consisted of questionnaires administered to patients, their surrogates, and physicians, plus chart reviews for abstracting clinical, treatment, and decision information. Phase II also collected information regarding the implementation of the intervention, such as patient-specific logs maintained by nurses assigned to patients as part of the intervention. SUPPORT patients were followed for six months after inclusion in the study. Those who did not die within six months or were lost to follow-up were matched against the National Death Index to identify deaths through 1997. Patients who did not die within one year or were lost to follow-up were matched against the National Death Index to identify deaths through 1997. All patients in five United States medical centers who met inclusion and exclusion criteria for nine disease categories: acute respiratory failure, chronic obstructive pulmonary disease, congestive heart failure, liver disease, coma, colon cancer, lung cancer, multiple organ system failure with malignancy, and multiple organ system failure with sepsis. SUPPORT is a combination of patients from 2 studies, each of which lasted 2 years. The first phase concerns 4,301 patients, whereas the second phase concerns 4,804 patients. Time wise, these studies were accessioned in 1989 (June 12) through 1991 (June 11) for phase I and in 1992 (January 7) through 1994 (January 24).

## 43903: Diabetes-130-Hospitals_(Fairlearn)  

The "Diabetes 130-Hospitals" dataset represents 10 years of clinical care at 130 U.S. hospitals and delivery networks, collected from 1999 to 2008. Each record represents the hospital admission record for a patient diagnosed with diabetes whose stay lasted between one to fourteen days. The features describing each encounter include demographics, diagnoses, diabetic medications, number of visits in the year preceding the encounter, and payer information, as well as whether the patient was readmitted after release, and whether the readmission occurred within 30 days of the release.

The original "Diabetes 130-Hospitals" dataset was collected by Beata Strack, Jonathan P. DeShazo, Chris Gennings, Juan L. Olmo, Sebastian Ventura, Krzysztof J. Cios, and John N. Clore in 2014.

This version of the dataset was derived by the Fairlearn team for the SciPy 2021 tutorial "Fairness in AI Systems: From social context to practice using Fairlearn". In this version, the target variable "readmitted" is binarized into whether the patient was re-admitted within thirty days. The full dataset pre-processing script can be found on GitHub: https://github.com/fairlearn/talks/blob/main/2021_scipy_tutorial/preprocess.py

## 46170: patient 
## 46853  | sepsis_survival_minimal_clinical_records

Description
Dataset Information What do the instances in this dataset represent?

For the primary cohort, they represent records of patients affected by sepsis potential preconditions (ante Sepsis-3 definition); for the study cohort, they represent only the patients' admissions defined by the novel Sepsis-3 definition.

Are there recommended data splits?

No recommendation, standard train-test split could be used. Can use three-way holdout split (i.e., training, validation/development, testing) when doing model selection.

Does the dataset contain data that might be considered sensitive in any way?

Yes. It contains information about the gender and age of the patient.

Was there any data preprocessing performed?

All the categorical variables have been encoded (so no preprocessing is necessary).

Additional Information

Primary cohort from Norway:

4 features for 110,204 patient admissions
file: 's41598-020-73558-3_sepsis_survival_primary_cohort.csv'
Study cohort (a subset of the primary cohort) from Norway:

4 features for 19,051 patient admissions
file: 's41598-020-73558-3_sepsis_survival_study_cohort.csv'
Validation cohort from South Korea:

4 features for 137 patients
file: 's41598-020-73558-3_sepsis_survival_validation_cohort.csv'
The validation cohort from South Korea was used by Chicco and Jurman (2020) as an external validation cohort to confirm the generalizability of their proposed approach.

Has Missing Values?

No

## 43550  | COVID-19-Hospitals-Treatment-Plan

Description
Context The COVID-19 pandemic has placed an unprecedented strain on health systems, with rapidly increasing demand for healthcare in hospitals and intensive care units (ICUs) worldwide. As the pandemic escalates, determining the resulting needs for healthcare resources (beds, staff, equipment) has become a key priority for many countries. Projecting future demand requires estimates of how long patients with COVID-19 need different levels of hospital care. While healthcare management has various use cases for using data science, patient length of stay is one critical parameter to observe and predict if one wants to improve the efficiency of the healthcare management in a hospital. This parameter helps hospitals to identify patients of high LOS risk (patients who will stay longer) at the time of admission. Once identified, patients with high LOS risk can have their treatment plan optimized to miminize LOS and lower the chance of staff/visitor infection. Also, prior knowledge of LOS can aid in logistics such as room and bed allocation planning. The problem is to manage the functioning of Hospitals in a professional and optimal manner.

Content The task is to accurately predict the Length of Stay for each patient on case by case basis so that the Hospitals can use this information for optimal resource allocation and better functioning. The length of stay is divided into 11 different classes ranging from 0-10 days to more than 100 days. Data : host_train.csv File containing features related to patient, hospital and Length of stay on case basis For each record in the dataset the following is provided::

case_id
Hospital
Hospital_type
Hospital_city
Hospital_region
Available-Extra-Rooms-in-Hospital Number of Extra rooms available in the Hospital Department Department overlooking the case ['radiotherapy' 'anesthesia' 'gynecology' 'TB Chest disease' 'surgery'] Ward_Type ['R' 'S' 'Q' 'P' 'T' 'U'] Ward_Facility ['F' 'E' 'D' 'B' 'A' 'C'] Bed_Grade Condition of Bed in the Ward patientid
CityCodePatient City Code for the patient Type of Admission Admission Type registered by the Hospital ['Emergency' 'Trauma' 'Urgent'] Illness_Severity Severity of the illness recorded at the time of admission ['Extreme' 'Moderate' 'Minor'] Patient_Visitors
Age Age category ['51-60' '71-80' '31-40' '41-50' '81-90' '61-70' '21-30' '11-20' '0-10' '91-100'] Admission_Deposit Deposit at the Admission Time Stay_Days Stay Days by the patient (target) ['0-10' '41-50' '31-40' '11-20' '51-60' '21-30' '71-80' 'More than 100 Days' '81-90' '61-70' '91-100']

Starter Kernels

EDA and Random Forest Benchmark

Inspiration

Predict the Length of Stay for each patient Interpret best model(s) and mine influence factors in LOS risk

More References

COVID-19 length of hospital stay A retrospective cohort study in a Fangcang shelter hospital


## 46848  | differentiated_thyroid_cancer_recurrence

Description
Description: The dataset is a comprehensive collection of clinical data relating to thyroid diseases. With attributes capturing a wide range of information from patient demographics (age, gender) to specific clinical findings (smoking history, radiotherapy history, thyroid function, physical examination findings), it provides a detailed overview of patients diagnosed with various forms of thyroid conditions. The dataset encapsulates aspects such as the presence of adenopathy, pathology findings, focality of the disease, and risk categorization. Further, it delves into the TNM classification system, providing insights into the size and extent of tumors (T), presence of cancer in nearby lymph nodes (N), and metastasis (M), thereby contributing to the staging of the disease. The clinical response to treatment and recurrence status is also recorded, offering valuable data for outcomes analysis.

Attribute Description:

Age: Numeric, represents the age of the patient. Gender: Categorical, 'M' for male, 'F' for female. Smoking: Binary, 'Yes' if the patient has a history of smoking, 'No' otherwise. Hx Smoking: Binary, indicating a historical record of smoking. Hx Radiotherapy: Binary, indicates if the patient has undergone radiotherapy. Thyroid Function: Categorical, reports the thyroid's functional state. Physical Examination: Text, describes findings from physical examination. Adenopathy: Binary, 'Yes' if adenopathy is present, 'No' otherwise. Pathology: Categorical, type of thyroid pathology diagnosed. Focality: Categorical, 'Multi-Focal' or 'Uni-Focal' disease spread. Risk: Categorical, assessed risk level ('Low', 'Intermediate', 'High'). T, N, M: Staging parameters as per the TNM classification. Stage: Categorical, stage of the disease. Response: Categorical, patient's response to treatment. Recurred: Binary, 'Yes' if the disease has recurred, 'No' otherwise. Use Case: This dataset is instrumental for researchers and clinicians focusing on thyroid diseases. Its detailed attributes facilitate analyses on the relationship between demographic factors, lifestyle choices (such as smoking), clinical findings, and treatment outcomes. Furthermore, it can serve as a valuable resource for predictive modeling of disease progression, recurrence, and response to therapy. Machine learning applications can leverage this dataset for developing algorithms that predict patient outcomes, guide treatment plans, and assess risk factors for disease recurrence or poor treatment response.

For what purpose was the dataset created?

It was a part of research in the field of AI and Medicine

Who funded the creation of the dataset?

No funding was provided.

What do the instances in this dataset represent?

Individual patients

Are there recommended data splits?

No

Does the dataset contain data that might be considered sensitive in any way?

No

Has Missing Values?

No


## 45950  | Heart_Failure_Prediction

Description
Description: This dataset, named "heart_failure_clinical_records.csv", consists of clinical records of patients with heart failure. It includes various attributes such as age, anaemia, creatinine phosphokinase levels, diabetes status, ejection fraction, high blood pressure presence, platelet count, serum creatinine levels, serum sodium levels, gender, smoking habits, follow-up time, and a binary indicator for death event.

Attribute Description:

age: age of the patient
anaemia: presence of anaemia (0: no, 1: yes)
creatinine_phosphokinase: level of creatinine phosphokinase in the blood
diabetes: presence of diabetes (0: no, 1: yes)
ejection_fraction: percentage of blood leaving the heart at each contraction
high_blood_pressure: presence of high blood pressure (0: no, 1: yes)
platelets: platelet count in the blood
serum_creatinine: level of serum creatinine in the blood
serum_sodium: level of serum sodium in the blood
sex: gender of the patient (0: female, 1: male)
smoking: smoking status of the patient (0: no, 1: yes)
time: follow-up time
DEATH_EVENT: indicator of death occurence during the follow-up period (0: no, 1: yes)
Use Case: This dataset can be used for analyzing the relationship between various clinical attributes and the occurrence of death events in patients with heart failure. It can help in predicting the risk factors associated with heart failure mortality.


## 46612  | heart_failure_clinical_records

Description
This dataset contains the medical records of 299 patients who had heart failure, collected during their follow-up period, where each patient profile has 13 clinical features.

Additional Information

A detailed description of the dataset can be found in the Dataset section of the following paper:

Davide Chicco, Giuseppe Jurman: "Machine learning can predict survival of patients with heart failure from serum creatinine and ejection fraction alone". BMC Medical Informatics and Decision Making 20, 16 (2020). https://doi.org/10.1186/s12911-020-1023-5

Thirteen (13) clinical features:

age: age of the patient (years)
anaemia: decrease of red blood cells or hemoglobin (boolean)
creatinine phosphokinase (CPK): level of the CPK enzyme in the blood (mcg/L)
diabetes: if the patient has diabetes (boolean)
ejection fraction: percentage of blood leaving the heart at each contraction (percentage)
high blood pressure: if the patient has hypertension (boolean)
platelets: platelets in the blood (kiloplatelets/mL)
sex: woman or man (binary)
serum creatinine: level of serum creatinine in the blood (mg/dL)
serum sodium: level of serum sodium in the blood (mEq/L)
smoking: if the patient smokes or not (boolean)
time: follow-up period (days)
[target] death event: if the patient died during the follow-up period (boolean)
For more information, please check Table 1, Table 2, and Table 3 of the following paper:

Davide Chicco, Giuseppe Jurman: "Machine learning can predict survival of patients with heart failure from serum creatinine and ejection fraction alone". BMC Medical Informatics and Decision Making 20, 16 (2020). https://doi.org/10.1186/s12911-020-1023-5

For this dataset, the target feature is 'platelets'. It is a regression problem.


## 42177  | echocardiogram-uci
Description
Title: Echocardiogram Data

Source Information: -- Donor: Steven Salzberg (salzberg@cs.jhu.edu) -- Collector: -- Dr. Evlin Kinney -- The Reed Institute -- P.O. Box 402603 -- Maimi, FL 33140-0603 -- Date Received: 28 February 1989

Past Usage: -- 1. Salzberg, S. (1988). Exemplar-based learning: Theory and implementation (Technical Report TR-10-88). Harvard University, Center for Research in Computing Technology, Aiken Computation Laboratory (33 Oxford Street; Cambridge, MA 02138). -- Steve applied his EACH program to predict survival (i.e., life or death), did not use the wall-motion attribute, and recorded 87 correct and 29 incorrect in an incremental application to this database. He also showed that, by tuning EACH to this domain, EACH was able to derive (non-incrementally) a set of 28 hyper-rectangles that could perfectly classify 119 instances. -- 2. Kan, G., Visser, C., Kooler, J., & Dunning, A. (1986). Short and long term predictive value of wall motion score in acute myocardial infarction. British Heart Journal, 56, 422-427. -- They predicted the same variable (whether patients will live one year after a heart attack) using a different set of 345 instances. Their statistical test recorded a 61% accuracy in predicting that a patient will die (post-hoc fit). -- 3. Elvin Kinney (in communication with Steven Salzberg) reported that a Cox regression application recorded a 60% accuracy in predicting that a patient will die.

Relevant Information: -- All the patients suffered heart attacks at some point in the past. Some are still alive and some are not. The survival and still-alive variables, when taken together, indicate whether a patient survived for at least one year following the heart attack.

The problem addressed by past researchers was to predict from the other variables whether or not the patient will survive at least one year. The most difficult part of this problem is correctly predicting that the patient will NOT survive. (Part of the difficulty seems to be the size of the data set.)

Number of Instances: 132

Number of Attributes: 13 (all numeric-valued)

Attribute Information:

survival -- the number of months patient survived (has survived, if patient is still alive). Because all the patients had their heart attacks at different times, it is possible that some patients have survived less than one year but they are still alive. Check the second variable to confirm this. Such patients cannot be used for the prediction task mentioned above.
still-alive -- a binary variable. 0=dead at end of survival period, 1 means still alive
age-at-heart-attack -- age in years when heart attack occurred
pericardial-effusion -- binary. Pericardial effusion is fluid around the heart. 0=no fluid, 1=fluid
fractional-shortening -- a measure of contracility around the heart lower numbers are increasingly abnormal
epss -- E-point septal separation, another measure of contractility.
Larger numbers are increasingly abnormal.
lvdd -- left ventricular end-diastolic dimension. This is a measure of the size of the heart at end-diastole. Large hearts tend to be sick hearts.
wall-motion-score -- a measure of how the segments of the left ventricle are moving
wall-motion-index -- equals wall-motion-score divided by number of segments seen. Usually 12-13 segments are seen in an echocardiogram. Use this variable INSTEAD of the wall motion score.
mult -- a derivate var which can be ignored
name -- the name of the patient (I have replaced them with "name")
group -- meaningless, ignore it
alive-at-1 -- Boolean-valued. Derived from the first two attributes. 0 means patient was either dead after 1 year or had been followed for less than 1 year. 1 means patient was alive at 1 year.


## 222    | echoMonths

Description
Author:
Source: Unknown -
Please cite:

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

Survival treated as the class attribute

As used by Kilpatrick, D. & Cameron-Jones, M. (1998). Numeric prediction using instance-based learning with encoding length selection. In Progress in Connectionist-Based Information Systems. Singapore: Springer-Verlag.

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

Title: Echocardiogram Data

Source Information: -- Donor: Steven Salzberg (salzberg@cs.jhu.edu) -- Collector: -- Dr. Evlin Kinney -- The Reed Institute -- P.O. Box 402603 -- Maimi, FL 33140-0603 -- Date Received: 28 February 1989

Past Usage: -- 1. Salzberg, S. (1988). Exemplar-based learning: Theory and implementation (Technical Report TR-10-88). Harvard University, Center for Research in Computing Technology, Aiken Computation Laboratory (33 Oxford Street; Cambridge, MA 02138). -- Steve applied his EACH program to predict survival (i.e., life or death), did not use the wall-motion attribute, and recorded 87 correct and 29 incorrect in an incremental application to this database. He also showed that, by tuning EACH to this domain, EACH was able to derive (non-incrementally) a set of 28 hyper-rectangles that could perfectly classify 119 instances. -- 2. Kan, G., Visser, C., Kooler, J., & Dunning, A. (1986). Short and long term predictive value of wall motion score in acute myocardial infarction. British Heart Journal, 56, 422-427. -- They predicted the same variable (whether patients will live one year after a heart attack) using a different set of 345 instances. Their statistical test recorded a 61% accuracy in predicting that a patient will die (post-hoc fit). -- 3. Elvin Kinney (in communication with Steven Salzberg) reported that a Cox regression application recorded a 60% accuracy in predicting that a patient will die.

Relevant Information: -- All the patients suffered heart attacks at some point in the past. Some are still alive and some are not. The survival and still-alive variables, when taken together, indicate whether a patient survived for at least one year following the heart attack.

The problem addressed by past researchers was to predict from the other variables whether or not the patient will survive at least one year. The most difficult part of this problem is correctly predicting that the patient will NOT survive. (Part of the difficulty seems to be the size of the data set.)

Number of Instances: 132

Number of Attributes: 13 (all numeric-valued)

Attribute Information:

survival -- the number of months patient survived (has survived, if patient is still alive). Because all the patients had their heart attacks at different times, it is possible that some patients have survived less than one year but they are still alive. Check the second variable to confirm this. Such patients cannot be used for the prediction task mentioned above.
still-alive -- a binary variable. 0=dead at end of survival period, 1 means still alive
age-at-heart-attack -- age in years when heart attack occurred
pericardial-effusion -- binary. Pericardial effusion is fluid around the heart. 0=no fluid, 1=fluid
fractional-shortening -- a measure of contracility around the heart lower numbers are increasingly abnormal
epss -- E-point septal separation, another measure of contractility.
Larger numbers are increasingly abnormal.
lvdd -- left ventricular end-diastolic dimension. This is a measure of the size of the heart at end-diastole. Large hearts tend to be sick hearts.
wall-motion-score -- a measure of how the segments of the left ventricle are moving
wall-motion-index -- equals wall-motion-score divided by number of segments seen. Usually 12-13 segments are seen in an echocardiogram. Use this variable INSTEAD of the wall motion score.
mult -- a derivate var which can be ignored
name -- the name of the patient (I have replaced them with "name")
group -- meaningless, ignore it
alive-at-1 -- Boolean-valued. Derived from the first two attributes. 0 means patient was either dead after 1 year or had been followed for less than 1 year. 1 means patient was alive at 1 year.


## 46846  | glioma_grading_clinical_and_mutation_features

Description
For what purpose was the dataset created?

Gliomas are the most common primary tumors of the brain. They can be graded as LGG (Lower-Grade Glioma) or GBM (Glioblastoma Multiforme) depending on the histological/imaging criteria. Clinical and molecular/mutation factors are also very crucial for the grading process. Molecular tests are expensive to help accurately diagnose glioma patients.

In this dataset, the most frequently mutated 20 genes and 3 clinical features are considered from TCGA-LGG and TCGA-GBM brain glioma projects.

The prediction task is to determine whether a patient is LGG or GBM with a given clinical and molecular/mutation features. The main objective is to find the optimal subset of mutation genes and clinical features for the glioma grading process to improve performance and reduce costs.

Who funded the creation of the dataset?

The Cancer Genome Atlas (TCGA) Project - NCI

What do the instances in this dataset represent?

In this dataset, the instances represent the records of patients who have brain glioma. The dataset was constructed based on TCGA-LGG and TCGA-GBM brain glioma projects.

Each record is characterized by 20 molecular features (each of which can be mutated or not_mutated (wildtype) depending on the TCGA Case_ID) and 3 clinical features (concerning the demographics of the patient).

Are there recommended data splits?

No. We suggest 10-fold cross-validation for feature selection, classification etc.

Does the dataset contain data that might be considered sensitive in any way?

There is information about race, age, and gender of the patient.

Was there any data preprocessing performed?

Yes.

The original and preprocessed files differ in the following ways:

There are 23 instances in the original file where Gender, Age_at_diagnosis, or Race feature values are '--', or 'not reported'. These instances were filtered out in the preprocessed dataset.
Despite being present in the original dataset, we do not include the columns Project, Case_ID, and Primary_Diagnosis columns in the preprocessed dataset.
Age_at_diagnosis feature values were converted from string to continuous value by adding day information to the corresponding year information in the dataset as a floating-point number for the preprocessing stage.
All processed and unprocessed files also exist in this directory.

Below is a list of the additional columns of the original dataset file (and their corresponding description):

Project column represents corresponding TCGA-LGG or TCGA-GBM project names.
Case_ID column refers to the related project Case_ID information.
Primary_Diagnosis column provides information related to the type of primary diagnosis.
Glioma grade class information (0 = "LGG"; 1 = "GBM")


## 200    | pbc

Description
Author:
Source: Unknown - Date unknown
Please cite:

Binarized version of the original data set (see version 1). It converts the numeric target feature to a two-class nominal target feature by computing the mean and classifying all instances with a lower target value as positive ('P') and all others as negative ('N').


## 46076  | AIDS_Virus_Infection_Prediction

Description
Description: The AIDS_Classification_50000.csv dataset is a comprehensive resource specifically compiled for researchers and healthcare professionals focusing on the statistical analysis of AIDS (Acquired Immunodeficiency Syndrome). Composed of 50,000 instances, this dataset encapsulates a broad spectrum of clinical and demographic variables related to AIDS patients. Each record in the dataset holds data across 23 columns, indicating various patient attributes including treatment details, demographic information, clinical test results, and disease progression indicators.

Attribute Description:

time: Time since the baseline measurement, in days.
trt: Treatment code (0, 1, 2), where each number signifies a different treatment regimen.
age: Age of the patient in years.
wtkg: Weight of the patient in kilograms.
hemo: Presence of Hemophilia (0 = No, 1 = Yes).
homo: Homosexual behavior (0 = No, 1 = Yes).
drugs: Drug use (0 = No, 1 = Yes).
karnof: Karnofsky score indicating patient's functional impairment (scores range from 0 to 100).
oprior: Number of opportunistic infections prior to study.
z30: Presence of Z30 gene (0 = No, 1 = Yes).
preanti: Months before receiving antiretroviral therapy.
race: Race (0 = Non-white, 1 = White).
gender: Gender (0 = Female, 1 = Male).
str2: Stratification variable 2.
strat: Overall stratification.
symptom: Presence of specific AIDS-related symptoms (0 = No, 1 = Yes).
treat: Treatment response (0 = No, 1 = Yes).
offtrt: Off treatment (0 = No, 1 = Yes).
cd40: CD4 count at the baseline.
cd420: CD4 count at 20 weeks.
cd80: CD4 count at 8 weeks.
cd820: CD4 count at 20 weeks post the 8-week measurement.
infected: HIV infection status (0 = Negative, 1 = Positive).
Use Case: This dataset is designed to facilitate a range of scientific inquiries, including the evaluation of treatment efficacy, the identification of prognostic factors for disease progression, and the development of predictive models for patient outcomes. By encompassing a rich variety of data points, the AIDS_Classification_50000.csv supports detailed statistical analyses and machine learning efforts aimed at enhancing our understanding of AIDS. It can particularly serve in the optimization of treatment protocols and in the advancement of targeted therapies, ultimately contributing to improved patient care and management strategies in the field of AIDS research.

## 45022  | Diabetes130US

Description
Dataset used in the tabular data benchmark https://github.com/LeoGrin/tabular-benchmark, transformed in the same way. This dataset belongs to the "classification on numerical features" benchmark.

Original link: https://openml.org/d/4541

Original description:

Author: Attila Reiss, Department Augmented Vision, DFKI, Germany, "attila.reiss '@' dfki.de
Date: August 2012.
Source: UCI
Please cite: Beata Strack, Jonathan P. DeShazo, Chris Gennings, Juan L. Olmo, Sebastian Ventura, Krzysztof J. Cios, and John N. Clore, “ Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Patient Records,” BioMed Research International, vol. 2014, Article ID 781670, 11 pages, 2014.

This data has been prepared to analyze factors related to readmission as well as other outcomes pertaining to patients with diabetes.

Source
The data are submitted on behalf of the Center for Clinical and Translational Research, Virginia Commonwealth University, a recipient of NIH CTSA grant UL1 TR00058 and a recipient of the CERNER data. John Clore (jclore '@' vcu.edu), Krzysztof J. Cios (kcios '@' vcu.edu), Jon DeShazo (jpdeshazo '@' vcu.edu), and Beata Strack (strackb '@' vcu.edu). This data is a de-identified abstract of the Health Facts database (Cerner Corporation, Kansas City, MO).

Data Set Information
The dataset represents 10 years (1999-2008) of clinical care at 130 US hospitals and integrated delivery networks. It includes over 50 features representing patient and hospital outcomes. Information was extracted from the database for encounters that satisfied the following criteria:
(1) It is an inpatient encounter (a hospital admission).
(2) It is a diabetic encounter, that is, one during which any kind of diabetes was entered to the system as a diagnosis.
(3) The length of stay was at least 1 day and at most 14 days.
(4) Laboratory tests were performed during the encounter.
(5) Medications were administered during the encounter.
The data contains such attributes as patient number, race, gender, age, admission type, time in hospital, medical specialty of admitting physician, number of lab test performed, HbA1c test result, diagnosis, number of medication, diabetic medications, number of outpatient, inpatient, and emergency visits in the year before the hospitalization, etc.

Attribute Information
Detailed description of all the attributes is provided in Table 1 of the paper.

Relevant Papers
Beata Strack, Jonathan P. DeShazo, Chris Gennings, Juan L. Olmo, Sebastian Ventura, Krzysztof J. Cios, and John N. Clore, “Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Patient Records,” BioMed Research International, vol. 2014, Article ID 781670, 11 pages, 2014.

## 4531   | parkinsons-telemonitoring

Description
Author: Athanasios Tsanas (tsanasthanasis '@' gmail.com) and Max Little (littlem '@' physics.ox.ac.uk)
Source: UCI Please cite:

Source: The dataset was created by Athanasios Tsanas (tsanasthanasis '@' gmail.com) and Max Little (littlem '@' physics.ox.ac.uk) of the University of Oxford, in collaboration with 10 medical centers in the US and Intel Corporation who developed the telemonitoring device to record the speech signals. The original study used a range of linear and nonlinear regression methods to predict the clinician's Parkinson's disease symptom score on the UPDRS scale.

Data Set Information:

This dataset is composed of a range of biomedical voice measurements from 42 people with early-stage Parkinson's disease recruited to a six-month trial of a telemonitoring device for remote symptom progression monitoring. The recordings were automatically captured in the patient's homes.

Columns in the table contain subject number, subject age, subject gender, time interval from baseline recruitment date, motor UPDRS, total UPDRS, and 16 biomedical voice measures. Each row corresponds to one of 5,875 voice recording from these individuals. The main aim of the data is to predict the motor and total UPDRS scores ('motor_UPDRS' and 'total_UPDRS') from the 16 voice measures.

The data is in ASCII CSV format. The rows of the CSV file contain an instance corresponding to one voice recording. There are around 200 recordings per patient, the subject number of the patient is identified in the first column. For further information or to pass on comments, please contact Athanasios Tsanas (tsanasthanasis '@' gmail.com) or Max Little (littlem '@' physics.ox.ac.uk).

Further details are contained in the following reference -- if you use this dataset, please cite: Athanasios Tsanas, Max A. Little, Patrick E. McSharry, Lorraine O. Ramig (2009), 'Accurate telemonitoring of Parkinson’s disease progression by non-invasive speech tests', IEEE Transactions on Biomedical Engineering (to appear).

Further details about the biomedical voice measures can be found in: Max A. Little, Patrick E. McSharry, Eric J. Hunter, Lorraine O. Ramig (2009), 'Suitability of dysphonia measurements for telemonitoring of Parkinson's disease', IEEE Transactions on Biomedical Engineering, 56(4):1015-1022

Attribute Information:

subject# - Integer that uniquely identifies each subject age - Subject age sex - Subject gender '0' - male, '1' - female test_time - Time since recruitment into the trial. The integer part is the number of days since recruitment. motor_UPDRS - Clinician's motor UPDRS score, linearly interpolated total_UPDRS - Clinician's total UPDRS score, linearly interpolated Jitter(%),Jitter(Abs),Jitter:RAP,Jitter:PPQ5,Jitter:DDP - Several measures of variation in fundamental frequency Shimmer,Shimmer(dB),Shimmer:APQ3,Shimmer:APQ5,Shimmer:APQ11,Shimmer:DDA - Several measures of variation in amplitude NHR,HNR - Two measures of ratio of noise to tonal components in the voice RPDE - A nonlinear dynamical complexity measure DFA - Signal fractal scaling exponent PPE - A nonlinear measure of fundamental frequency variation

Relevant Papers:

Little MA, McSharry PE, Hunter EJ, Ramig LO (2009), 'Suitability of dysphonia measurements for telemonitoring of Parkinson's disease', IEEE Transactions on Biomedical Engineering, 56(4):1015-1022

Little MA, McSharry PE, Roberts SJ, Costello DAE, Moroz IM. 'Exploiting Nonlinear Recurrence and Fractal Scaling Properties for Voice Disorder Detection', BioMedical Engineering OnLine 2007, 6:23 (26 June 2007)

Citation Request:

If you use this dataset, please cite the following paper: A Tsanas, MA Little, PE McSharry, LO Ramig (2009) 'Accurate telemonitoring of Parkinson’s disease progression by non-invasive speech tests', IEEE Transactions on Biomedical Engineering (to appear).

## 45050  | Amazon_Electronics_Dataset

Description
Amazon Reviews data (data source) The repository has several datasets. For this case study, we are using the Electronics dataset

## 43040  | post-operative

Description
Author: Sharon Summers Source: UCI - 1993 Please cite: UCI

Post-Operative Patient Data Set

The classification task of this database is to determine where patients in a postoperative recovery area should be sent to next. Because hypothermia is a significant concern after surgery (Woolery, L. et. al. 1991), the attributes correspond roughly to body temperature measurements.

Attribute information
L-CORE (patient's internal temperature in C): high (> 37), mid (>= 36 and <= 37), low (< 36)
L-SURF (patient's surface temperature in C): high (> 36.5), mid (>= 36.5 and <= 35), low (< 35)
L-O2 (oxygen saturation in %): excellent (>= 98), good (>= 90 and < 98), fair (>= 80 and < 90), poor (< 80)
L-BP (last measurement of blood pressure): high (> 130/90), mid (<= 130/90 and >= 90/70), low (< 90/70)
SURF-STBL (stability of patient's surface temperature): stable, mod-stable, unstable
CORE-STBL (stability of patient's core temperature) stable, mod-stable, unstable
BP-STBL (stability of patient's blood pressure) stable, mod-stable, unstable
COMFORT (patient's perceived comfort at discharge, measured as an integer between 0 and 20)
decision ADM-DECS (discharge decision): I (patient sent to Intensive Care Unit), S (patient prepared to go home), A (patient sent to general hospital floor)

## 46843  | Cirrhosis_Patient_Survival_Prediction

Description
For what purpose was the dataset created?

Cirrhosis results from prolonged liver damage, leading to extensive scarring, often due to conditions like hepatitis or chronic alcohol consumption. The data provided is sourced from a Mayo Clinic study on primary biliary cirrhosis (PBC) of the liver carried out from 1974 to 1984.

Who funded the creation of the dataset?

Mayo Clinic

What do the instances in this dataset represent?

People

Does the dataset contain data that might be considered sensitive in any way?

Gender, Age

Was there any data preprocessing performed?

Drop all the rows where miss value (NA) were present in the Drug column
Impute missing values with mean results
One-hot encoding for all category attributes
Additional Information

During 1974 to 1984, 424 PBC patients referred to the Mayo Clinic qualified for the randomized placebo-controlled trial testing the drug D-penicillamine. Of these, the initial 312 patients took part in the trial and have mostly comprehensive data. The remaining 112 patients didn't join the clinical trial but agreed to record basic metrics and undergo survival tracking. Six of these patients were soon untraceable after their diagnosis, leaving data for 106 of these individuals in addition to the 312 who were part of the randomized trial.

ID: unique identifier
N_Days: number of days between registration and the earlier of death, transplantation, or study analysis time in July 1986
Status: status of the patient C (censored), CL (censored due to liver tx), or D (death)
Drug: type of drug D-penicillamine or placebo
Age: age in [days]
Sex: M (male) or F (female)
Ascites: presence of ascites N (No) or Y (Yes)
Hepatomegaly: presence of hepatomegaly N (No) or Y (Yes)
Spiders: presence of spiders N (No) or Y (Yes)
Edema: presence of edema N (no edema and no diuretic therapy for edema), S (edema present without diuretics, or edema resolved by diuretics), or Y (edema despite diuretic therapy)
Bilirubin: serum bilirubin in [mg/dl]
Cholesterol: serum cholesterol in [mg/dl]
Albumin: albumin in [gm/dl]
Copper: urine copper in [ug/day]
Alk_Phos: alkaline phosphatase in [U/liter]
SGOT: SGOT in [U/ml]
Triglycerides: triglicerides in [mg/dl]
Platelets: platelets per cubic [ml/1000]
Prothrombin: prothrombin time in seconds [s]
Stage: histologic stage of disease (1, 2, 3, or 4)
Class Labels

Status: status of the patient 0 = D (death), 1 = C (censored), 2 = CL (censored due to liver transplantation)


## 40683  | postoperative-patient-data

Description
Author:
Source: Unknown -
Please cite:

Title: Postoperative Patient Data

Source Information: -- Creators: Sharon Summers, School of Nursing, University of Kansas Medical Center, Kansas City, KS 66160 Linda Woolery, School of Nursing, University of Missouri, Columbia, MO 65211 -- Donor: Jerzy W. Grzymala-Busse (jerzy@cs.ukans.edu) (913)864-4488 -- Date: June 1993

Past Usage:

A. Budihardjo, J. Grzymala-Busse, L. Woolery (1991). Program LERS_LB 2.5 as a tool for knowledge acquisition in nursing, Proceedings of the 4th Int. Conference on Industrial & Engineering Applications of AI & Expert Systems, pp. 735-740.

L. Woolery, J. Grzymala-Busse, S. Summers, A. Budihardjo (1991). The use of machine learning program LERS_LB 2.5 in knowledge acquisition for expert system development in nursing. Computers in Nursing 9, pp. 227-234.

Relevant Information: The classification task of this database is to determine where patients in a postoperative recovery area should be sent to next.
Because hypothermia is a significant concern after surgery (Woolery, L. et. al. 1991), the attributes correspond roughly to body temperature measurements.

Results: -- LERS (LEM2): 48% accuracy

Number of Instances: 90

Number of Attributes: 9 including the decision (class attribute)

Attribute Information:

L-CORE (patient's internal temperature in C): high (> 37), mid (>= 36 and <= 37), low (< 36)
L-SURF (patient's surface temperature in C): high (> 36.5), mid (>= 36.5 and <= 35), low (< 35)
L-O2 (oxygen saturation in %): excellent (>= 98), good (>= 90 and < 98), fair (>= 80 and < 90), poor (< 80)
L-BP (last measurement of blood pressure): high (> 130/90), mid (<= 130/90 and >= 90/70), low (< 90/70)
SURF-STBL (stability of patient's surface temperature): stable, mod-stable, unstable
CORE-STBL (stability of patient's core temperature) stable, mod-stable, unstable
BP-STBL (stability of patient's blood pressure) stable, mod-stable, unstable
COMFORT (patient's perceived comfort at discharge, measured as an integer between 0 and 20)
decision ADM-DECS (discharge decision): I (patient sent to Intensive Care Unit), S (patient prepared to go home), A (patient sent to general hospital floor)
Missing Attribute Values: Attribute 8 has 3 missing values

Class Distribution: I (2) S (24) A (64)

Information about the dataset CLASSTYPE: nominal CLASSINDEX: last

## 43562  | Heart-Disease-patients

Description
Context: The leading cause of death in the developed world is heart disease. Therefore there needs to be work done to help prevent the risks of of having a heart attack or stroke. Content: Use this dataset to predict which patients are most likely to suffer from a heart disease in the near future using the features given. Acknowledgement: This data comes from the University of California Irvine's Machine Learning Repository at https://archive.ics.uci.edu/ml/datasets/Heart+Disease.

## 1097   | ICU

Description
Author:
Source: Unknown - Date unknown
Please cite:

Datasets of Data And Story Library, project illustrating use of basic statistic methods, converted to arff format by Hakan Kjellerstrand. Source: TunedIT: http://tunedit.org/repo/DASL

DASL file http://lib.stat.cmu.edu/DASL/Datafiles/ICU.html

Reference: Authorization: Contact authors Description: The data consist of 200 subjects from a larger study on the survival of patients following admission to an adult intensive care unit (ICU). The study used logistic regression to predict the probability of survival for these patients until their discharge from the hospital. The dependent variable is the binary variable Vital Status (STA). Nineteen possible predictor variables, both discrete and continuous, were also observed.

Number of cases: 200 Variable Names:

ID: ID number of the patient STA: Vital status (0 = Lived, 1 = Died) AGE: Patient's age in years SEX: Patient's sex (0 = Male, 1 = Female) RACE: Patient's race (1 = White, 2 = Black, 3 = Other) SER: Service at ICU admission (0 = Medical, 1 = Surgical) CAN: Is cancer part of the present problem? (0 = No, 1 = Yes) CRN: History of chronic renal failure (0 = No, 1 = Yes) INF:



## 43686  | chronic-kidney-disease

Description
Author: L.Jerlin Rubini Source: UCI - 2015 Please cite: UCI

Chronic_Kidney_Disease Data Set

This dataset can be used to predict the chronic kidney disease and it can be collected from the hospital nearly 2 months of period.

Attribute information
We use 24 + class = 25 ( 11 numeric ,14 nominal)

Age(numerical) age in years
Blood Pressure(numerical) bp in mm/Hg
Specific Gravity(nominal)
Albumin(nominal)
Sugar(nominal)
Red Blood Cells(nominal)
Pus Cell (nominal)
Pus Cell clumps(nominal)
Bacteria(nominal)
Blood Glucose Random(numerical) 11.Blood Urea(numerical)
Serum Creatinine(numerical)
Sodium(numerical)
Potassium(numerical)
Hemoglobin(numerical) 16.Packed Cell Volume(numerical)
White Blood Cell Count(numerical)
Red Blood Cell Count(numerical)
Hypertension(nominal)
Diabetes Mellitus(nominal)
Coronary Artery Disease(nominal)
Appetite(nominal)
Pedal Edema(nominal)
Anemia(nominal)
Class (nominal)


## 43797  | Hepatitis-C-Prediction-Dataset

Description
Context The data set contains laboratory values of blood donors and Hepatitis C patients and demographic values like age. The data was obtained from UCI Machine Learning Repository: https://archive.ics.uci.edu/ml/datasets/HCV+data Content All attributes except Category and Sex are numerical. Attributes 1 to 4 refer to the data of the patient:

X (Patient ID/No.)
Category (diagnosis) (values: '0=Blood Donor', '0s=suspect Blood Donor', '1=Hepatitis', '2=Fibrosis', '3=Cirrhosis')
Age (in years)
Sex (f,m) Attributes 5 to 14 refer to laboratory data:
ALB
ALP
ALT
AST
BIL
CHE
CHOL
CREA
GGT
PROT The target attribute for classification is Category (2): blood donors vs. Hepatitis C patients (including its progress ('just' Hepatitis C, Fibrosis, Cirrhosis). Acknowledgements Creators: Ralf Lichtinghagen, Frank Klawonn, Georg Hoffmann Donor: Ralf Lichtinghagen: Institute of Clinical Chemistry; Medical University Hannover (MHH); Hannover, Germany; lichtinghagen.ralf '' mh-hannover.de Donor: Frank Klawonn; Helmholtz Centre for Infection Research; Braunschweig, Germany; frank.klawonn '' helmholtz-hzi.de Donor: Georg Hoffmann; Trillium GmbH; Grafrath, Germany; georg.hoffmann '' trillium.de Relevant Papers Lichtinghagen R et al. J Hepatol 2013; 59: 236-42 Hoffmann G et al. Using machine learning techniques to generate laboratory diagnostic pathways a case study. J Lab Precis Med 2018; 3: 58-67 Other Datasets
Stroke Prediction Dataset: LINK


## 43687  | Breast-cancer-prediction

Context This dataset includes data from a random sample of 20,000 digital and 20,000 film-screen mammograms received by women age 60-89 years within the Breast Cancer Surveillance Consortium (BCSC) between January 2005 and December 2008. Some women contribute multiple examinations to the dataset. Data is useful in teaching about data analysis, epidemiological study designs, or statistical methods for binary outcomes or correlated data. Content The data set contains 39998 rows and 13 cols. Attributes are described as follows: Field Name **Type (Format) **Description

AgeAtTheTimeOf_Mammography number Patient's age in years at time of mammogram

Radiologists_Assessment string Radiologist's assessment based on the BI-RADS scale

IsBinaryIndicatorOfCancer_Diagnosis boolean Binary indicator of cancer diagnosis within one year of screening mammogram (false= No cancer diagnosis, true= Cancer diagnosis)

ComparisonMammogramFrom_Mammography string Comparison mammogram from prior mammography examination available

PatientsBIRADSBreastDensity string Patient's BI-RADS breast density as recorded at time of mammogram

FamilyHistoryOfBreastCancer string Family history of breast cancer in a first degree relative

CurrentUseOfHormoneTherapy string Current use of hormone therapy at time of mammogram

Binary_Indicator string Binary indicator of whether the woman had ever received a prior mammogram

HistoryOfBreast_Biopsy string Prior history of breast biopsy

IsFilmOrDigitalMammogram boolean Film or digital mammogram (true=Digital mammogram, false=Film mammogram)

Cancer_Type string Type of cancer