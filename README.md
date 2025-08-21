# NCI Comorbidity Macro (Python Implementation)

This repository provides a **Python translation of the NCI Comorbidity SAS Macro**, commonly used in health services research to compute comorbidity measures from administrative claims data.  

It replicates the functionality of the SAS macro, including:  
- Mapping ICD-9 and ICD-10 diagnosis codes to comorbidity conditions.  
- Applying the **ruleout algorithm** (to reduce false positives from single outpatient claims).  
- Generating **patient-level comorbidity flags**.  
- Computing both **Charlson Comorbidity Index (CCI)** and **NCI Comorbidity Index** scores.  

---

## Features
- ✅ Reads SAS datasets directly with `pandas.read_sas`.  
- ✅ Handles ICD-9 and ICD-10 diagnosis codes.  
- ✅ Implements the NCI "ruleout" logic (medPAR, outpatient, and carrier claim checks).  
- ✅ Outputs patient-level comorbidity indicators with first observed dates.  
- ✅ Computes **Charlson** and **NCI weighted indices** for each patient.  
- ✅ Exports results to CSV (optional).  

---

* pandas
* numpy
