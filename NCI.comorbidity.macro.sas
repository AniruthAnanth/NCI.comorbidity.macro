/***********************************************************************************************************************************/
/* NCI.comorbidity.macro.sas                                                                                                       */
/* Created: 9/21/2021                                                                                                              */
/* Updated: 3/15/2022 to remove codes E12 and E14 from diabetes.                                                                   */
/***********************************************************************************************************************************/
/* If you encounter any problems with this macro, please email the SAS log file to seer-medicare@imsweb.com.                       */
/***********************************************************************************************************************************/
/* This SAS macro scans ICD-9 and ICD-10 diagnosis codes from hospital and physician claims for sixteen comorbid                   */
/* conditions and creates variables to indicate each condition found and the earliest date in which it was found.                  */
/* It also calculates 2 comorbidity indices (Charlson Score and NCI Index) for a patient with respect to cancer.                   */
/* Note, this version does not look for HCPCS codes or ICD procedure codes.                                                        */
/*                                                                                                                                 */
/* The macro is designed to look for claims reported in the year prior to diagnosis of cancer, however, the macro                  */
/* can accommodate any time period by inputting different start and end dates for the window of time for which                     */
/* claims are to be scanned for ICD-9 and ICD-10 definitions of comorbid conditions. NCI recommends calculating                    */
/* comorbidity indices using claims for the 12 calendar months prior to the month of diagnosis.                                    */
/*                                                                                                                                 */
/* The macro may be run using only hospital claims (MedPAR) or a combination of hospital and carrier/outpatient claims             */
/* (NCH and Outpatient). If carrier/outpatient claims are used, it has a switch in the call statement that may be used so that     */
/* it can remove claims that are considered to have unreliable diagnosis coding ("ruleouts"). Note, if the RULEOUT Algorithm       */
/* is used, then the input dataset should include claims for at least 30 days before and after the comorbidity "window".           */
/*                                                                                                                                 */
/* In order to use this macro:,                                                                                                    */
/* 1. Include this file in your SAS program                                                                                        */
/*    %include '/directory path/NCI.comorbidity.macro.sas';                                                                        */
/* 2. Create a clean file of claim records to send to the macro. You may include information from the claims files MedPAR,         */
/*    Outpatient and/or Carrier (NCH). All claim records of interest should be "set together" into a single SAS dataset.           */
/* 3. After setting up your dataset, call the macro COMORB. For example:                                                           */
/*    %COMORB(CLAIMS,PATIENT_ID,START_DATE,END_DATE,CLAIM_FROM_DATE,CLAIM_THRU_DATE,FILETYPE,ICD_DGNS_CD1-ICD_DGNS_CD25,Y,COMORB); */
/*    would send the dataset CLAIMS, with the person identifier PATIENT_ID to the macro. The dataset includes diagnosis codes      */
/*    ICD_DGNS_CD1-ICD_DGNS_CD25. Diagnosis codes are in ICD-9 or ICD-10 format. The file source of each claim record is found     */
/*    in the variable, FILETYPE (M=MedPAR, O=Outpatient, N=NCH). The start date of the claim is designated as CLAIM_FROM_DATE.     */
/*    The end date of the claim is designated as CLAIM_THRU_DATE. The CLAIM_THRU_DATE is used to determine if the diagnosis codes  */
/*    are ICD-9 (before 10/1/2015) or ICD-10 (on or after 10/1/2015). The 'Y' indicates that the RULEOUT algorithm will be used.   */
/* 4. The macro returns a SAS dataset (default name=Comorbidities) which contains 1 record for each person who had at least one    */
/*    claim within the window. The variables included in this output data set are the patient ID(s), Charlson Comorbidity Index,   */
/*    NCI Comorbidity Index, and each of the comorbid condition indicator flags for the time period of interest, as well as the    */
/*    date of the first claim for each condition. The data set is sorted by ID.                                                    */
/***********************************************************************************************************************************/
/* INFILE: Dataset name: a SAS dataset of Medicare claims that contains the following variables:                                   */
/* ID: Variable names: Unique ID for each patient. Can be a combination of more than one variable (i.e. patient_id sequence_num).  */
/* STARTDATE: Variable name: Date the comorbidity window starts, in SAS date format.                                               */
/* ENDDATE: Variable name: Date the comorbidity window ends, in SAS date format.                                                   */
/* CLAIMSTARTDATE: Variable name: Start date of the claim found on the claim file, in SAS date format. This can be created         */
/*   by using the input() function (e.g. CLAIMSTARTDATE = input(CLM_FROM_DT,yymmdd8.) or input(ADMSN_DT,yymmdd8.).                 */
/* CLAIMENDDATE: Variable name: End date of the claim found on the claim file, in SAS date format. This can be created             */
/*   by using the input() function (e.g. CLAIMENDDATE = input(CLM_THRU_DT,yymmdd8.) or input(DSCHRG_DT,yymmdd8.).                  */
/* CLAIMTYPE: Variable name: the source of the claim record ('M'=MedPAR, 'O'=Outpatient, 'N'=NCH). Note, do not use DME.           */
/* DXVARLIST: List of variable names: the diagnosis codes in ICD-9 or ICD-10 (e.g. ICD_DGNS_CD1-ICD_DGNS_CD25). If there are       */
/*   multiple variables, some of which cannot be included in a range, please list them using spaces to separate each single        */
/*   element or range (e.g. ADMTG_DGNS_CD PRNCPAL_DGNS_CD LINE_ICD_DGNS_CD ICD_DGNS_CD1-ICD_DGNS_CD25).                            */
/* RULEOUT: Flag: Set this to 'Y' if the "ruleout algorithm" should be invoked (further details below), otherwise set this to 'N'. */
/* OUTFILE: Dataset name: a SAS dataset with the comorbidity indicators, dates, and scores/indices determined by the macro.        */
/*   If an output dataset name is not specified, the default name is Comorbidities.                                                */
/***********************************************************************************************************************************/
/* RULEOUT Algorithm:                                                                                                              */
/*   The ruleout algorithm requires a more stringent criteria for which claims to include when identifying comorbidities.          */
/*   Any "stand-alone" outpatient (Outpatient or NCH) claims that are not confirmed by other claims are considered to be           */
/*   "ruleout" diagnoses and are excluded from consideration. A condition is confirmed if any claim for that condition is          */
/*   found in MedPAR or 2+ instances of claims for that condition are found more than 30 days apart in Outpatient or NCH.          */
/*   All other claims are excluded.                                                                                                */
/***********************************************************************************************************************************/

*** Begin macro COMORB;
%macro COMORB(INFILE,ID,STARTDATE,ENDDATE,CLAIMSTARTDATE,CLAIMENDDATE,CLAIMTYPE,DXVARLIST,RULEOUT,OUTFILE);

%LET conditions = acute_mi history_mi chf pvd cvd copd dementia paralysis diabetes diabetes_comp renal_disease mild_liver_disease liver_disease ulcers rheum_disease aids;

data claims;
set &INFILE(keep=&ID &STARTDATE &ENDDATE &CLAIMSTARTDATE &CLAIMENDDATE &CLAIMTYPE &DXVARLIST);

*** Select claim records in appropriate window;
*** Keep 30 days extra on both sides of the window for the RULEOUT Algorithm (for confirming diagnoses in the Outpatient and Carrier files);
%IF (&RULEOUT=Y OR &RULEOUT=1 OR &RULEOUT=R) %THEN %DO;
 where (&STARTDATE - 30) <= &CLAIMSTARTDATE <= (&ENDDATE + 30);
 inwindow = (&STARTDATE <= &CLAIMSTARTDATE <= &ENDDATE);
%END;
%ELSE %DO; 
 where &STARTDATE <= &CLAIMSTARTDATE <= &ENDDATE;
 inwindow = 1;
%END;

 *** Determine if the claim has ICD-9 or ICD-10 diagnosis codes, based on the claim end date;
 if &CLAIMENDDATE<mdy(10,1,2015) then ICDVRSN = 9;
 else if &CLAIMENDDATE>= mdy(10,1,2015) then ICDVRSN = 10;

 length &conditions 3.;
 array _conditions &conditions;
 do over _conditions; _conditions = 0; end;

 array dxcodes(*) $ &DXVARLIST;
 do i=1 to dim(dxcodes);
  if not missing(dxcodes(i)) then do;
   *** Convert any lowercase letters to uppercase;
   dxcode = upcase(dxcodes(i));

   *** Indentify comorbidities based on ICD-9 diagnosis codes in the claims;
   if ICDVRSN=9 then do;
    if dxcode in:('410') then acute_mi=1;  /* Acute myocardial infarction */
    if dxcode in:('412') then history_mi=1;  /* History myocardial infarction */
    if dxcode in:('39891','40201','40211','40291','40401','40403','40411','40413','40491','40493','428') or ('4254'<=:dxcode<=:'4259') then chf=1;  /* Congestive heart failure */
    /* Note, ICD-9 437.3 is not included here for PVD, even though it is listed in the Quan paper. */ 
    if dxcode in:('0930','440','441','4471','5571','5579','V434') or ('4431'<=:dxcode<=:'4439') then pvd=1;  /* Peripheral vascular disease */
    if dxcode in:('36234') or ('430'<=:dxcode<=:'438') then cvd=1;  /* Cerebrovascular disease */
    if dxcode in:('4168','4169','5064','5081','5088') or ('490'<=:dxcode<=:'505') then copd=1;  /* Chronic pulmonary disease */
    if dxcode in:('290','2941','3312') then dementia=1;  /* Dementia */
    if dxcode in:('3341','342','343','3449') or ('3440'<=:dxcode<=:'3446') then paralysis=1;  /* Hemiplegia or paraplegia */
    if dxcode in:('2500','2501','2502','2503','2508','2509') then diabetes=1;  /* Diabetes w/out chronic complications */
    if dxcode in:('2504','2505','2506','2507') then diabetes_comp=1;  /* Diabetes w/chronic complications */
    if dxcode in:('40301','40311','40391','40402','40403','40412','40413','40492','40493','582','585','586','5880','V420','V451','V56') or 
       ('5830'<=:dxcode<=:'5837') then renal_disease=1;  /* Renal disease */
    if dxcode in:('07022','07023','07032','07033','07044','07054','0706','0709','570','571','5733','5734','5738','5739','V427') then mild_liver_disease=1;  /* Mild liver disease */
    if ('4560'<=:dxcode<=:'4562') or ('5722'<=:dxcode<=:'5728') then liver_disease=1;  /* Moderate/Severe liver disease */
    if ('531'<=:dxcode<=:'534') then ulcers=1;  /* Peptic ulcer disease */
    if dxcode in:('4465','7148','725') or ('7100'<=:dxcode<=:'7104') or ('7140'<=:dxcode<=:'7142') then rheum_disease=1;  /* Rheumatic disease */
    if ('042'<=:dxcode<=:'044') then aids=1;  /* AIDS/HIV */
   end;

   *** Indentify comorbidities based on ICD-10 diagnosis codes in the claims;
   else if ICDVRSN=10 then do;
    if dxcode in:('I21','I22') then acute_mi=1;  /* Acute myocardial infarction */
    if dxcode in:('I252') then history_mi=1;  /* History myocardial infarction */
    if dxcode in:('I099','I110','I130','I132','I255','I420','I43','I50','P290') or ('I425'<=:dxcode<=:'I429') then chf=1;  /* Congestive heart failure */
    if dxcode in:('I70','I71','I731','I738','I739','I771','I790','I792','K551','K558','K559','Z958','Z959') then pvd=1;  /* Peripheral vascular disease */
    if dxcode in:('G45','G46','H340','I6') then cvd=1;  /* Cerebrovascular disease */
    if dxcode in:('I278','I279','J684','J701','J703') or ('J40'<=:dxcode<=:'J47') or ('J60'<=:dxcode<=:'J67') then copd=1;  /* Chronic pulmonary disease */
    if dxcode in:('F051','G30','G311') or ('F00'<=:dxcode<=:'F03') then dementia=1;  /* Dementia */
    if dxcode in:('G041','G114','G801','G802','G81','G82','G839') or ('G830'<=:dxcode<=:'G834') then paralysis=1;  /* Hemiplegia or paraplegia */
    if dxcode in:('E100','E101','E106','E108','E109','E110','E111','E116','E118','E119','E130','E131','E136','E138','E139') then diabetes=1;  /* Diabetes w/out chronic complications */
    if dxcode in:('E102','E103','E104','E105','E107','E112','E113','E114','E115','E117','E132','E133','E134','E135','E137') then diabetes_comp=1;  /* Diabetes w/chronic complications */
    /* Note, ICD-10 E12 and E14 are not included here for diabetes, even though they are listed in the Quan paper. */ 
    if dxcode in:('I120','I131','N18','N19','N250','Z940','Z992') or ('N032'<=:dxcode<=:'N037') or 
       ('N052'<=:dxcode<=:'N057') or ('Z490'<=:dxcode<=:'Z492') then renal_disease=1;  /* Renal disease */
    if dxcode in:('B18','K709','K717','K73','K74','K760','K768','K769','Z944') or ('K700'<=:dxcode<=:'K703') or 
       ('K713'<=:dxcode<=:'K715') or ('K762'<=:dxcode<=:'K764') then mild_liver_disease=1;  /* Mild liver disease */
    if dxcode in:('I850','I859','I864','I982','K704','K711','K721','K729','K765','K766','K767') then liver_disease=1;  /* Moderate/Severe liver disease */
    if ('K25'<=:dxcode<=:'K28') then ulcers=1;  /* Peptic ulcer disease */
    if dxcode in:('M05','M06','M315','M32','M33','M34','M351','M353','M360') then rheum_disease=1;  /* Rheumatic disease */
    if dxcode in:('B20','B21','B22','B24') then aids=1;  /* AIDS/HIV */
   end;
  end;
 end;

 drop i dxcode &DXVARLIST;
run;

*** Sort claims by ID and date of claim;
proc sort data=claims;
 by &ID &CLAIMSTARTDATE;
run;

*** Find the first claim date, first claim date in the window, and last claim date for each comorbidity;
data conditions;
 set claims;
 by &ID &CLAIMSTARTDATE;

 *** identifies the last (or only) ID variable;
 %LET IDvar = %SCAN(&ID,-1);

 *** identifies subjects that have ANY claims in the window;
 retain anyclaims;
 if first.&IDvar then anyclaims = 0;
 if inwindow then anyclaims = 1;

%DO i=1 %TO 16;
 %LET cond = %SCAN(&conditions,&i);

 if first.&IDvar then do; &cond._date=.; first_&cond.=.; last_&cond.=.; hosp_&cond.=0; end;
 retain &cond._date first_&cond. last_&cond. hosp_&cond.;

 if &cond then do;
  if inwindow and &cond._date=. then &cond._date = &CLAIMSTARTDATE;
  if first_&cond.=. then first_&cond. = &CLAIMSTARTDATE; 
  last_&cond. = &CLAIMSTARTDATE;
  if &CLAIMTYPE=:'M' then hosp_&cond. = 1;
  format &cond._date first_&cond. last_&cond. mmddyy10.;
 end;

%END;

 if last.&IDvar then output;
 drop &CLAIMTYPE &CLAIMSTARTDATE &CLAIMENDDATE ICDVRSN inwindow;
run;

*** set the default output filename;
%IF %LENGTH(&OUTFILE)=0 %THEN %LET OUTFILE=Comorbidities;

*** Check for comorbidities and create comorbidity indicators and Charlson & NCI Index scores;
data &OUTFILE;
 set conditions;

*** Rule out unconfirmed conditions if the RULEOUT indicator flag is set in macro call;
%DO i=1 %TO 16;
 %LET cond = %SCAN(&conditions,&i);
  %IF (&RULEOUT=Y OR &RULEOUT=1 OR &RULEOUT=R) %THEN %DO;
  if &cond._date>. then do; if hosp_&cond.=0 and (last_&cond. - first_&cond.)<=30 then &cond._date = .; end;
  %END;
  &cond. = (&cond._date>.);
%END;

 *** Calculate the Charlson Comorbidity Score for prior conditions;
 Charlson = 
 1*(acute_mi or history_mi) +
 1*(chf) +
 1*(pvd) +
 1*(cvd) +
 1*(copd) +
 1*(dementia) +
 2*(paralysis) +
 1*(diabetes and not diabetes_comp) +
 2*(diabetes_comp) +
 2*(renal_disease) +
 1*(mild_liver_disease and not liver_disease) +
 3*(liver_disease) +
 1*(ulcers) +
 1*(rheum_disease) +
 6*(aids);
 
 *** Calculate the NCI Comorbidity Index for prior conditions;
 NCI_index = 
 0.12624*(acute_mi) +
 0.07999*(history_mi) +
 0.64441*(chf) +
 0.26232*(pvd) +
 0.27868*(cvd) +
 0.52487*(copd) +
 0.72219*(dementia) +
 0.39882*(paralysis) +
 0.29408*(diabetes or diabetes_comp) +
 0.47010*(renal_disease) +
 0.73803*(mild_liver_disease or liver_disease) +
 0.07506*(ulcers) +
 0.21905*(rheum_disease) +
 0.58362*(aids);
 
 *** Only keep subjects with at least one claim in the window;
 if anyclaims;
 
 keep &ID &STARTDATE &ENDDATE acute_mi--aids acute_mi_date--aids_date Charlson NCI_index;
 drop first_: last_: hosp_:;
 
 label 
 Charlson                = 'Charlson comorbidity score'
 NCI_index               = 'NCI comorbidity index'
 acute_mi                = 'Acute Myocardial Infarction'
 history_mi              = 'History of Myocardial Infarction'
 chf                     = 'Congestive Heart Failure'
 pvd                     = 'Peripheral Vascular Disease'
 cvd                     = 'Cerebrovascular Disease'
 copd                    = 'Chronic Obstructive Pulmonary Disease'
 dementia                = 'Dementia'
 paralysis               = 'Hemiplegia or Paraplegia'
 diabetes                = 'Diabetes'
 diabetes_comp           = 'Diabetes with Complications'
 renal_disease           = 'Moderate-Severe Renal Disease'
 mild_liver_disease      = 'Mild Liver Disease'
 liver_disease           = 'Moderate-Severe Liver Disease'
 ulcers                  = 'Peptic Ulcer Disease'
 rheum_disease           = 'Rheumatologic Disease'
 aids                    = 'AIDS'
 acute_mi_date           = 'First indication of Acute Myocardial Infarction'
 history_mi_date         = 'First indication of History of Myocardial Infarction'
 chf_date                = 'First indication of Congestive Heart Failure'
 pvd_date                = 'First indication of Peripheral Vascular Disease'
 cvd_date                = 'First indication of Cerebrovascular Disease'
 copd_date               = 'First indication of Chronic Obstructive Pulmonary Disease'
 dementia_date           = 'First indication of Dementia'
 paralysis_date          = 'First indication of Hemiplegia or Paraplegia'
 diabetes_date           = 'First indication of Diabetes'
 diabetes_comp_date      = 'First indication of Diabetes with Complications'
 renal_disease_date      = 'First indication of Moderate-Severe Renal Disease'
 mild_liver_disease_date = 'First indication of Mild Liver Disease'
 liver_disease_date      = 'First indication of Moderate-Severe Liver Disease'
 ulcers_date             = 'First indication of Peptic Ulcer Disease'
 rheum_disease_date      = 'First indication of Rheumatologic Disease'
 aids_date               = 'First indication of AIDS'
 ;
run;

%mend; 
*** End macro COMORB;