

## Table 1: Description of columns in Appointments_Data

File: `raw-data/appointments_data.csv`

| Attribute                       | Description                                                     | Remarks                                                                                                |
| :------------------------------ | :-------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------- |
| **Appointment_ID** | Unique identifier for the appointment.                          |                                                                                                        |
| **Patient_ID** | Unique identifier for the patient.                              |                                                                                                        |
| **Appointment_Date** | Date when the appointment occurred.                             | YYYY-MM-DD.                                                                                            |
| **Appointment_Time** | Time when the appointment occurred.                             |                                                                                                        |
| **Appointment_Status** | Status of the appointment                                       | Completed, Scheduled, or Cancelled                                                                     |
| **Patient_Name** | Name of the patient.                                            |                                                                                                        |
| **Patient_Date_Of_Birth** | Patient’s date of birth.                                        | YYYY-MM-DD.                                                                                            |
| **Patient_Gender** | Patient’s gender.                                               | M: Male, F: Female.                                                                                    |
| **Patient_Address** | Address of the patient.                                         |                                                                                                        |
| **Patient_Preferred_Hospital_ID** | Preferred hospital ID of the patient.                           |                                                                                                        |
| **Patient_Preferred_Pharmacy_ID** | Preferred pharmacy ID of the patient.                           |                                                                                                        |
| **Patient_Preferred_Insurance_Provider** | Preferred insurance provider of the patient.                  |                                                                                                        |
| **Emergency_Contact_Relationship** | Relationship to the patient of the emergency contact.             |                                                                                                        |
| **Emergency_Contact_Name** | Name of the emergency contact.                                  |                                                                                                        |
| **Emergency_Contact_Phone** | Phone number of the emergency contact.                          |                                                                                                        |
| **Emergency_Contact_Address** | Address of the emergency contact.                               |                                                                                                        |
| **Professional_Name** | Name of the professional handling the appointment.              |                                                                                                        |
| **Professional_Role** | Role of the professional                                        | e.g., Neurologist, Speech Therapist.                                                                   |
| **Department_ID** | Unique identifier for the department in which the professional works. |                                                                                                        |
| **Department_Name** | Name of the department.                                         | Based on the speciality.                                                                               |
| **Head_of_Department** | Name of the head of the department.                             |                                                                                                        |
| **Hospital_ID** | Unique identifier for the hospital.                             |                                                                                                        |
| **Hospital_Name** | Name of the hospital.                                           | Contains the name of the city in which it is located and the type of the hospital.                     |
| **Hospital_Location** | Location of the hospital.                                       |                                                                                                        |
| **Hospital_Contact** | Phone number of the hospital.                                   |                                                                                                        |

---

## Table 2: Description of columns in Prescription_Billing_Insurance_Data

file: `raw-data/prescription_billing_insurance_data.csv`

| Attribute                            | Description                                        | Remarks                                                                                                                                                                                                                                 |
| :----------------------------------- | :------------------------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Prescription_ID** | Unique identifier for the prescription.            |                                                                                                                                                                                                                                         |
| **Record_ID** | Identifier for the medical record associated with the prescription. |                                                                                                                                                                                                                                         |
| **Pharmacy_ID** | Unique identifier for the pharmacy.                |                                                                                                                                                                                                                                         |
| **Prescription_Detail_ID** | Unique identifier for the prescription detail.     |                                                                                                                                                                                                                                         |
| **Medication_ID** | Unique identifier for the medication prescribed.   |                                                                                                                                                                                                                                         |
| **Medication_Dosage** | Dosage of the medication prescribed.               |                                                                                                                                                                                                                                         |
| **Medication_Quantity** | Quantity of the medication prescribed.             |                                                                                                                                                                                                                                         |
| **Total_Medication_Billing_Amount** | Total billing amount for the prescribed medication. |                                                                                                                                                                                                                                         |
| **Dosage_Start_Date** | Start date for the prescribed medication dosage.   |                                                                                                                                                                                                                                         |
| **Dosage_End_Date** | End date for the prescribed medication dosage.     |                                                                                                                                                                                                                                         |
| **Pharmacy_Name** | Name of the pharmacy.                              |                                                                                                                                                                                                                                         |
| **Pharmacy_Location** | Location of the pharmacy.                          |                                                                                                                                                                                                                                         |
| **Pharmacy_Contact** | Contact number of the pharmacy.                    |                                                                                                                                                                                                                                         |
| **Pharmacy_Email** | Email address of the pharmacy.                     |                                                                                                                                                                                                                                         |
| **Pharmacy_Operating_Hours** | Operating hours of the pharmacy.                   |                                                                                                                                                                                                                                         |
| **Pharmacy_Manager_Name** | Name of the pharmacy manager.                      |                                                                                                                                                                                                                                         |
| **Pharmacy_Website** | Website URL of the pharmacy.                       |                                                                                                                                                                                                                                         |
| **Pharmacy_Services_Offered** | Services offered by the pharmacy.                  | It is actually multivalued, but for simplicity treat it as a single-valued list of multiple services.                                                                                                                                   |
| **Medication_Name** | Name of the medication.                            |                                                                                                                                                                                                                                         |
| **Manufacturer** | Manufacturer of the medication.                    |                                                                                                                                                                                                                                         |
| **Medication_Dosage_Form** | Form of the medication.                            | e.g., Tablet, Syrup.                                                                                                                                                                                                                    |
| **Medication_Strength** | Strength of the medication                         | e.g., 500mg.                                                                                                                                                                                                                            |
| **Medication_Price** | Price of the medication.                           |                                                                                                                                                                                                                                         |
| **Claim_ID** | Unique identifier for the insurance claim.         |                                                                                                                                                                                                                                         |
| **Claim_Status** | Status of the insurance claim.                     |                                                                                                                                                                                                                                         |
| **Claim_Status_Reason** | Reason for the claim status.                       |                                                                                                                                                                                                                                         |
| **Claim_Amount** | Total amount of the claim.                         |                                                                                                                                                                                                                                         |
| **Approved_Amount** | Amount approved by the insurance provider.         |                                                                                                                                                                                                                                         |
| **Claim_Date** | Date of the insurance claim.                       |                                                                                                                                                                                                                                         |
| **Approval_Date** | Date when the insurance claim was approved.        |                                                                                                                                                                                                                                         |
| **Insurance_Provider_ID** | Unique identifier for the insurance provider.      |                                                                                                                                                                                                                                         |
| **Insurance_Provider_Name** | Name of the insurance provider.                    |                                                                                                                                                                                                                                         |
| **Insurance_Provider_Contact** | Contact number of the insurance provider.          |                                                                                                                                                                                                                                         |
| **Insurance_Provider_Coverage_Type** | Types of coverage offered by the insurance provider. | It is actually multivalued, but for simplicity treat it as a single-valued list of multiple coverage types.                                                                                                                            |

---

## Table 3: Description of columns in Service_Billing_Insurance_Data

file: `raw-data/service_billing_insurance_data.csv`

| Attribute                            | Description                                        | Remarks                                                                                                                                                                                                                                 |
| :----------------------------------- | :------------------------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Claim_ID** | Unique identifier for the insurance claim.         |                                                                                                                                                                                                                                         |
| **Claim_Status** | Status of the insurance claim.                     |                                                                                                                                                                                                                                         |
| **Claim_Status_Reason** | Reason for the claim status.                       |                                                                                                                                                                                                                                         |
| **Claim_Amount** | Total amount of the claim.                         |                                                                                                                                                                                                                                         |
| **Approved_Amount** | Amount approved by the insurance provider.         |                                                                                                                                                                                                                                         |
| **Claim_Date** | Date of the insurance claim.                       |                                                                                                                                                                                                                                         |
| **Approval_Date** | Date when the insurance claim was approved.        |                                                                                                                                                                                                                                         |
| **Insurance_Provider_ID** | Unique identifier for the insurance provider.      |                                                                                                                                                                                                                                         |
| **Insurance_Provider_Name** | Name of the insurance provider.                    |                                                                                                                                                                                                                                         |
| **Insurance_Provider_Contact** | Contact number of the insurance provider.          |                                                                                                                                                                                                                                         |
| **Insurance_Provider_Coverage_Type** | Type of coverage offered by the insurance provider. | It is actually multivalued, but for simplicity treat it as a single-valued list of multiple coverage types.                                                                                                                            |
| **Service_Billing_ID** | Unique identifier for the service billing.         |                                                                                                                                                                                                                                         |
| **Appointment_ID** | ID of the appointment linked to the service billing. |                                                                                                                                                                                                                                         |
| **Surgery_ID** | ID of the surgery linked to the service billing.   | Surgeries carried out at a private hospital are billed to the patient, which can be covered by insurance or directly paid.                                                                                                               |
| **Test_ID** | ID of the test linked to the service billing.      | Only Private tests are billed, NHS tests are automatically covered by NHS.                                                                                                                                                              |
| **Service_Billing_Amount** | Amount billed for the service.                     |                                                                                                                                                                                                                                         |
| **Service_Billing_Payment_Status** | Payment status of the service billing.             | Paid, Pending, or Partially Paid                                                                                                                                                                                                        |
| **Service_Billing_Amount_Paid** | Amount paid for the service.                       |                                                                                                                                                                                                                                         |
| **Service_Billing_Payment_Date** | Date when payment for the service was made.        |                                                                                                                                                                                                                                         |

---

## Table 4: Description of columns in Medical_Appointments_Data

file: `raw-data/medical_appointments_data.csv`

| Attribute          | Description                                                    | Remarks                                                                                               |
| :----------------- | :------------------------------------------------------------- | :---------------------------------------------------------------------------------------------------- |
| **Record_ID** | Unique identifier for the medical record.                      |                                                                                                       |
| **Diagnosis** | Medical diagnosis related to the test, surgery, or appointment. |                                                                                                       |
| **Notes** | Notes related to the medical record.                           |                                                                                                       |
| **Appointment_ID** | Identifier for the appointment linked to the medical record.   | NULL, if the medical record is related to a surgery or a medical test.                                |
| **Surgery_ID** | Identifier for the surgery linked to the medical record.       | NULL, if the medical record is related to an appointment or a medical test.                           |
| **Test_ID** | Identifier for the medical test linked to the record.          | NULL, if the medical record is related to an appointment or a surgery.                                |
| **Patient_ID** | Unique identifier for the patient.                             |                                                                                                       |
| **Appointment_Date** | Date when the appointment occurred.                            |                                                                                                       |
| **Appointment_Time** | Time when the appointment occurred.                            |                                                                                                       |
| **Appointment_Status** | Status of the appointment.                                     | Completed, Scheduled, or Cancelled                                                                    |

---

## Table 5: Description of columns in Medical_Surgeries_Data

file: `raw-data/medical_surgeries_data.csv`

| Attribute                  | Description                                                    | Remarks |
| :------------------------- | :------------------------------------------------------------- | :------ |
| **Record_ID** | Unique identifier for the medical record.                      |         |
| **Diagnosis** | Medical diagnosis related to the test, surgery, or appointment. |         |
| **Notes** | Notes related to the medical record.                           |         |
| **Appointment_ID** | Identifier for the appointment linked to the medical record.   |         |
| **Surgery_ID** | Identifier for the surgery linked to the medical record.       |         |
| **Test_ID** | Identifier for the medical test linked to the record.          |         |
| **Patient_ID** | Unique identifier for the patient.                             |         |
| **Surgery_Professional_ID** | Professional ID for the surgery.                               |         |
| **Surgery_Hospital_ID** | Hospital ID where the surgery was performed.                   |         |
| **Surgery_Date** | Date when the surgery was conducted.                           |         |
| **Surgery_Type** | Type of surgery performed.                                     |         |
| **Surgery_Notes** | Notes related to the surgery.                                  |         |
| **Surgery_Post_Operative_Care** | Details of post-operative care for the surgery.                |         |
| **Surgery_Outcome** | Outcome of the surgery.                                        |         |

---

## Table 6: Description of columns in Medical_Tests_Data

file: `raw-data/medical_tests_data.csv`

| Attribute                          | Description                                        | Remarks                       |
| :--------------------------------- | :------------------------------------------------- | :---------------------------- |
| **Record_ID** | Unique identifier for the medical record.          |                               |
| **Diagnosis** | Medical diagnosis related to the test, surgery, or appointment. |                               |
| **Notes** | Notes related to the medical record.               |                               |
| **Appointment_ID** | Identifier for the appointment linked to the medical record. |                               |
| **Surgery_ID** | Identifier for the surgery linked to the medical record. |                               |
| **Test_ID** | Identifier for the medical test linked to the record. |                               |
| **Patient_ID** | Unique identifier for the patient.                 |                               |
| **Test_Recommended_By_Professional_ID** | ID of the professional who recommended the test. |                               |
| **Test_Name** | Name of the medical test.                          |                               |
| **Test_Results** | Results of the medical test.                       |                               |
| **Test_Date** | Date when the test was conducted.                  |                               |
| **Lab_ID** | Unique identifier for the lab where the test was conducted. |                               |
| **Test_Billing_Type** | Billing type for the test.                         | NHS or Private.               |
| **Lab_Name** | Name of the laboratory.                            |                               |
| **Lab_Location** | Location of the laboratory.                        |                               |
| **Lab_Contact** | Contact details for the lab.                       |                               |
| **Lab_Type** | Type of laboratory.                                | e.g., Forensic Science, Cardiology |