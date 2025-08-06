import os
import csv
import logging
from datetime import datetime
import uuid
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy import insert

# Assuming your models.py file is correctly set up and in the same directory
from models import (
    Base, Patient, EmergencyContact, Hospital, Department, Professional, ServiceBilling,
    ProfessionalDepartments, Pharmacy, Lab, InsuranceProvider, Medication, InsuranceClaim,
    Appointment, Surgery, Test, MedicalRecord, Prescription, PrescriptionDetail,
)

# --- Configuration ---
DATA_DIR = os.path.join(os.path.dirname(__file__), '../raw-data')
DB_PATH = os.path.join(os.path.dirname(__file__), '../nhs_db.db')
BATCH_SIZE = 500

# --- Global ID Tracking Sets ---
processed_hospital_ids = set()
processed_department_ids = set()
processed_professional_keys = set()  # (name, role) tuples
processed_patient_ids = set()
processed_emergency_contact_keys = set()  # (patient_id, name, phone) tuples
processed_pharmacy_ids = set()
processed_lab_ids = set()
processed_medication_ids = set()
processed_insurance_provider_ids = set()
processed_appointment_ids = set()
processed_surgery_ids = set()
processed_test_ids = set()
processed_medical_record_ids = set()
processed_prescription_ids = set()
processed_prescription_detail_ids = set()
processed_insurance_claim_ids = set()
processed_service_billing_ids = set()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Helper Functions ---

def parse_date(date_str):
    """Safely parses a YYYY-MM-DD string into a date object."""
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str, '%Y-%m-%d').date()
    except ValueError:
        return None

def parse_datetime(date_str, time_str):
    """Safely parses date and time strings into a datetime object."""
    if not date_str or not time_str:
        return None
    try:
        return datetime.strptime(f"{date_str} {time_str}", '%Y-%m-%d %H:%M')
    except ValueError:
        return None

def cast_float(value):
    """Safely casts a value to a float, returning None on failure."""
    if value in (None, ''):
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None

def flush_batch_if_needed(session, cache_dict, batch_size, force=False):
    """Flushes cached objects to database if batch size is reached or forced."""
    if len(cache_dict) >= batch_size or force:
        if cache_dict:
            session.bulk_save_objects(cache_dict.values())
            session.commit()
            count = len(cache_dict)
            cache_dict.clear()
            return count
    return 0

def flush_objects_if_needed(session, objects_list, batch_size, force=False):
    """Flushes list of objects to database if batch size is reached or forced."""
    if len(objects_list) >= batch_size or force:
        if objects_list:
            session.add_all(objects_list)
            session.commit()
            count = len(objects_list)
            objects_list.clear()
            return count
    return 0

# --- Data Processing Functions ---

def process_appointments_data(session, counters):
    """Processes hospitals, departments, professionals, patients, contacts, and appointments from a single file."""
    logging.info("Processing appointments_data.csv for multiple entities...")
    
    # Use local caches to avoid hitting the DB repeatedly
    hospitals_cache, depts_cache, profs_cache, patients_cache = {}, {}, {}, {}
    contacts_list = []
    prof_dept_relations = set()
    appointments_cache = {}  # Use dict to deduplicate by AppointmentID
    
    file_path = os.path.join(DATA_DIR, 'appointments_data.csv')
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            # # skip the first 200,000 rows
            # if idx < 200000:
            #     continue
            if (idx + 1) % 50000 == 0:
                logging.info(f"... processed {idx + 1} rows from appointments_data.csv")

            # if idx == 1000000:  # For testing, limit to 1 million rows
            #     logging.info("Reached 1,000,000 rows limit for testing.")
            #     break

            # Hospital
            hosp_id = row['Hospital_ID'] if row['Hospital_ID'] else None
            if not hosp_id:
                logging.warning(f"Skipping row with missing Hospital_ID: {row}")
                continue
            if hosp_id not in processed_hospital_ids and hosp_id not in hospitals_cache:
                hospitals_cache[hosp_id] = Hospital(
                    HospitalID=hosp_id, 
                    Name=row['Hospital_Name'], 
                    Location=row['Hospital_Location'],
                    ContactNumber=row.get('Hospital_Contact', None),
                )
                processed_hospital_ids.add(hosp_id)
            
            # Department
            dept_id = row['Department_ID'] if row['Department_ID'] else None
            if not dept_id:
                logging.warning(f"Skipping row with missing Department_ID: {row}")
                continue
            if dept_id not in processed_department_ids and dept_id not in depts_cache:
                depts_cache[dept_id] = Department(
                    DepartmentID=dept_id, 
                    HospitalID=hosp_id, 
                    Name=row['Department_Name'],
                    HeadOfDepartment=row['Head_of_Department']
                )
                processed_department_ids.add(dept_id)

            # Professional
            prof_key = (row['Professional_Name'], row['Professional_Role'])
            if all(prof_key) and prof_key not in processed_professional_keys and prof_key not in profs_cache:
                profs_cache[prof_key] = Professional(ProfessionalID=str(uuid.uuid4()), Name=prof_key[0], Role=prof_key[1])
                processed_professional_keys.add(prof_key)

            # Professional-Department Relationship
            if all(prof_key) and dept_id:
                prof_dept_relations.add((prof_key, dept_id))

            # Patient
            pat_id = row['Patient_ID'] if row['Patient_ID'] else None
            if not pat_id:
                logging.warning(f"Skipping row with missing Patient_ID: {row}")
                continue
            if pat_id not in processed_patient_ids and pat_id not in patients_cache:
                patients_cache[pat_id] = Patient(
                    PatientID=pat_id, 
                    Name=row['Patient_Name'], 
                    DOB=parse_date(row['Patient_Date_Of_Birth']),
                    Gender=row['Patient_Gender'], 
                    Address=row['Patient_Address'],
                    PrefHospitalID=row['Patient_Preferred_Hospital_ID'] if row['Patient_Preferred_Hospital_ID'] else None,
                    PrefPharmacyID=row['Patient_Preferred_Pharmacy_ID'] if row['Patient_Preferred_Pharmacy_ID'] else None,
                    PrefInsuranceProviderID=row['Patient_Preferred_Insurance_Provider'] if row['Patient_Preferred_Insurance_Provider'] else None
                )
                processed_patient_ids.add(pat_id)

            # Emergency Contact
            contact_key = (pat_id, row['Emergency_Contact_Name'], row['Emergency_Contact_Phone'])
            if pat_id and row.get('Emergency_Contact_Name') and contact_key not in processed_emergency_contact_keys:
                contacts_list.append(EmergencyContact(
                    ContactID=str(uuid.uuid4()),
                    PatientID=pat_id, 
                    Name=row['Emergency_Contact_Name'], 
                    Phone=row['Emergency_Contact_Phone'],
                    Relationship=row['Emergency_Contact_Relationship'], 
                    Address=row['Emergency_Contact_Address']
                ))
                processed_emergency_contact_keys.add(contact_key)

            # Store appointment data for later processing (deduplicated by AppointmentID)
            app_id = row['Appointment_ID'] if row['Appointment_ID'] else None
            if app_id and all(prof_key) and app_id not in processed_appointment_ids and app_id not in appointments_cache:
                appointments_cache[app_id] = {
                    'AppointmentID': app_id,
                    'PatientID': pat_id,
                    'prof_key': prof_key,
                    'DepartmentID': dept_id,
                    'DateTime': parse_datetime(row['Appointment_Date'], row['Appointment_Time']),
                    'Status': row['Appointment_Status']
                }
                processed_appointment_ids.add(app_id)

            # Batch flush emergency contacts only (since they don't have dependencies)
            counters['emergency_contacts'] += flush_objects_if_needed(session, contacts_list, BATCH_SIZE)

    # Process all collected entities in dependency order
    logging.info("Processing collected entities in batches...")
    
    # 1. Hospitals first (no dependencies)
    logging.info("Processing Hospitals in batches...")
    hospital_list = list(hospitals_cache.values())
    for i in range(0, len(hospital_list), BATCH_SIZE):
        logging.info(f"Processing hospitals batch {i // BATCH_SIZE + 1} of {len(hospital_list) // BATCH_SIZE + 1}")
        batch = hospital_list[i:i + BATCH_SIZE]
        session.bulk_save_objects(batch)
        session.commit()
        counters['hospitals'] += len(batch)
    
    # 2. Departments (depend on hospitals)
    logging.info("Processing Departments in batches...")
    dept_list = list(depts_cache.values())
    for i in range(0, len(dept_list), BATCH_SIZE):
        logging.info(f"Processing departments batch {i // BATCH_SIZE + 1} of {len(dept_list) // BATCH_SIZE + 1}")
        batch = dept_list[i:i + BATCH_SIZE]
        session.bulk_save_objects(batch)
        session.commit()
        counters['departments'] += len(batch)
    
    # 3. Professionals (no dependencies)
    logging.info("Processing Professionals in batches...")
    prof_list = list(profs_cache.values())
    for i in range(0, len(prof_list), BATCH_SIZE):
        logging.info(f"Processing professionals batch {i // BATCH_SIZE + 1} of {len(prof_list) // BATCH_SIZE + 1}")
        batch = prof_list[i:i + BATCH_SIZE]
        session.bulk_save_objects(batch)
        session.commit()
        counters['professionals'] += len(batch)
    
    # 4. Patients (may reference hospitals/pharmacies/insurance providers)
    logging.info("Processing Patients in batches...")
    patient_list = list(patients_cache.values())
    for i in range(0, len(patient_list), BATCH_SIZE):
        logging.info(f"Processing patients batch {i // BATCH_SIZE + 1} of {len(patient_list) // BATCH_SIZE + 1}")
        batch = patient_list[i:i + BATCH_SIZE]
        session.bulk_save_objects(batch)
        session.commit()
        counters['patients'] += len(batch)
    
    # 5. Flush remaining emergency contacts
    counters['emergency_contacts'] += flush_objects_if_needed(session, contacts_list, BATCH_SIZE, force=True)
    
    # 6. Create M:N relationships now that Professionals have DB-generated IDs
    logging.info("Processing Professional-Department relationships in batches...")
    profs_with_ids = {(p.Name, p.Role): p.ProfessionalID for p in session.query(Professional.ProfessionalID, Professional.Name, Professional.Role)}
    relation_mappings = [
        {'ProfessionalID': profs_with_ids[prof_key], 'DepartmentID': dept_id}
        for prof_key, dept_id in prof_dept_relations if prof_key in profs_with_ids
    ]
    if relation_mappings:
        for i in range(0, len(relation_mappings), BATCH_SIZE):
            batch = relation_mappings[i:i + BATCH_SIZE]
            stmt = insert(ProfessionalDepartments).values(batch)
            session.execute(stmt)
            session.commit()
        counters['prof_dept_relations'] = len(relation_mappings)
    
    # 7. Process appointments with professional IDs (now deduplicated)
    logging.info("Processing appointments in batches...")
    appointment_objects = []
    for app_data in appointments_cache.values():
        prof_id = profs_with_ids.get(app_data['prof_key'])
        if prof_id:
            appointment_objects.append(Appointment(
                AppointmentID=app_data['AppointmentID'],
                PatientID=app_data['PatientID'],
                ProfessionalID=prof_id,
                DepartmentID=app_data['DepartmentID'],
                DateTime=app_data['DateTime'],
                Status=app_data['Status']
            ))
    
    for i in range(0, len(appointment_objects), BATCH_SIZE):
        batch = appointment_objects[i:i + BATCH_SIZE]
        session.add_all(batch)
        session.commit()
        counters['appointments'] += len(batch)

def process_medical_events_data(session, counters):
    """Processes tests, appointments, surgeries, and their associated medical records."""
    logging.info("Processing medical event files (Tests, Appointments, Surgeries)...")
    
    # Collect all data first, then process in batches
    labs_cache, medical_records_cache = {}, {}
    tests_list, appointments_list, surgeries_list = [], [], []
    
    # 1. Process medical_tests_data.csv
    file_path = os.path.join(DATA_DIR, 'medical_tests_data.csv')
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            if (idx + 1) % 50000 == 0:
                logging.info(f"  ...processed {idx + 1} rows from medical_tests_data.csv")
            lab_id = row['Lab_ID'] if row['Lab_ID'] else None
            if not lab_id:
                logging.warning(f"Skipping row with missing Lab_ID: {row}")
                continue
            if lab_id not in processed_lab_ids and lab_id not in labs_cache:
                labs_cache[lab_id] = Lab(
                    LabID=lab_id, 
                    Name=row['Lab_Name'], 
                    Location=row['Lab_Location'],
                    ContactNumber=row['Lab_Contact'],
                    TestType=row['Lab_Type']
                )
                processed_lab_ids.add(lab_id)
            
            test_id = row['Test_ID'] if row['Test_ID'] else None
            if test_id and test_id not in processed_test_ids:
                tests_list.append(Test(
                    TestID=test_id, 
                    PatientID=row['Patient_ID'] if row['Patient_ID'] else None, 
                    RecommendingProfessionalID=row['Test_Recommended_By_Professional_ID'] if row['Test_Recommended_By_Professional_ID'] else None,
                    LabID=lab_id, 
                    TestName=row['Test_Name'], 
                    Results=row['Test_Results'],
                    Date=parse_date(row['Test_Date']),
                    BillingType=row['Test_Billing_Type']
                ))
                processed_test_ids.add(test_id)
            
            record_id = row['Record_ID'] if row['Record_ID'] else None
            if record_id and record_id not in processed_medical_record_ids and record_id not in medical_records_cache:
                medical_records_cache[record_id] = MedicalRecord(
                    RecordID=record_id, 
                    PatientID=row['Patient_ID'] if row['Patient_ID'] else None, 
                    Diagnosis=row['Diagnosis'], 
                    Notes=row['Notes'], 
                    TestID=test_id
                )
                processed_medical_record_ids.add(record_id)

    # 2. Process medical_appointments_data.csv
    file_path = os.path.join(DATA_DIR, 'medical_appointments_data.csv')
    with open(file_path, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            app_id = row['Appointment_ID'] if row['Appointment_ID'] else None
            # Check if this appointment is already processed globally
            if app_id and app_id not in processed_appointment_ids:
                appointments_list.append(Appointment(
                    AppointmentID=app_id, 
                    PatientID=row['Patient_ID'] if row['Patient_ID'] else None, 
                    DateTime=parse_datetime(row['Appointment_Date'], row['Appointment_Time']), 
                    Status=row['Appointment_Status']
                ))
                processed_appointment_ids.add(app_id)
            elif app_id in processed_appointment_ids:
                logging.debug(f"Skipping duplicate Appointment_ID: {app_id}")
            
            record_id = row['Record_ID'] if row['Record_ID'] else None
            if record_id:
                if record_id in medical_records_cache:
                    medical_records_cache[record_id].AppointmentID = app_id # Merge
                elif record_id not in processed_medical_record_ids:
                    medical_records_cache[record_id] = MedicalRecord(
                        RecordID=record_id, 
                        PatientID=row['Patient_ID'] if row['Patient_ID'] else None, 
                        Diagnosis=row['Diagnosis'], 
                        Notes=row['Notes'], 
                        AppointmentID=app_id
                    )
                    processed_medical_record_ids.add(record_id)

    # 3. Process medical_surgeries_data.csv
    file_path = os.path.join(DATA_DIR, 'medical_surgeries_data.csv')
    with open(file_path, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            surg_id = row['Surgery_ID'] if row['Surgery_ID'] else None
            if surg_id and surg_id not in processed_surgery_ids:
                surgeries_list.append(Surgery(
                    SurgeryID=surg_id, 
                    PatientID=row['Patient_ID'] if row['Patient_ID'] else None, 
                    ProfessionalID=row['Surgery_Professional_ID'] if row['Surgery_Professional_ID'] else None,
                    HospitalID=row['Surgery_Hospital_ID'] if row['Surgery_Hospital_ID'] else None,
                    Date=parse_date(row['Surgery_Date']), 
                    Type=row['Surgery_Type'],
                    Notes=row['Surgery_Notes'],
                    PostOpCare=row['Surgery_Post_Operative_Care'],
                    Outcome=row['Surgery_Outcome']
                ))
                processed_surgery_ids.add(surg_id)

            record_id = row['Record_ID'] if row['Record_ID'] else None
            if record_id:
                if record_id in medical_records_cache:
                    medical_records_cache[record_id].SurgeryID = surg_id # Merge
                elif record_id not in processed_medical_record_ids:
                    medical_records_cache[record_id] = MedicalRecord(
                        RecordID=record_id, 
                        PatientID=row['Patient_ID'] if row['Patient_ID'] else None, 
                        Diagnosis=row['Diagnosis'], 
                        Notes=row['Notes'], 
                        SurgeryID=surg_id
                    )
                    processed_medical_record_ids.add(record_id)

    # Now process all collected data in batches
    logging.info("Processing Labs in batches...")
    lab_list = list(labs_cache.values())
    for i in range(0, len(lab_list), BATCH_SIZE):
        batch = lab_list[i:i + BATCH_SIZE]
        session.bulk_save_objects(batch)
        session.commit()
        counters['labs'] += len(batch)

    logging.info("Processing Tests in batches...")
    for i in range(0, len(tests_list), BATCH_SIZE):
        batch = tests_list[i:i + BATCH_SIZE]
        session.add_all(batch)
        session.commit()

    logging.info("Processing Appointments in batches...")
    for i in range(0, len(appointments_list), BATCH_SIZE):
        batch = appointments_list[i:i + BATCH_SIZE]
        session.add_all(batch)
        session.commit()

    logging.info("Processing Surgeries in batches...")
    for i in range(0, len(surgeries_list), BATCH_SIZE):
        batch = surgeries_list[i:i + BATCH_SIZE]
        session.add_all(batch)
        session.commit()
    
    logging.info("Processing Medical Records in batches...")
    medical_records_list = list(medical_records_cache.values())
    for i in range(0, len(medical_records_list), BATCH_SIZE):
        batch = medical_records_list[i:i + BATCH_SIZE]
        session.add_all(batch)
        session.commit()
    counters['medical_records'] += len(medical_records_cache)

def process_prescriptions_and_claims_data(session, counters):
    """Processes prescriptions, their details, and associated insurance claims."""
    logging.info("Processing prescription_billing_insurance_data.csv...")
    file_path = os.path.join(DATA_DIR, 'prescription_billing_insurance_data.csv')
    
    # Collect all data first
    pharmacies_cache, medications_cache, providers_cache = {}, {}, {}
    prescriptions_list, prescription_details_list, insurance_claims_list = [], [], []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            # Pharmacy
            pharm_id = row['Pharmacy_ID'] if row['Pharmacy_ID'] else None
            if pharm_id and pharm_id not in processed_pharmacy_ids and pharm_id not in pharmacies_cache:
                pharmacies_cache[pharm_id] = Pharmacy(
                    PharmacyID=pharm_id, 
                    Name=row['Pharmacy_Name'], 
                    Location=row['Pharmacy_Location'],
                    ContactNumber=row['Pharmacy_Contact'],
                    OperatingHours=row['Pharmacy_Operating_Hours'],
                    ManagerName=row['Pharmacy_Manager_Name'],
                    Website=row['Pharmacy_Website'],
                    ServicesOffered=row['Pharmacy_Services_Offered']
                )
                processed_pharmacy_ids.add(pharm_id)

            # Medication
            med_id = row['Medication_ID'] if row['Medication_ID'] else None
            if med_id and med_id not in processed_medication_ids and med_id not in medications_cache:
                medications_cache[med_id] = Medication(
                    MedicationID=med_id, 
                    Name=row['Medication_Name'], 
                    Manufacturer=row['Manufacturer'], 
                    DosageForm=row['Medication_Dosage_Form'],
                    Strength=row['Medication_Strength'],
                    Price=cast_float(row['Medication_Price'])
                )
                processed_medication_ids.add(med_id)

            # Insurance Provider
            ins_id = row['Insurance_Provider_ID'] if row['Insurance_Provider_ID'] else None
            if ins_id and ins_id not in processed_insurance_provider_ids and ins_id not in providers_cache:
                providers_cache[ins_id] = InsuranceProvider(
                    InsuranceProviderID=ins_id, 
                    Name=row['Insurance_Provider_Name'],
                    ContactNumber=row['Insurance_Provider_Contact'],
                    CoverageType=row['Insurance_Provider_Coverage_Type']
                )
                processed_insurance_provider_ids.add(ins_id)

            # Prescription
            pres_id = row['Prescription_ID'] if row['Prescription_ID'] else None
            if pres_id and pres_id not in processed_prescription_ids:
                prescriptions_list.append(Prescription(
                    PrescriptionID=pres_id, 
                    RecordID=row['Record_ID'] if row['Record_ID'] else None, 
                    PharmacyID=pharm_id
                ))
                processed_prescription_ids.add(pres_id)
            
            # Prescription Detail
            pres_detail_id = row['Prescription_Detail_ID'] if row['Prescription_Detail_ID'] else None
            if pres_detail_id and pres_detail_id not in processed_prescription_detail_ids:
                prescription_details_list.append(PrescriptionDetail(
                    PrescriptionDetailID=pres_detail_id, 
                    PrescriptionID=pres_id, 
                    MedicationID=med_id,
                    Dosage=row['Medication_Dosage'], 
                    Quantity=row['Medication_Quantity'] if row['Medication_Quantity'] else None,
                    TotalBillingAmount=cast_float(row['Total_Medication_Billing_Amount']),
                    StartDate=parse_date(row['Dosage_Start_Date']),
                    EndDate=parse_date(row['Dosage_End_Date'])
                ))
                processed_prescription_detail_ids.add(pres_detail_id)

            # Insurance Claim
            claim_id = row['Claim_ID'] if row['Claim_ID'] else None
            if claim_id and claim_id not in processed_insurance_claim_ids:
                insurance_claims_list.append(InsuranceClaim(
                    ClaimID=claim_id, 
                    InsuranceProviderID=ins_id, 
                    PrescriptionDetailID=pres_detail_id,
                    Status=row['Claim_Status'], 
                    StatusReason=row['Claim_Status_Reason'],
                    AmountClaimed=cast_float(row['Claim_Amount']), 
                    ApprovedAmount=cast_float(row['Approved_Amount']),
                    Date=parse_date(row['Claim_Date']),
                    ApprovalDate=parse_date(row['Approval_Date'])
                ))
                processed_insurance_claim_ids.add(claim_id)

    # Process all collected data in dependency order
    logging.info("Processing Pharmacies in batches...")
    pharmacy_list = list(pharmacies_cache.values())
    for i in range(0, len(pharmacy_list), BATCH_SIZE):
        batch = pharmacy_list[i:i + BATCH_SIZE]
        session.bulk_save_objects(batch)
        session.commit()
        counters['pharmacies'] += len(batch)

    logging.info("Processing Medications in batches...")
    medication_list = list(medications_cache.values())
    for i in range(0, len(medication_list), BATCH_SIZE):
        batch = medication_list[i:i + BATCH_SIZE]
        session.bulk_save_objects(batch)
        session.commit()
        counters['medications'] += len(batch)

    logging.info("Processing Insurance Providers in batches...")
    provider_list = list(providers_cache.values())
    for i in range(0, len(provider_list), BATCH_SIZE):
        batch = provider_list[i:i + BATCH_SIZE]
        session.bulk_save_objects(batch)
        session.commit()
        counters['insurance_providers'] += len(batch)

    logging.info("Processing Prescriptions in batches...")
    for i in range(0, len(prescriptions_list), BATCH_SIZE):
        batch = prescriptions_list[i:i + BATCH_SIZE]
        session.add_all(batch)
        session.commit()

    logging.info("Processing Prescription Details in batches...")
    for i in range(0, len(prescription_details_list), BATCH_SIZE):
        batch = prescription_details_list[i:i + BATCH_SIZE]
        session.add_all(batch)
        session.commit()

    logging.info("Processing Insurance Claims in batches...")
    for i in range(0, len(insurance_claims_list), BATCH_SIZE):
        batch = insurance_claims_list[i:i + BATCH_SIZE]
        session.add_all(batch)
        session.commit()

def process_service_billing_data(session, counters):
    """Processes service billings and their associated insurance claims."""
    logging.info("Processing service_billing_insurance_data.csv...")
    file_path = os.path.join(DATA_DIR, 'service_billing_insurance_data.csv')

    # Collect all data first
    providers_cache = {p.InsuranceProviderID: p for p in session.query(InsuranceProvider)}
    new_providers_list = []
    service_billings_list = []
    insurance_claims_list = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            # Insurance Provider
            ins_id = row['Insurance_Provider_ID'] if row['Insurance_Provider_ID'] else None
            if ins_id and ins_id not in processed_insurance_provider_ids and ins_id not in providers_cache:
                provider = InsuranceProvider(
                    InsuranceProviderID=ins_id, 
                    Name=row['Insurance_Provider_Name'],
                    ContactNumber=row['Insurance_Provider_Contact'],
                    CoverageType=row['Insurance_Provider_Coverage_Type']
                )
                providers_cache[ins_id] = provider
                new_providers_list.append(provider)
                processed_insurance_provider_ids.add(ins_id)

            # Service Billing
            billing_id = row['Service_Billing_ID'] if row['Service_Billing_ID'] else None
            if billing_id and billing_id not in processed_service_billing_ids:
                service_billings_list.append(ServiceBilling(
                    BillingID=billing_id, 
                    PatientID=get_patient_id_from_billing(session, row),
                    AppointmentID=row['Appointment_ID'] if row['Appointment_ID'] else None,
                    SurgeryID=row['Surgery_ID'] if row['Surgery_ID'] else None, 
                    TestID=row['Test_ID'] if row['Test_ID'] else None,
                    Amount=cast_float(row['Service_Billing_Amount']), 
                    PaymentStatus=row['Service_Billing_Payment_Status'],
                    AmountPaid=cast_float(row['Service_Billing_Amount_Paid']),
                    PaymentDate=parse_date(row['Service_Billing_Payment_Date'])
                ))
                processed_service_billing_ids.add(billing_id)

            # Insurance Claim
            claim_id = row['Claim_ID'] if row['Claim_ID'] else None
            if claim_id and claim_id not in processed_insurance_claim_ids:
                insurance_claims_list.append(InsuranceClaim(
                    ClaimID=claim_id, 
                    InsuranceProviderID=ins_id, 
                    ServiceBillingID=billing_id,
                    Status=row['Claim_Status'], 
                    StatusReason=row['Claim_Status_Reason'],
                    AmountClaimed=cast_float(row['Claim_Amount']), 
                    ApprovedAmount=cast_float(row['Approved_Amount']),
                    Date=parse_date(row['Claim_Date']),
                    ApprovalDate=parse_date(row['Approval_Date'])
                ))
                processed_insurance_claim_ids.add(claim_id)

    # Process in batches
    if new_providers_list:
        logging.info("Processing new Insurance Providers in batches...")
        for i in range(0, len(new_providers_list), BATCH_SIZE):
            batch = new_providers_list[i:i + BATCH_SIZE]
            session.bulk_save_objects(batch)
            session.commit()
            counters['insurance_providers'] += len(batch)

    logging.info("Processing Service Billings in batches...")
    for i in range(0, len(service_billings_list), BATCH_SIZE):
        batch = service_billings_list[i:i + BATCH_SIZE]
        session.add_all(batch)
        session.commit()

    logging.info("Processing Insurance Claims in batches...")
    for i in range(0, len(insurance_claims_list), BATCH_SIZE):
        batch = insurance_claims_list[i:i + BATCH_SIZE]
        session.add_all(batch)
        session.commit()

def get_patient_id_from_billing(session, row):
    """Helper function to get PatientID for ServiceBilling from related entities."""
    # Try to get PatientID from Appointment, Surgery, or Test
    if row.get('Appointment_ID'):
        app = session.query(Appointment).filter_by(AppointmentID=row['Appointment_ID'] if row['Appointment_ID'] else None).first()
        if app:
            return app.PatientID
    if row.get('Surgery_ID'):
        surg = session.query(Surgery).filter_by(SurgeryID=row['Surgery_ID'] if row['Surgery_ID'] else None).first()
        if surg:
            return surg.PatientID
    if row.get('Test_ID'):
        test = session.query(Test).filter_by(TestID=row['Test_ID'] if row['Test_ID'] else None).first()
        if test:
            return test.PatientID
    return None

def main():
    """Main function to orchestrate the database initialization."""
    logging.info("Starting database initialization.")
    
    # Clear global tracking sets at the start
    global processed_hospital_ids, processed_department_ids, processed_professional_keys
    global processed_patient_ids, processed_emergency_contact_keys, processed_pharmacy_ids
    global processed_lab_ids, processed_medication_ids, processed_insurance_provider_ids
    global processed_appointment_ids, processed_surgery_ids, processed_test_ids
    global processed_medical_record_ids, processed_prescription_ids, processed_prescription_detail_ids
    global processed_insurance_claim_ids, processed_service_billing_ids
    
    processed_hospital_ids.clear()
    processed_department_ids.clear()
    processed_professional_keys.clear()
    processed_patient_ids.clear()
    processed_emergency_contact_keys.clear()
    processed_pharmacy_ids.clear()
    processed_lab_ids.clear()
    processed_medication_ids.clear()
    processed_insurance_provider_ids.clear()
    processed_appointment_ids.clear()
    processed_surgery_ids.clear()
    processed_test_ids.clear()
    processed_medical_record_ids.clear()
    processed_prescription_ids.clear()
    processed_prescription_detail_ids.clear()
    processed_insurance_claim_ids.clear()
    processed_service_billing_ids.clear()
    
    engine = create_engine(f'sqlite:///{DB_PATH}')
    
    # Optional: Recreate the schema for a clean run
    logging.info("Dropping all existing tables...")
    Base.metadata.drop_all(engine)
    logging.info("Creating new tables...")
    Base.metadata.create_all(engine)
    
    Session = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    session = Session()

    # WARNING: These PRAGMAs significantly increase performance but risk database corruption on crash.
    try:
        session.execute(text("PRAGMA journal_mode = MEMORY;"))
        session.execute(text("PRAGMA synchronous = OFF;"))
        session.execute(text("PRAGMA cache_size = -100000;")) # Set cache to 100MB
    except Exception as e:
        logging.error(f"Failed to set SQLite PRAGMAs: {e}")

    counters = {k: 0 for k in [
        'hospitals', 'departments', 'professionals', 'prof_dept_relations', 'patients', 
        'emergency_contacts', 'pharmacies', 'medications', 'insurance_providers', 'prescriptions', 
        'prescription_details', 'insurance_claims', 'service_billings', 'labs', 'tests', 
        'appointments', 'surgeries', 'medical_records'
    ]}

    try:
        # The order of these calls is important to satisfy foreign key constraints.
        process_appointments_data(session, counters)
        process_medical_events_data(session, counters)
        process_prescriptions_and_claims_data(session, counters)
        process_service_billing_data(session, counters)

        logging.info("Data loading process completed successfully.")

    except Exception as e:
        logging.error(f"A critical error occurred: {e}", exc_info=True)
        session.rollback()
    finally:
        # Final counts for objects added directly to the session might need to be queried
        # but this gives a good overview of objects processed through caches.
        logging.info("=== DATABASE INITIALIZATION SUMMARY ===")
        for key, value in counters.items():
            if value > 0:
                logging.info(f"{key.replace('_', ' ').title()}: {value}")
        session.close()
        logging.info("Database session closed.")

if __name__ == "__main__":
    main()