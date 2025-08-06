from sqlalchemy import (
    Column, String, DateTime, Float, ForeignKey, Text, Table, Integer
)
from sqlalchemy import Date as DateColumn
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()

# Junction table for Professionals and Departments (M:N)
ProfessionalDepartments = Table(
    'ProfessionalDepartments', Base.metadata,
    Column('ProfessionalID', String, ForeignKey('Professionals.ProfessionalID'), primary_key=True),
    Column('DepartmentID', String, ForeignKey('Departments.DepartmentID'), primary_key=True)
)

class Patient(Base):
    __tablename__ = 'Patients'
    PatientID = Column(String, primary_key=True)
    Name = Column(String)
    DOB = Column(DateColumn)
    Gender = Column(String)
    Address = Column(String)
    PrefHospitalID = Column(String, ForeignKey('Hospitals.HospitalID'))
    PrefPharmacyID = Column(String, ForeignKey('Pharmacies.PharmacyID'))
    PrefInsuranceProviderID = Column(String, ForeignKey('InsuranceProviders.InsuranceProviderID'))

class EmergencyContact(Base):
    __tablename__ = 'EmergencyContacts'
    ContactID = Column(String, primary_key=True)
    PatientID = Column(String, ForeignKey('Patients.PatientID'))
    Relationship = Column(String)
    Name = Column(String)
    Phone = Column(String)
    Address = Column(String)

class Hospital(Base):
    __tablename__ = 'Hospitals'
    HospitalID = Column(String, primary_key=True)
    Name = Column(String)
    Location = Column(String)
    ContactNumber = Column(String)

class Department(Base):
    __tablename__ = 'Departments'
    DepartmentID = Column(String, primary_key=True)
    HospitalID = Column(String, ForeignKey('Hospitals.HospitalID'))
    Name = Column(String)
    HeadOfDepartment = Column(String)

class Professional(Base):
    __tablename__ = 'Professionals'
    ProfessionalID = Column(String, primary_key=True)
    Name = Column(String)
    Role = Column(String)
    departments = relationship(
        "Department",
        secondary=ProfessionalDepartments,
        backref="professionals"
    )

class Pharmacy(Base):
    __tablename__ = 'Pharmacies'
    PharmacyID = Column(String, primary_key=True)
    Name = Column(String)
    Location = Column(String)
    ContactNumber = Column(String)
    OperatingHours = Column(String)
    ManagerName = Column(String)
    Website = Column(String)
    ServicesOffered = Column(Text)

class Lab(Base):
    __tablename__ = 'Labs'
    LabID = Column(String, primary_key=True)
    Name = Column(String)
    Location = Column(String)
    ContactNumber = Column(String)
    TestType = Column(String)

class InsuranceProvider(Base):
    __tablename__ = 'InsuranceProviders'
    InsuranceProviderID = Column(String, primary_key=True)
    Name = Column(String)
    ContactNumber = Column(String)
    CoverageType = Column(Text)

class Medication(Base):
    __tablename__ = 'Medications'
    MedicationID = Column(String, primary_key=True)
    Name = Column(String)
    Manufacturer = Column(String)
    DosageForm = Column(String)
    Strength = Column(String)
    Price = Column(Float)

class Appointment(Base):
    __tablename__ = 'Appointments'
    AppointmentID = Column(String, primary_key=True)
    PatientID = Column(String, ForeignKey('Patients.PatientID'))
    ProfessionalID = Column(String, ForeignKey('Professionals.ProfessionalID'))
    DepartmentID = Column(String, ForeignKey('Departments.DepartmentID'))
    DateTime = Column(DateTime)
    Status = Column(String)

class Surgery(Base):
    __tablename__ = 'Surgeries'
    SurgeryID = Column(String, primary_key=True)
    PatientID = Column(String, ForeignKey('Patients.PatientID'))
    ProfessionalID = Column(String, ForeignKey('Professionals.ProfessionalID'))
    HospitalID = Column(String, ForeignKey('Hospitals.HospitalID'))
    Date = Column(DateColumn)
    Type = Column(String)
    Notes = Column(Text)
    PostOpCare = Column(Text)
    Outcome = Column(String)

class Test(Base):
    __tablename__ = 'Tests'
    TestID = Column(String, primary_key=True)
    PatientID = Column(String, ForeignKey('Patients.PatientID'))
    RecommendingProfessionalID = Column(String, ForeignKey('Professionals.ProfessionalID'))
    LabID = Column(String, ForeignKey('Labs.LabID'))
    TestName = Column(String)
    Results = Column(Text)
    Date = Column(DateColumn)
    BillingType = Column(String)

class MedicalRecord(Base):
    __tablename__ = 'MedicalRecords'
    RecordID = Column(String, primary_key=True)
    PatientID = Column(String, ForeignKey('Patients.PatientID'))
    AppointmentID = Column(String, ForeignKey('Appointments.AppointmentID'), nullable=True)
    SurgeryID = Column(String, ForeignKey('Surgeries.SurgeryID'), nullable=True)
    TestID = Column(String, ForeignKey('Tests.TestID'), nullable=True)
    Diagnosis = Column(Text)
    Notes = Column(Text)

class Prescription(Base):
    __tablename__ = 'Prescriptions'
    PrescriptionID = Column(String, primary_key=True)
    RecordID = Column(String, ForeignKey('MedicalRecords.RecordID'))
    PharmacyID = Column(String, ForeignKey('Pharmacies.PharmacyID'))

class PrescriptionDetail(Base):
    __tablename__ = 'PrescriptionDetails'
    PrescriptionDetailID = Column(String, primary_key=True)
    PrescriptionID = Column(String, ForeignKey('Prescriptions.PrescriptionID'))
    MedicationID = Column(String, ForeignKey('Medications.MedicationID'))
    Dosage = Column(String)
    Quantity = Column(Integer)
    TotalBillingAmount = Column(Float)
    StartDate = Column(DateColumn)
    EndDate = Column(DateColumn)

class ServiceBilling(Base):
    __tablename__ = 'ServiceBillings'
    BillingID = Column(String, primary_key=True)
    PatientID = Column(String, ForeignKey('Patients.PatientID'))
    AppointmentID = Column(String, ForeignKey('Appointments.AppointmentID'), nullable=True)
    SurgeryID = Column(String, ForeignKey('Surgeries.SurgeryID'), nullable=True)
    TestID = Column(String, ForeignKey('Tests.TestID'), nullable=True)
    Amount = Column(Float)
    PaymentStatus = Column(String)
    AmountPaid = Column(Float)
    PaymentDate = Column(DateColumn)

class InsuranceClaim(Base):
    __tablename__ = 'InsuranceClaims'
    ClaimID = Column(String, primary_key=True)
    InsuranceProviderID = Column(String, ForeignKey('InsuranceProviders.InsuranceProviderID'))
    PrescriptionDetailID = Column(String, ForeignKey('PrescriptionDetails.PrescriptionDetailID'), nullable=True)
    ServiceBillingID = Column(String, ForeignKey('ServiceBillings.BillingID'), nullable=True)
    Status = Column(String)
    StatusReason = Column(Text)
    AmountClaimed = Column(Float)
    ApprovedAmount = Column(Float)
    Date = Column(DateColumn)
    ApprovalDate = Column(DateColumn)
