import pandas as pd
import numpy as np
import random

def generate_comprehensive_tiered_dataset(num_rows=50000):
    # Mapping every specialization to tiered jobs (Tier 1: >8.5, Tier 2: 7.5-8.5, Tier 3: <7.5)
    logic = {
        # Engineering Branches (B.Tech/M.Tech)
        "Electrical and Electronics Engineering": [["Power Systems Engineer", "Control Systems Lead"], ["Electrical Engineer"], ["Maintenance Engineer"]],
        "Electronics and Communication Engineering": [["VLSI Engineer", "Embedded Systems Lead"], ["Electronics Engineer"], ["Network Technician"]],
        "Computer Science and Engineering": [["AI Engineer", "Data Scientist"], ["Software Engineer"], ["Technical Support"]],
        "Mechanical Engineering": [["R&D Engineer", "Design Engineer"], ["Production Engineer"], ["Maintenance Technician"]],
        "Civil Engineering": [["Structural Engineer", "Urban Planner"], ["Site Engineer"], ["Draftsman"]],
        "Biotechnology": [["Genetic Engineer", "Senior Scientist"], ["Bioprocess Engineer"], ["Lab Assistant"]],
        "Biomedical Engineering": [["Medical Device Designer", "Bio-instrumentation Lead"], ["Biomedical Engineer"], ["Equipment Technician"]],
        "Aeronautical Engineering": [["Aerodynamics Specialist"], ["Aircraft Maintenance Engineer"], ["Technical Writer"]],
        "Aerospace Engineering": [["Spacecraft Systems Engineer"], ["Satellite Controller"], ["Quality Inspector"]],
        "Metallurgical Engineering": [["Materials Scientist"], ["Process Metallurgist"], ["Quality Control"]],
        "Textile Engineering": [["Fiber Scientist"], ["Production Manager"], ["Textile Designer"]],
        "Marine Engineering": [["Naval Architect"], ["Marine Engineer"], ["Ship Maintenance"]],
        "Chemical Engineering": [["Process Design Engineer"], ["Chemical Plant Operator"], ["Safety Officer"]],
        "Information Technology": [["Cloud Architect", "Solutions Architect"], ["IT Engineer"], ["Helpdesk Specialist"]],
        "Petroleum Engineering": [["Reservoir Engineer"], ["Drilling Engineer"], ["Site Supervisor"]],
        "Environmental Engineering": [["Sustainability Consultant"], ["Environmental Engineer"], ["Waste Manager"]],
        "Mining Engineering": [["Mine Manager"], ["Extraction Engineer"], ["Safety Inspector"]],
        
        # Science & Healthcare Branches (B.Sc/M.Sc)
        "Physics": [["Astrophysicist", "Quantum Researcher"], ["Physics Teacher"], ["Lab Technician"]],
        "Chemistry": [["Analytical Chemist"], ["Pharmacologist"], ["Chemical Assistant"]],
        "Mathematics": [["Actuary", "Cryptographer"], ["Mathematics Professor"], ["Data Entry Operator"]],
        "Statistics": [["Statistical Modeler", "Data Analyst"], ["Survey Researcher"], ["Data Clerk"]],
        "Computer Science": [["Web Developer", "Database Admin"], ["Systems Analyst"], ["Computer Instructor"]],
        "Electronics": [["Circuit Designer"], ["Hardware Repair Lead"], ["Testing Assistant"]],
        "Forensic Science": [["Forensic Pathologist"], ["Crime Scene Investigator"], ["Evidence Custodian"]],
        "Nursing": [["Nurse Practitioner", "Nurse Manager"], ["Registered Nurse"], ["Nursing Assistant"]],
        "Medical Laboratory Technology": [["Lab Director"], ["Medical Lab Technologist"], ["Phlebotomist"]],
        "Radiology": [["Senior Radiologist"], ["Radiology Technician"], ["X-Ray Assistant"]],
        "Operation Theatre Technology": [["OT Manager"], ["OT Technician"], ["Surgical Assistant"]],
        "Optometry": [["Senior Optometrist"], ["Eye Consultant"], ["Optical Assistant"]],
        "Dialysis Technology": [["Dialysis Specialist"], ["Dialysis Technician"], ["Health Aide"]],
        "Anesthesia Technology": [["Chief Anesthesia Tech"], ["Anesthesia Technician"], ["Medical Assistant"]]
    }

    # Defining which specialization belongs to which Degree category
    btech_specs = [
        "Electrical and Electronics Engineering", "Electronics and Communication Engineering",
        "Computer Science and Engineering", "Mechanical Engineering", "Civil Engineering",
        "Biotechnology", "Biomedical Engineering", "Aeronautical Engineering",
        "Aerospace Engineering", "Metallurgical Engineering", "Textile Engineering",
        "Marine Engineering", "Chemical Engineering", "Information Technology",
        "Petroleum Engineering", "Environmental Engineering", "Mining Engineering"
    ]

    rows = []
    all_specs = list(logic.keys())

    for _ in range(num_rows):
        spec = random.choice(all_specs)
        
        # Assign Degree based on specialization
        if spec in btech_specs:
            degree = random.choice(["B.Tech", "M.Tech"])
        else:
            degree = random.choice(["B.Sc", "M.Sc"])
            
        cgpa = round(random.uniform(6.0, 10.0), 2)
        
        # 88% Strong logic for accuracy, 12% noise
        if random.random() < 0.88:
            if cgpa >= 8.5:
                job = random.choice(logic[spec][0]) # Tier 1
            elif cgpa >= 7.5:
                job = random.choice(logic[spec][1]) # Tier 2
            else:
                job = random.choice(logic[spec][2]) # Tier 3
        else:
            # Noise: random job from any category
            rand_spec = random.choice(all_specs)
            job = random.choice([j for sub in logic[rand_spec] for j in sub])

        rows.append([degree, spec, cgpa, job])

    return pd.DataFrame(rows, columns=['Degree', 'Specialization', 'CGPA', 'Job_Role'])

# Generate and Save
df = generate_comprehensive_tiered_dataset(50000)
df.to_csv("Final_Job_Data.csv", index=False)
print("Comprehensive dataset saved!")