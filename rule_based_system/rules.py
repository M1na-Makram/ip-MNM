from rule_based_system.expert_system import PatientData, HeartRules, HeartDiseaseExpert


def run_expert_system(patient_dict):
    engine = HeartRules()
    engine.reset()
    engine.declare(PatientData(**patient_dict))
    engine.run()


if __name__ == "__main__":
    print("----------Heart Disease System-----------")
    
    data = {
        "age": int(input("Enter your age please: ")),
        "cholesterol": int(input("Cholesterol: ")),
        "blood_pressure": int(input("Blood Pressure: ")),
        "smoking": input("Smoking (y/n): ").lower() == "y",
        "diabetes": input("Diabetes (y/n): ").lower() == "y",
        "obesity": input("Obesity (y/n): ").lower() == "y",
        "exercise": input("Exercise (low/normal/high): ").lower(),
        "chest_pain": input("Chest Pain (yes/no): ").lower(),
        "family_history": input("Family History (y/n): ").lower() == "y",
        "rest_ecg": input("ECG (Normal/Abnormal): ").lower()
    }

    print("\n===========================================")
    print("Running Analysis...")
    run_expert_system(data)
    print("Analysis completed")
    print("===========================================")