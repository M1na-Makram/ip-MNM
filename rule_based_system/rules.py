from rule_based_system.expert_system import PatientData, HeartDiseaseExpert

def run_expert_system(patient_dict):
    """
    Instantiates the expert system, declares the patient's facts,
    and runs the engine to evaluate the rules.
    """
    engine = HeartDiseaseExpert()
    engine.reset()  # Initialize the engine's internal state
    engine.declare(PatientData(**patient_dict))  # Declare the patient dict as a Fact
    engine.run()  # Run the engine to infer matches

if __name__ == "__main__":
    print("--- Testing Patient 1 (High Risk Profile) ---")
    patient_high_risk = {
        'age': 55,
        'cholesterol': 250,
        'blood_pressure': 145,
        'smoking': 'yes',
        'exercise': 'no',
        'bmi': 31,
        'blood_sugar': 130,
        'chest_pain': 'typical',
        'max_heart_rate': 110,
        'family_history': 'yes'
    }
    run_expert_system(patient_high_risk)
    
    print("\n--- Testing Patient 2 (Low Risk Profile) ---")
    patient_low_risk = {
        'age': 30,
        'cholesterol': 180,
        'blood_pressure': 115,
        'smoking': 'no',
        'exercise': 'regular',
        'bmi': 22,
        'blood_sugar': 90,
        'chest_pain': 'asymptomatic',
        'max_heart_rate': 160,
        'family_history': 'no'
    }
    run_expert_system(patient_low_risk)
