from rule_based_system.expert_system import PatientData, HeartDiseaseExpert


def run_expert_system(patient_dict):
    engine = HeartDiseaseExpert()
    engine.reset()
    engine.declare(PatientData(**patient_dict))
    engine.run()


if __name__ == "__main__":
    print("Patient 1 - High Risk Profile")
    run_expert_system({
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
    })

    print("\nPatient 2 - Low Risk Profile")
    run_expert_system({
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
    })