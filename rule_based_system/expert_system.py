import collections
import collections.abc
# Patch for experta compatibility with Python 3.10+
collections.Mapping = collections.abc.Mapping
collections.Iterable = collections.abc.Iterable
collections.MutableMapping = collections.abc.MutableMapping
collections.Sequence = collections.abc.Sequence

from experta import Fact, KnowledgeEngine, Rule, P

class PatientData(Fact):
    """Fact representing patient health data."""
    pass

class HeartDiseaseExpert(KnowledgeEngine):
    
    # 1. Cholesterol > 240 AND age > 50 -> high risk
    @Rule(PatientData(cholesterol=P(lambda x: x > 240), age=P(lambda x: x > 50)))
    def rule_1(self):
        print("Rule 1 fired: Cholesterol > 240 AND age > 50 -> Risk: high")

    # 2. Blood pressure > 140 AND smoking = yes -> high risk
    @Rule(PatientData(blood_pressure=P(lambda x: x > 140), smoking='yes'))
    def rule_2(self):
        print("Rule 2 fired: Blood pressure > 140 AND smoking = yes -> Risk: high")

    # 3. Exercise = regular AND bmi < 25 -> low risk
    @Rule(PatientData(exercise='regular', bmi=P(lambda x: x < 25)))
    def rule_3(self):
        print("Rule 3 fired: Exercise = regular AND bmi < 25 -> Risk: low")

    # 4. Blood sugar > 120 AND age > 45 -> medium risk
    @Rule(PatientData(blood_sugar=P(lambda x: x > 120), age=P(lambda x: x > 45)))
    def rule_4(self):
        print("Rule 4 fired: Blood sugar > 120 AND age > 45 -> Risk: medium")

    # 5. Chest pain = typical AND max_heart_rate < 120 -> high risk
    @Rule(PatientData(chest_pain='typical', max_heart_rate=P(lambda x: x < 120)))
    def rule_5(self):
        print("Rule 5 fired: Chest pain = typical AND max_heart_rate < 120 -> Risk: high")

    # 6. BMI > 30 AND blood_pressure > 130 -> medium risk
    @Rule(PatientData(bmi=P(lambda x: x > 30), blood_pressure=P(lambda x: x > 130)))
    def rule_6(self):
        print("Rule 6 fired: BMI > 30 AND blood_pressure > 130 -> Risk: medium")

    # 7. Family history = yes AND age > 40 -> medium risk
    @Rule(PatientData(family_history='yes', age=P(lambda x: x > 40)))
    def rule_7(self):
        print("Rule 7 fired: Family history = yes AND age > 40 -> Risk: medium")

    # 8. All healthy indicators -> low risk
    # Definition for healthy: cholesterol < 200, BP < 120, BMI < 25, no smoking, regular exercise
    @Rule(PatientData(
        cholesterol=P(lambda x: x < 200),
        blood_pressure=P(lambda x: x < 120),
        bmi=P(lambda x: x < 25),
        smoking='no',
        exercise='regular'
    ))
    def rule_8(self):
        print("Rule 8 fired: All healthy indicators -> Risk: low")

    # 9. No exercise AND smoking = yes -> high risk
    @Rule(PatientData(exercise='no', smoking='yes'))
    def rule_9(self):
        print("Rule 9 fired: No exercise AND smoking = yes -> Risk: high")

    # 10. Age < 35 AND bmi < 25 AND no smoking -> low risk
    @Rule(PatientData(age=P(lambda x: x < 35), bmi=P(lambda x: x < 25), smoking='no'))
    def rule_10(self):
        print("Rule 10 fired: Age < 35 AND bmi < 25 AND no smoking -> Risk: low")
