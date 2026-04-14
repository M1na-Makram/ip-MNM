import collections
import collections.abc

# experta was written before Python 3.10 moved these out of the top-level
# collections namespace - this just patches them back in
collections.Mapping = collections.abc.Mapping
collections.Iterable = collections.abc.Iterable
collections.MutableMapping = collections.abc.MutableMapping
collections.Sequence = collections.abc.Sequence

from experta import Fact, KnowledgeEngine, Rule, P


class PatientData(Fact):
    pass


class HeartDiseaseExpert(KnowledgeEngine):

    # older patients with high cholesterol are a pretty clear high-risk combo
    @Rule(PatientData(cholesterol=P(lambda x: x > 240), age=P(lambda x: x > 50)))
    def high_chol_older_patient(self):
        print("High cholesterol + age > 50 -> high risk")

    # hypertension + smoking is a bad combo regardless of other factors
    @Rule(PatientData(blood_pressure=P(lambda x: x > 140), smoking='yes'))
    def hypertension_smoker(self):
        print("BP > 140 + smoker -> high risk")

    @Rule(PatientData(exercise='regular', bmi=P(lambda x: x < 25)))
    def active_healthy_weight(self):
        print("Regular exercise + healthy BMI -> low risk")

    # elevated blood sugar at middle age often points to early diabetes risk
    @Rule(PatientData(blood_sugar=P(lambda x: x > 120), age=P(lambda x: x > 45)))
    def elevated_sugar_middle_age(self):
        print("Blood sugar > 120 + age > 45 -> medium risk")

    @Rule(PatientData(chest_pain='typical', max_heart_rate=P(lambda x: x < 120)))
    def chest_pain_low_hr(self):
        print("Typical chest pain + low max HR -> high risk")

    @Rule(PatientData(bmi=P(lambda x: x > 30), blood_pressure=P(lambda x: x > 130)))
    def obese_hypertensive(self):
        print("BMI > 30 + elevated BP -> medium risk")

    @Rule(PatientData(family_history='yes', age=P(lambda x: x > 40)))
    def family_history_older(self):
        print("Family history + age > 40 -> medium risk")

    # all markers in a healthy range - this is the "textbook healthy" case
    @Rule(PatientData(
        cholesterol=P(lambda x: x < 200),
        blood_pressure=P(lambda x: x < 120),
        bmi=P(lambda x: x < 25),
        smoking='no',
        exercise='regular'
    ))
    def all_healthy_indicators(self):
        print("All indicators healthy -> low risk")

    @Rule(PatientData(exercise='no', smoking='yes'))
    def sedentary_smoker(self):
        print("No exercise + smoker -> high risk")

    # young, not overweight, non-smoker - baseline low risk
    @Rule(PatientData(age=P(lambda x: x < 35), bmi=P(lambda x: x < 25), smoking='no'))
    def young_healthy_nonsmoker(self):
        print("Age < 35 + healthy BMI + non-smoker -> low risk")