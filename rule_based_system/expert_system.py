import collections
import collections.abc
collections.Mapping = collections.abc.Mapping
collections.Iterable = collections.abc.Iterable
collections.MutableMapping = collections.abc.MutableMapping
collections.Sequence = collections.abc.Sequence

from experta import Fact, KnowledgeEngine, Rule, P


class PatientData(Fact):
    pass


class HeartRules(KnowledgeEngine):
    
    @Rule(PatientData(age=P(lambda x: x > 50)))
    def ageRule(self):
        print("Risk: \n Old Age")

    @Rule(PatientData(cholesterol=P(lambda x: x > 240)))
    def cholRule(self):
        print("Risk: \n High cholesterol")

    @Rule(PatientData(blood_pressure=P(lambda x: x > 140)))
    def blopresRule(self):
        print("Risk: \n High Blood Pressure")

    @Rule(PatientData(smoking=P(lambda x: x in [True, 'yes', 'y'])))
    def smokingRule(self):
        print("Risk: \n Smoking")

    @Rule(PatientData(diabetes=P(lambda x: x in [True, 'yes', 'y'])))
    def diabetesRule(self):
        print("Risk: \n Diabetes")

    @Rule(PatientData(obesity=P(lambda x: x in [True, 'yes', 'y'])))
    def obesityRule(self):
        print("Risk: \n Obesity")

    @Rule(PatientData(exercise=P(lambda x: str(x).lower() in ['low', 'none', 'no', 'light'])))
    def exerRule(self):
        print("Risk: \n Low Exercise")
    
    @Rule(PatientData(chest_pain=P(lambda x: str(x).lower() in ['yes', 'typical', 'typical angina'])))
    def chesRule(self):
        print("Risk: \n Chest Pain")

    @Rule(PatientData(family_history=P(lambda x: x in [True, 'yes', 'y'])))
    def familyRule(self):
        print("Risk: \n Family History")

    @Rule(PatientData(rest_ecg=P(lambda x: str(x).lower() in ['abnormal', 'st-t wave', 'lv hypertrophy'])))
    def ecgRule(self):
        print("Risk: \n Abnormal ECG")


HeartDiseaseExpert = HeartRules