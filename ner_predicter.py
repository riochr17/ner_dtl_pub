from sklearn.tree import DecisionTreeClassifier
import pickle
from ner_feature_extractor import NERFeatureExtractor

class NERPredicter:
	def __init__(self, model_filename = 'decision_tree_classifier_model_ner.pkl'):
		decision_tree_pkl_filename = model_filename
		decision_tree_model_pkl = open(decision_tree_pkl_filename, 'rb')
		self.dtc = pickle.load(decision_tree_model_pkl)
		self.ner_predictor = NERFeatureExtractor(self.dtc.predict)

	def predict(self, sent):
		self.ner_predictor.parseEntityName(sent)
		return self.ner_predictor.res_all
