from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

import string
import re
from nltk.tag import CRFTagger
try:
	import pycrfsuite
except ImportError:
	pass
	print "ga ketemu"

class NERFeatureExtractor:

	def read_label_file(self, filename):
		return open(filename).read().split('\n')

	def __init__(self, iob_predictor):
		self.iob_predictor = iob_predictor
		self.stemmer = StemmerFactory().create_stemmer()
		self.TAGGER3 = CRFTagger()
		self.TAGGER3.set_model_file('../ists/dataset/all_indo_man_tag_corpus_model.crf.tagger')
		self.label_words = self.read_label_file('label-words.txt')
		self.label_posses = self.read_label_file('label-posses.txt')
		self.label_lemmas = self.read_label_file('label-lemmas.txt')
		self.label_iob_feature = self.read_label_file('label-iob_feature.txt')
		self.label_iob_classes = self.read_label_file('label-iob_classes.txt')

	def getPOSTag(self, _temporary_tokens):
		strin = []
		for token_tag in _temporary_tokens:
			strin.append(unicode(token_tag.decode('utf-8')))

		return [(token.encode('ascii','ignore'), tag.encode('ascii','ignore')) for (token, tag) in self.TAGGER3.tag_sents([strin])[0]]

	def features(self, tokens, index, history):
		# print history
		# print tokens
		"""
		`tokens`  = a POS-tagged sentence [(w1, t1), ...]
		`index`   = the index of the token we want to extract features for
		`history` = the previous predicted IOB tags
		"""

		# Pad the sequence with placeholders
		tokens = [('[START2]', '[START2]'), ('[START1]', '[START1]')] + list(tokens) + [('[END1]', '[END1]'), ('[END2]', '[END2]')]
		history = ['[START2]', '[START1]'] + list(history)

		# shift the index with 2, to accommodate the padding
		index += 2

		word, pos = tokens[index]
		prevword, prevpos = tokens[index - 1]
		prevprevword, prevprevpos = tokens[index - 2]
		nextword, nextpos = tokens[index + 1]
		nextnextword, nextnextpos = tokens[index + 2]
		previob = history[index - 1]
		contains_dash = '-' in word
		contains_dot = '.' in word
		allascii = all([True for c in word if c in string.ascii_lowercase])

		allcaps = word == word.capitalize()
		capitalized = word[0] in string.ascii_uppercase

		prevallcaps = prevword == prevword.capitalize()
		prevcapitalized = prevword[0] in string.ascii_uppercase

		nextallcaps = prevword == prevword.capitalize()
		nextcapitalized = prevword[0] in string.ascii_uppercase

		return [word, str(self.stemmer.stem(word)), str(pos), str(allascii), str(nextword), str(self.stemmer.stem(nextword)), str(nextpos), str(nextnextword), str(nextnextpos), str(prevword), str(self.stemmer.stem(prevword)), str(prevpos), str(prevprevword), str(prevprevpos), str(previob), str(contains_dash), str(contains_dot), str(allcaps), str(capitalized), str(prevallcaps), str(prevcapitalized), str(nextallcaps), str(nextcapitalized)]

	def normalizeFeature(self, featx):
		out = []
		if featx[0] in self.label_words:
			out.append(self.label_words.index(featx[0]))
		else:
			out.append(-1)

		if featx[1] in self.label_lemmas:
			out.append(self.label_lemmas.index(featx[1]))
		else:
			out.append(-1)

		if featx[2] in self.label_posses:
			out.append(self.label_posses.index(featx[2]))
		else:
			out.append(-1)
	
		out.append(1 if featx[3] else 0)

		if featx[4] in self.label_words:
			out.append(self.label_words.index(featx[4]))
		else:
			out.append(-1)

		if featx[5] in self.label_lemmas:
			out.append(self.label_lemmas.index(featx[5]))
		else:
			out.append(-1)

		if featx[6] in self.label_posses:
			out.append(self.label_posses.index(featx[6]))
		else:
			out.append(-1)

		if featx[7] in self.label_words:
			out.append(self.label_words.index(featx[7]))
		else:
			out.append(-1)

		if featx[8] in self.label_posses:
			out.append(self.label_posses.index(featx[8]))
		else:
			out.append(-1)

		if featx[9] in self.label_words:
			out.append(self.label_words.index(featx[9]))
		else:
			out.append(-1)

		if featx[10] in self.label_lemmas:
			out.append(self.label_lemmas.index(featx[10]))
		else:
			out.append(-1)

		if featx[11] in self.label_posses:
			out.append(self.label_posses.index(featx[11]))
		else:
			out.append(-1)

		if featx[12] in self.label_words:
			out.append(self.label_words.index(featx[12]))
		else:
			out.append(-1)
		
		if featx[13] in self.label_posses:
			out.append(self.label_posses.index(featx[13]))
		else:
			out.append(-1)
		
		if featx[14] in self.label_iob_feature:
			out.append(self.label_iob_feature.index(featx[14]))
		else:
			out.append(-1)
	
		out.append(1 if featx[15] else 0)
		out.append(1 if featx[16] else 0)
		out.append(1 if featx[17] else 0)
		out.append(1 if featx[18] else 0)
		out.append(1 if featx[19] else 0)
		out.append(1 if featx[20] else 0)
		out.append(1 if featx[21] else 0)
		out.append(1 if featx[22] else 0)

		return out

	def parseEntityName(self, _sent = ""):
		tokens = self.getPOSTag(_sent.split())
		history = []
		self.res_all = []
		last_feature = []
		for i in range(len(tokens)):
			last_feature = self.features(tokens, i, history)
			iob_res = self.iob_predictor([self.normalizeFeature(last_feature)])[0]
			history.append(iob_res)
			self.res_all.append((tokens[i], self.label_iob_classes[iob_res]))
