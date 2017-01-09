from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer

# distinct_categories = ['comp.windows.x', 'rec.autos', 'sci.electronics', 'talk.politics.mideast']
# similar_categories = ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', ]
# trainset = fetch_20newsgroups(subset='train', categories=categories)

# vectorizer = TfidfVectorizer()
# vectors = vectorizer.fit_transform(trainset.data)

# vectors.shape

# for vector in vectors[1:]:
# 	if vector.dot(v1.transpose())[0,0] > .5:
# 		print(vector.dot(v1.transpose())[0,0])


import filemapper as fm
import os, glob
import sys
import scipy
import numpy as np

reload(sys)
sys.setdefaultencoding('ISO-8859-1')

class Recommender:

	def __init__(self):
		self.article_list = []
		self.true_categories = []
		self.categories = ['business', 'entertainment', 'politics', 'sport', 'tech']

		category_index = 0
		for category in self.categories:
			directory = os.getcwd() + '/bbc-2/' + category + '/'
			files = os.listdir(directory)
			articles = [file for file in files if file.endswith('.txt')]
			for article in articles:
				# print directory + article
				with open(directory + article) as f:
					# print directory + article
					content = f.read().encode('utf-8').strip()
					self.article_list.append(content)
					self.true_categories.append(category_index)
			category_index += 1
			
		vectorizer = TfidfVectorizer()
		self.vectors = vectorizer.fit_transform(self.article_list)

	def get_title(article):
		return article[0:article.find('\n')]

	def cosine_similarity(vector1, vector2):
		return vector1.dot(vector2.transpose())[0,0]

	def jaccard_similarity(vector1, vector2, length):
		intersection = 0
		# indicator1 = vector1 > 0
		# indicator2 = vector2 > 0
		# print "indicator1", indicator1
		# union = 0
		for i in xrange(length):
			value1 = vector1[0,i]
			value2 = vector2[0,i]
			# if value1 > 0 or value2 > 0:
			# 	union +=1
			if value1 > 0 and value2 > 0:
				intersection += 1
		# intersection = indicator1.dot(indicator2.transpose())[0,0]
		return (intersection) / (2 * length - intersection)

	def load(self):
		

# print self.article_list
# print self.true_categories

vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(self.article_list)

print vectors[0]
print "jaccard", jaccard_similarity(vectors[0], vectors[1], len(self.article_list))

print vectors.shape
v1 = vectors[0]
print "v1[0]", v1[0], type(v1[0])
print "v1[0][0,0]", v1[0][0,4349]

# derp
# for vector in vectors[1:]:
# 	if vector.dot(v1.transpose())[0,0] > .2:
# 		print(vector.dot(v1.transpose())[0,0])

print get_title(self.article_list[0])
# print vectors.nonzero()[0].shape()
max_index = len(self.article_list)
for i in xrange(1, max_index):
	vector = vectors[i]
	if cosine_similarity(vector, v1) > .2:
		print get_title(self.article_list[i])
		print(vector.dot(v1.transpose())[0,0])
	if jaccard_similarity(vector, v1, max_index) > .1:
		print "jaccard", jaccard_similarity(vector, v1, max_index)
# print article_list
# print true_categories
	# for file in glob.glob(directory + "*.txt"):
	# 	print file 
	# articles = fm.load(directory)

