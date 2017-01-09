import os, sys
from random import randint, sample
from time import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy import sparse
from numpy import mean

reload(sys)
sys.setdefaultencoding('ISO-8859-1')

# For formatting output
COLOR_BOLD = '\033[1m'
COLOR_RED = '\033[91m'
COLOR_GREEN = '\033[92m'
COLOR_CYAN = '\033[96m'
COLOR_DARKCYAN = '\033[36m'
COLOR_END = '\033[0m'

class Recommender:
	"""
	A class that reads documents from a corpus and provides recommendations for 
	future articles to read. Represents documents as vectors found through TF-IDF
	to allow for classifcation, and incorporates the Scikit-learn library for 
	efficient data stroage.

	The similarity recommendation method finds the closest articles using 
	similarity indices. 

	The clustering method clusters the documents and evaluates the clustering.

	The clustering recommendation method recommends new articles based on the
	predicted cluster of a given article.
	"""
	def __init__(self):
		self.article_list = []
		self.categories = ['business', 'entertainment', 'politics', 'sport', 'tech']
		self.true_article_categories = []
		self.num_categories = len(self.categories)

		# Loops through every category file, adding the document bodies to the
		# list of articles and indexing which category the documents belong to
		# in the true_article_categories list
		category_index = 0
		for category in self.categories:
			directory = os.getcwd() + '/bbc-articles/' + category + '/'
			files = os.listdir(directory)
			articles = [file for file in files if file.endswith('.txt')]
			for article in articles:
				with open(directory + article) as f:
					content = f.read().encode('utf-8').strip()
					self.article_list.append(content)
					self.true_article_categories.append(category_index)
			category_index += 1
		self.num_articles = len(self.article_list)
			
		# Uses the Scikit-learn TF-IDF vectorizer to process the articles
		self.vectorizer = TfidfVectorizer()
		self.vectors = self.vectorizer.fit_transform(self.article_list)

		# Sets up a k-means classifer with the number of clusters set to the 
		# number of categories
		self.kmeans = KMeans(n_clusters = self.num_categories, init = 'k-means++')

	def get_title(self, article):
		""" 
		Returns the contents of an article up to the first newline 
		"""
		return article[0:article.find('\n')]

	def cosine_similarity(self, vector1, vector2):
		""" 
		Finds the dot product of two vectors 
		"""
		return vector1.dot(vector2.transpose())[0,0]

	def jaccard_similarity(self, vector1, vector2):
		"""
		Finds the jaccard similarity of two vectors, or the magnitude of
		the intersection over the magnitude of the union
		"""
		tokens1 = sparse.find(vector1)[1] # Finds the indices of nonzero elements
		tokens2 = sparse.find(vector2)[1] # Finds the indices of nonzero elements

		intersection_size = 0
		for token in tokens1:
			if token in tokens2:
				intersection_size += 1
		if len(tokens1) == 0:
			return 0
		else:
			return float(intersection_size) / (len(tokens1) + len(tokens2) - intersection_size)

	def generate_similarity_recommendations(self, num_trials=5, similarity="cosine", start_index=None, verbose=True):
		"""
		For a set number of trials, pick an unread article and find articles
		that are closest to that article as measured by a similarity index
		"""
		if start_index is None:
			start_index = randint(0, self.num_articles - 1)

		if similarity == "jaccard":
			similarity = self.jaccard_similarity
		else:
			similarity = self.cosine_similarity

		runtimes = []

		current_index = start_index

		if verbose:
			print COLOR_BOLD + COLOR_RED + "Calculating Most Similar Articles..."
			if similarity == self.cosine_similarity:
				print "Using Cosine Similarity..."
			else:
				print "Using Jaccard Similarity..."
			print "" + COLOR_END

		for trial in xrange(1, num_trials + 1):
			t0 = time()
			if verbose:
				print COLOR_BOLD + "Trial " + str(trial) + COLOR_END
			current_vector = self.vectors[current_index]

			# Creates a list of article indices and similarities, and sorts
			# bases on the highest level of similarity
			indexed_similarities = []
			for i in xrange(self.num_articles):
				if i != current_index:
					other_vector = self.vectors[i]
					similarity_index = similarity(current_vector, other_vector)
					if similarity_index < .999: # if not a duplicate in the dataset
						indexed_similarities.append((i, similarity_index))
			indexed_similarities.sort(key=lambda (x,y): y, reverse=True)

			# Prints the original article and the top 5 most similar articles
			if verbose:
				print COLOR_DARKCYAN + "Original Article:"
				print self.get_title(self.article_list[current_index])
				print "" + COLOR_END
				print COLOR_GREEN + "Recommended Articles:"
				for other_index, score in indexed_similarities[:5]:
					print self.get_title(self.article_list[other_index]), "-", 
					print "%.3f" % similarity(current_vector, self.vectors[other_index])
				print "" + COLOR_END
			end_time = time() - t0
			runtimes.append(end_time)
			if verbose:
				print "Time:", "%.3f" % end_time + "s"
				print ""

			# Generates a new article to examine
			current_index = randint(0, self.num_articles - 1)

		print "Mean time to generate:", "%.3f" % mean(runtimes) + "s"

	def generate_clusters(self, num_trials = 1, verbose=True):
		"""
		Fits the articles using a k-means classifier and evalutes the clustering
		"""
		runtimes = []
		homogeneities = []
		completenesses = []
		vmeasures = []

		for _ in xrange(num_trials):
			if verbose:
				print COLOR_BOLD + COLOR_RED + "Clustering Articles Using k-means..." + COLOR_END
			t0 = time()
			self.kmeans.fit(self.vectors)
			labels = self.kmeans.labels_

			if verbose:
				print COLOR_GREEN + COLOR_BOLD + "Clusters Generated" 
				print "" + COLOR_END
				print COLOR_BOLD + "Cluster Fit Metrics:" + COLOR_END

			homogeneity = metrics.homogeneity_score(self.true_article_categories, labels)
			homogeneities.append(homogeneity)
			if verbose:
				print "Homogeneity:", "%0.3f" % homogeneity 

			completeness = metrics.completeness_score(self.true_article_categories, labels) 
			completenesses.append(completeness)
			if verbose:
				print "Completeness:", "%0.3f" % completeness
			
			vmeasure = metrics.v_measure_score(self.true_article_categories, labels) 
			vmeasures.append(vmeasure)
			if verbose:
				print "V-measure:", "%0.3f" % vmeasure
				print ""
			
			end_time = time() - t0
			runtimes.append(end_time)
			if verbose:
				print "Time:", "%.3f" % end_time + "s"
				print ""

			if verbose:
				# Creates a data structure that stores the indices of documents
				# classified in each cluster
				cluster_indices = [set() for i in xrange(self.num_categories)]
				for i in xrange(len(labels)):
					cluster = labels[i]
					cluster_indices[cluster].add(i)
			
				print COLOR_GREEN + COLOR_BOLD + "Sample Articles From Each Cluster:" + COLOR_END
				for i in xrange(len(cluster_indices)):
					print COLOR_DARKCYAN + COLOR_BOLD + "Cluster %s:" % (i + 1) + COLOR_END + COLOR_DARKCYAN
					for j in sample(cluster_indices[i], 4):
						print self.get_title(self.article_list[j])
					print "" + COLOR_END

		if num_trials > 1:
			print "Mean time to cluster:", "%.3f" % mean(runtimes) + "s"
			print "Mean homogeneity:", "%.3f" % mean(homogeneities) + "s"
			print "Mean completeness:", "%.3f" % mean(completenesses) + "s"
			print "Mean v-measure:", "%.3f" % mean(vmeasures) + "s"

	def generate_cluster_recommendations(self, num_trials=5, start_index=None, verbose=True):
		"""
		For a set number of trials, pick an unread article, predict its cluster
		from the trained k-means, and sample articles from the cluster for
		recommendation
		"""
		if start_index is None:
			start_index = randint(0, self.num_articles - 1)

		runtimes = []

		if verbose:
			print COLOR_BOLD + COLOR_RED + "Clustering Articles Using k-means..." + COLOR_END

		t0 = time()
		self.kmeans.fit(self.vectors)
		print "Time to cluster:", "%.3f" % (time() - t0) + "s"

		labels = self.kmeans.labels_
		current_index = start_index
		if verbose:
			print COLOR_GREEN + COLOR_BOLD + "Clusters Generated" 
			print "" + COLOR_END

		for trial in xrange(1, num_trials + 1):
			t0 = time()
			if verbose:
				print COLOR_BOLD + "Trial " + str(trial) + COLOR_END
			assigned_cluster = self.kmeans.predict(self.vectors[current_index])

			# Finds indices for articles in the predicted cluster
			assigned_cluster_elements = set()
			for i in xrange(len(labels)):
				if labels[i] == assigned_cluster:
					assigned_cluster_elements.add(i)

			if verbose:
				print COLOR_DARKCYAN + "Original Article:"
				print self.get_title(self.article_list[current_index]) + " - " +\
						str(self.categories[self.true_article_categories[current_index]])
				print "" + COLOR_END

				# Samples the predicted cluster for new articles
				print COLOR_GREEN + "Recommended Articles:"

			for i in sample(assigned_cluster_elements, 4):
				if verbose:
					print self.get_title(self.article_list[i])
					print "" + COLOR_END
				else:
					self.get_title(self.article_list[i])

			end_time = time() - t0
			runtimes.append(end_time)

			if verbose:
				print "Time to recommend:", "%.3f" % end_time + "s"
				print ""

			# Generates a new article to examine
			current_index = randint(0, self.num_articles - 1)

		print "Mean time to recommend:", "%.3f" % mean(runtimes) + "s"

def main():
	recommender = Recommender()
	if len(sys.argv) < 2 or sys.argv[1] == "-s":
		recommender.generate_similarity_recommendations(similarity = "jaccard")
	elif sys.argv[1] == "-k":
		recommender.generate_clusters()
	elif sys.argv[1] == "-c":
		recommender.generate_cluster_recommendations()
	elif sys.argv[1] == "-t":
		num_trials = 100
		print "Evaluating Approaches..."
		print "Cosine Similarity Recommendations:"
		recommender.generate_similarity_recommendations(verbose = False, num_trials = num_trials)
		print "Jaccard Similarity Recommendations:"
		recommender.generate_similarity_recommendations(similarity = "jaccard", verbose = False, num_trials = num_trials)
		print "Clustered Recommendations:"
		recommender.generate_cluster_recommendations(verbose = False, num_trials = num_trials)
		print "Number of trials:", num_trials
	elif sys.argv[1] == "-e":
		num_trials = 20
		print "Clustering Evaluation..."
		recommender.generate_clusters(verbose = False, num_trials = num_trials)
		print "Number of trials:", num_trials
	else:
		print \
"News Article Recommender \n\
Usage: python final_project.py [options] \n\
	options: \n\
	-s Generates article recommendations using similarity indices \n\
	-k Clusters the articles using k-means and evalues the clusterings \n\
	-c Generates article recommendations using clustering \n\
	-t Evaluates runtimes of similarity and clustering recommendations \n\
	-e Evaluates the clustering by using standard entropy measures \n\
	-h Show this help message"

if __name__ == "__main__": main()

