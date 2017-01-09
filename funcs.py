import math
import filemapper as fm
import string
import operator
from collections import defaultdict
import Queue as queue
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import unicodedata



class ArticleScorer:
    """
        A class that takes a document and returns a dictionary with 
        all its stemmed words with corresponding tf-idf score.
    """

    def __init__(self, infile, allDocs):
        # dictionary of the words and tf - idf computations
        self.word_indices = {}

        # list that holds all the words in the article you are interested
        self.infile = infile.split()

        # list of strings of all the documents
        self.allDocs = allDocs


    # returns true if term is in the doc (string), returns false otherwise
    def stringInDoc(self, term, doc):
        if term in doc:
            return True
        else:
            return False


    # returns frequency of a term in document
    def tf(self, term):
        return float(self.infile.count(term))


    # returns inverse document frequency
    def idf(self, term):
        count = 0
        for i in self.allDocs:
            if self.stringInDoc(term, i):
                count = count + 1
#        print(count)
        return 1 + math.log((1 + len(self.allDocs))/float(1 + count))

    # return tf-idf product
    def product(self, term):
        return self.tf(term)*self.idf(term)

    # return dictionary with top 5 tagging words
    def represent(self):
        for word in self.infile:
            if word in self.word_indices.keys():
                continue
            else:
                self.word_indices[word] = self.product(word)
        normalizer = sum(self.word_indices.values())
        
        for key, value in self.word_indices.items():
            self.word_indices[key] = value / normalizer
        
        sorted_x = sorted(self.word_indices.items(), key=operator.itemgetter(1))
        sorted_x.reverse()
        # self.word_indices = dict(sorted_x)
        self.word_indices = dict(sorted_x)

        # print "allDocs", self.allDocs

        return self.word_indices

    def cosine(self, doc):
        pass


# removes all occurrences of a substring from a string
def remove_all(substr, str):
    index = 0
    length = len(substr)
    while string.find(str, substr) != -1:
        index = string.find(str, substr)
        str = str[0:index] + ' ' + str[index+length:]
    return str

# finds the cosine between two documents and similaritites
def cosine(doc_and_sim_1, doc_and_sim_2):
    dict1 = doc_and_sim_1[1]
    dict2 = doc_and_sim_2[1]
    similarity = 0
    for key in dict1:
        if key in dict2:
            similarity += dict1[key] * dict2[key]
    return similarity

def extract_title(docStr):
    print docStr[:docStr.find('  ')]

def main():
    #opens all the files in the folder
    all_files = fm.load('test')
    docs = []
    
    #instantiating Porter Stemming
    stemmer = PorterStemmer()


    for f in all_files:
        docStr = ""
        for i in fm.read(f): docStr += i
        # docStr.replace('\n', ' ')
        # docStr.replace('\\', ' ')
        # docStr.replace('.', ' ')
        # docStr.replace(',', ' ')
        docStr = remove_all("\n", docStr)
        docStr = remove_all("\\", docStr)
        docStr = remove_all('"', docStr)
        docStr = remove_all(':', docStr)
        docStr = remove_all(".", docStr)
        docStr = remove_all(",", docStr)
        
        #unicode encoding for stemming
        p = docStr.decode('cp850').replace(u"\u2019", u"\x27")
        res = " ".join([ stemmer.stem(kw) for kw in p.split(" ")])
        k = unicodedata.normalize('NFKD', res).encode('ascii','ignore')

        docs.append(k)



    docOfInterest = docs[5]

    #testing on article
    a = ArticleScorer(docOfInterest, docs)
    print(a.represent())


    docs_and_similarities = []

    for docOfInterest in docs:
        # print docOfInterest
        a = ArticleScorer(docOfInterest, docs)
        # print(a.represent())
        docs_and_similarities.append((docOfInterest, a.represent()))

        print docs_and_similarities

    best_matches = queue.PriorityQueue()
    article = docs_and_similarities[0]
    for other in docs_and_similarities[1:]:
        print "score", cosine(article, other)
        best_matches.put((-1 * cosine(article, other), other[0]))
    for i in xrange(5):
        print extract_title(best_matches.get()[1])

if __name__ == "__main__": main()
