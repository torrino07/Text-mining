""" Libraries """

## Parsing libs ##
import html
import pprint
import re
from html.parser import HTMLParser

## Training and vectorization libs ##
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

## ML Models libs ##
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

## Output libs ##
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
"""           """

""" Global Variables """
global Word, Targets

Word = 'crude'
Targets1 = ['others','crude']
Targets2 = ['acq','corn','crude','earn','grain','interest','money-fx','ship','trade','wheat']
"""                  """

""" Paths """
filepath1 = ["/Users/giuliano/Desktop/DSP/Data/sgm_Files/reut2-%03d.sgm" % r for r in range(0, 22)]
filepath2 = "/Users/giuliano/Desktop/DSP/Data/all-topics-strings.lc.txt"
#filepath2 = "/Users/giuliano/Desktop/DSP/Data/topics.txt"
"""       """


""" Methods """

class ReutersParser(HTMLParser):
    """
    ReutersParser subclasses HTMLParser and is used to open the SGML
    files associated with the Reuters-21578 categorised test collection.

    The parser is a generator and will yield a single document at a time.
    Since the data will be chunked on parsing, it is necessary to keep 
    some internal state of when tags have been "entered" and "exited".
    Hence the in_body, in_topics and in_topic_d boolean members.
    """
    def __init__(self, encoding= 'latin-1'):
        """
        Initialise the superclass (HTMLParser) and reset the parser.
        Sets the encoding of the SGML files by default to latin-1.
        """
        html.parser.HTMLParser.__init__(self)
        self._reset()
        self.encoding = encoding

    def _reset(self):
        """
        This is called only on initialisation of the parser class
        and when a new topic-body tuple has been generated. It
        resets all off the state so that a new tuple can be subsequently
        generated.
        """
        self.in_body = False
        self.in_topics = False
        self.in_topic_d = False
        self.body = ""
        self.topics = []
        self.topic_d = ""

    def parse(self, fd):
        """
        parse accepts a file descriptor and loads the data in chunks
        in order to minimise memory usage. It then yields new documents
        as they are parsed.
        """
        self.docs = []
        for chunk in fd:
            self.feed(chunk.decode(self.encoding))
            for doc in self.docs:
                yield doc
            self.docs = []
        self.close()

    def handle_starttag(self, tag, attrs):
        """
        This method is used to determine what to do when the parser
        comes across a particular tag of type "tag". In this instance
        we simply set the internal state booleans to True if that particular
        tag has been found.
        """
        if tag == "reuters":
            pass
        elif tag == "body":
            self.in_body = True
        elif tag == "topics":
            self.in_topics = True
        elif tag == "d":
            self.in_topic_d = True 

    def handle_endtag(self, tag):
        """
        This method is used to determine what to do when the parser
        finishes with a particular tag of type "tag". 

        If the tag is a <REUTERS> tag, then we remove all 
        white-space with a regular expression and then append the 
        topic-body tuple.

        If the tag is a <BODY> or <TOPICS> tag then we simply set
        the internal state to False for these booleans, respectively.

        If the tag is a <D> tag (found within a <TOPICS> tag), then we
        append the particular topic to the "topics" list and 
        finally reset it.
        """
        if tag == "reuters":
            self.body = re.sub(r'\s+', r' ', self.body)
            self.docs.append( (self.topics, self.body) )
            self._reset()
        elif tag == "body":
            self.in_body = False
        elif tag == "topics":
            self.in_topics = False
        elif tag == "d":
            self.in_topic_d = False
            self.topics.append(self.topic_d)
            self.topic_d = ""  

    def handle_data(self, data):
        """
        The data is simply appended to the appropriate member state
        for that particular tag, up until the end closing tag appears.
        """
        if self.in_body:
            self.body += data
        elif self.in_topic_d:
            self.topic_d += data


def obtain_topic_tags(filepath):
    """
    Open the topic list file and import all of the topic names
    taking care to strip the trailing "\n" from each word.
    """
    topics = open(filepath, "r").readlines()
    topics = [t.strip() for t in topics]
    return topics


def filter_doc_list_through_topics(topics, docs):
    """
    Reads all of the documents and creates a new list of two-tuples
    that contain a single feature entry and the body text, instead of
    a list of topics. It removes all geographic features and only 
    retains those documents which have at least one non-geographic
    topic.
    """
    ref_docs = []
    for d in docs:
        if d[0] == [] or d[0] == "":
            continue
        for t in d[0]:
            if t in topics:
                d_tup = (t, d[1])
                ref_docs.append(d_tup)
                break
    return ref_docs


def create_tfidf_training_data(docs):
    """
    Creates a document corpus list (by stripping out the
    class labels), then applies the TF-IDF transform to this
    list. 

    The function returns both the class label vector (y) and 
    the corpus token/feature matrix (X).
    """
    # Create the training data class labels
    y = [d[0] for d in docs]

    Y = Dummy(Word,y)
    S = y

    # Create the document corpus list
    corpus = [d[1] for d in docs]


    # Create the TF-IDF vectoriser and transform the corpus
    vectorizer = TfidfVectorizer(min_df=1)
    X = vectorizer.fit_transform(corpus)
    return X, Y, S

def Dummy(word,y):
    """
    It creates a dummy variable equal to one if i == 'crude', 0 otherwise
    """
    Dummy = []
    for i in y:
        if i == word:
            Dummy.append(1)
        else:
            Dummy.append(0)

    return Dummy


def values(y):
    """
    Assigns values to unique topics labels
    """
    d = {i: k for k, i in enumerate(set(y))}
    values = [d[i] for i in y]
    return values

def train_kkn(X, y):
    """
    Create and train the Naive Bayes.
    """
    kkn =  KNeighborsClassifier(n_neighbors = 3)
    kkn.fit(X, y)
    return kkn

def train_lr(X,y):
    """
    Create and train the Logistic regression.
    """
    lg = LogisticRegression(solver='lbfgs')
    lg.fit(X,y)

    return lg

def train_svm(X, y):
    """
    Create and train the Support Vector Machine.
    """
    svm = SVC(C=1000000.0, gamma='auto', kernel='rbf')
    svm.fit(X, y)
    return svm

def Reader(filepath):
    """
    Parse the document and force all generated docs into
    a list so that it can be printed out to the console
    """
    docs = []
    parser = ReutersParser()
    for i in filepath:
        for j in parser.parse(open(i, 'rb')):
            docs.append(j)

    return docs

def fr(Y):

    n = len(Y)
    ones = Y.count(1)
    zeros = Y.count(0)
    Frequency = [ones/n,zeros/n]

    return Frequency

def BinomialCategories1(Y):

    Actual = fr(Y)

    my_labels = 'Crude','Others'
    my_colors = ['lightsalmon','indianred']
    my_explode = (0, 0.1)

    plt.pie(Actual, labels=my_labels, autopct='%1.1f%%', startangle=15, shadow = True, colors=my_colors, explode=my_explode)
    plt.title('Crude Versus other topics')
    plt.axis('equal')
    plt.show()

    plt.hist(Y,rwidth=0.8,color='darkred',density=True)
    plt.ylabel('Frequency')
    plt.title('Other topics vs Crude', fontsize=10)
    plt.show()

def BinomialCategories2(Y,ypred1,ypred2,ypred3):

    Actual = fr(Y)
    Prediction1 = fr(ypred1)
    Prediction2 = fr(ypred2)
    Prediction3 = fr(ypred3)

    objects = ('Crude','Others')
    width1, width2, width3 = 0.20, 0.25, 0.30

    
    plt.bar(np.arange(len(Actual)),Actual, color='darkred', width=width1, label='Actual')
    plt.bar(np.arange(len(Prediction1))+ width1, Prediction1, color='darksalmon', width=width1, label='Prediction via k-NN ')
    plt.ylabel('Frequency')
    plt.title('Crude vs Other  topics')
    plt.legend()
    plt.show()

    plt.bar(np.arange(len(Actual)),Actual, color='darkred', width=width1, label='Actual')
    plt.bar(np.arange(len(Prediction2))+ width1, Prediction2, color='salmon', width=width1, label='Prediction via Logistic Regression')
    plt.ylabel('Frequency')
    plt.title('Crude vs Other  topics')
    plt.legend()
    plt.show()

    plt.bar(np.arange(len(Actual)),Actual, color='darkred', width=width1, label='Actual')
    plt.bar(np.arange(len(Prediction3))+ width1, Prediction3, color='lightsalmon', width=width1, label='Prediction via Support vector machine')
    plt.ylabel('Frequency')
    plt.title('Crude vs Other  topics')
    plt.legend()
    plt.show()


    my_labels = 'Crude','Others'
    my_colors1 = ['darkred','indianred']
    my_colors2 = ['salmon','lightsalmon']
    my_explode = (0, 0.1)
    
    plt.suptitle('Crude Versus other topics')

    plt.subplot(1,4,1)
    plt.title("Actual")
    plt.pie(Actual, labels=my_labels, autopct='%1.1f%%', startangle=15, shadow = True, colors=my_colors1, explode=my_explode)

    plt.subplot(1,4,2)
    plt.title("k-NN ")
    plt.pie(Prediction1, labels=my_labels, autopct='%1.1f%%', startangle=15, shadow = True, colors=my_colors2, explode=my_explode)

    plt.subplot(1,4,3)
    plt.title("Logistic Regression")
    plt.pie(Prediction2, labels=my_labels, autopct='%1.1f%%', startangle=15, shadow = True, colors=my_colors2, explode=my_explode)

    plt.subplot(1,4,4)
    plt.title("SVM")
    plt.pie(Prediction3, labels=my_labels, autopct='%1.1f%%', startangle=15, shadow = True, colors=my_colors2, explode=my_explode)
    plt.show()

    plt.hist([Y,ypred1], rwidth=0.8,color=['darkred','salmon'],density=True,label=['Actual','k-NN'])
    plt.title('Histogram: Crude vs other topics', fontsize=10)
    plt.legend(loc='upper right')
    plt.show()

    plt.hist([Y,ypred2], rwidth=0.8,color=['darkred','salmon'],density=True,label=['Actual','Logistic regression'])
    plt.title('Histogram: Crude vs other topics', fontsize=10)
    plt.legend(loc='upper right')
    plt.show()

    plt.hist([Y,ypred3], rwidth=0.8,color=['darkred','salmon'],density=True,label=['Actual',' SVM'])
    plt.title('Histogram: Crude vs other topics', fontsize=10)
    plt.legend(loc='upper right')
    plt.show()


def counter(S):

    numTargets2 = []
    for i in Targets2:
        numTargets2.append(S.count(i))

    return numTargets2

def filter(S):

    newListc = []
    for i in S:
        for j in Targets2:
            if i == j:
                newListc.append(i)
    return newListc

def frequency(numTargets2,n):


    Frequencies = []
    for i in numTargets2:
        Frequencies.append(i/n)

    return Frequencies


def MultipleModel(S):

    S = filter(S)
    print(S)

    n = len(S)

    numTargets2 = counter(S)

    Frequencies = frequency(numTargets2,n)
  

    objects = ('acq','corn','crude','earn','grain','interest','money-fx','ship','trade','wheat')
    y_pos = np.arange(len(objects))

    plt.bar(y_pos, Frequencies, align='center', alpha=0.5, color='darkred')
    plt.xticks(y_pos, objects)
    plt.ylabel('Frequency')
    plt.title('Topics')
    plt.show()
    
    plt.hist(S ,rwidth=0.8,color='darkred',density=True)
    plt.title('Topics')
    plt.ylabel('Frequency')
    plt.show()

    print(numTargets2)
    print(Frequencies)


def TrainTestSet(filepath1,filepath2):
    """ 
    Obtain yest and train data for both regressor 
    matrix and dependent variable
    """

    docs = Reader(filepath1)

    topics = obtain_topic_tags(filepath2)
    ref_docs = filter_doc_list_through_topics(topics, docs)
 
    X, Y, S = create_tfidf_training_data(ref_docs)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, Y,S

def Output(filepath1,filepath2):

    X_train, X_test, y_train, y_test, Y, S= TrainTestSet(filepath1,filepath2)
    # Create and train the Logistic regression
    kkn = train_kkn(X_train, y_train)
    # Make an array of predictions on the test set
    y_pred1 = kkn.predict(X_test)
    # Output the hit-rate and the confusion matrix for each model
    Score1 = kkn.score(X_test, y_test)
    CM1 = confusion_matrix(y_test, y_pred1)
    print("Accuracy: {}".format(Score1))
    print(CM1)
    print(classification_report(y_test, y_pred1, target_names=Targets1))


    # Create and train the Naive bayes
    lg = train_lr(X_train, y_train)
    # Make an array of predictions on the test set
    y_pred2 = lg.predict(X_test)
    # Output the hit-rate and the confusion matrix for each model
    Score2 = lg.score(X_test, y_test)
    CM2 = confusion_matrix(y_test, y_pred2)
    print("Accuracy: {}".format(Score2))
    print(CM2)
    print(classification_report(y_test, y_pred2, target_names=Targets1))
  
    # Create and train the Support Vector Machine
    svm = train_svm(X_train, y_train)
    # Make an array of predictions on the test set
    y_pred3 = svm.predict(X_test)
    # Output the hit-rate and the confusion matrix for each model
    Score3 = svm.score(X_test, y_test)
    CM3 = confusion_matrix(y_test, y_pred3)
    print("Accuracy: {}".format(Score3))
    print(CM3)
    print(classification_report(y_test, y_pred3, target_names=Targets1))
   
    

    MultipleModel(S)

    BinomialCategories1(y_test)

    BinomialCategories2(y_test,y_pred1.tolist(),y_pred2.tolist(),y_pred3.tolist())
    

"""                  """

""" Main """
def main(filepath1,filepath2):

    # Initialization
    
    # Estimation
    Output(filepath1,filepath2)

"""       """

if __name__ == "__main__":
    main(filepath1,filepath2)
