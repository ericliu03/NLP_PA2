from __future__ import division

from corpus import Document, BlogsCorpus, NamesCorpus
from maximum_entropy import MaximumEntropy, MaximumEntropyWithPrior

import sys
from random import shuffle, seed
from unittest import TestCase, main, skip

class EvenOdd(Document):
    def features(self):
        """Is the data even or odd?"""
        return [self.data % 2 == 0]

class Name(Document):
    def features(self):
        name = self.data
        return [name[0].lower(), name[-1].lower()]

class NameNgrams(Document):
    def features(self):
        name = self.data
        return [name[0].lower()+name[1].lower(), name[-1].lower()]

class BagOfWords(Document):
    def features(self):
        return self.data.split()

class BagOfWordsImproved(Document):
    def features(self):
        table_no_punct = dict((ord(i), u'') for i in u'!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\t\n\x0b\x0c\r')
        split_data = self.data.split()
        n = 0
        while n <= (len(split_data) - 1):
            split_data[n] = split_data[n].lower().translate(table_no_punct)
            if split_data[n] == u'':
                del split_data[n]
                continue
            n += 1
        return split_data

class BagOfWordsImprovedForBernoulli(Document):
    def features(self):
        table_no_punct = dict((ord(i), u'') for i in u'!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\t\n\x0b\x0c\r')
        split_data = self.data.split()
        n = 0
        temp = {}
        while n <= (len(split_data) - 1):
            split_data[n] = split_data[n].lower().translate(table_no_punct)
            if (split_data[n] in temp) or split_data[n] == u'':
                del split_data[n]
                continue
            else:
                temp[split_data[n]] = True
            n += 1

        return split_data

def accuracy(classifier, test, verbose=sys.stderr):
    correct = [classifier.classify(x) == x.label for x in test]
    if verbose:
        print >> verbose, "%.2d%% " % (100 * sum(correct) / len(correct)),
    return sum(correct) / len(correct)

def measure(classifier, test, verbose=sys.stderr):
    label0 = test[0].label
    number_of_accuracy = 0
    tp = fp = fn = tn = 0
    results = [(classifier.classify(x), x.label) for x in test]
    for test, answer in results:
        if test == answer:
            number_of_accuracy += 1
            if answer == label0:
                tp += 1
            else:
                tn += 1
        else:
            if answer == label0:
                fn += 1
            else:
                fp += 1
    a = (number_of_accuracy / len(results))
    p = (tp/(tp+fp))
    r = (tp/(tp+fn))
    f1 = (2*p*r/(p+r))
    if verbose:
        print >> verbose, "accuracy: %.2f " % a\
                        , "precision: %.2f " % p\
                        , "recall: %.2f " % r\
                        , "F1: %.2f " % f1
    return (number_of_accuracy / len(results))


class MaximumEntropyTest(TestCase):
    u"""Tests for the Maximum Entropy classifier."""

    # def test_even_odd(self):
    #     """Classify numbers as even or odd"""
    #     classifier = MaximumEntropy()
    #     classifier.train([EvenOdd(0, True), EvenOdd(1, False)])
    #     test = [EvenOdd(i, i % 2 == 0) for i in range(2, 1000)]
    #     self.assertEqual(accuracy(classifier, test), 1.0)

    # def split_names_corpus(self, document_class=NameNgrams):
    #     """Split the names corpus into training and test sets"""
    #     names = NamesCorpus(document_class=document_class)
    #     self.assertEqual(len(names), 5001 + 2943) # see names/README
    #     seed(hash("names"))
    #     shuffle(names)
    #     return names[:6000], names[6000:]
    #
    # def test_names_nltk(self):
    #     """Classify names using NLTK features"""
    #     train, test = self.split_names_corpus()
    #     classifier = MaximumEntropy()
    #     classifier.train(train)
    #     self.assertGreater(accuracy(classifier, test), 0.70)

    def split_blogs_corpus(self, document_class):
        """Split the blog post corpus into training and test sets"""
        blogs = BlogsCorpus(document_class=document_class)
        # self.assertEqual(len(blogs), 3232)
        seed(hash("blogs"))
        shuffle(blogs)
        return blogs[:3000], blogs[3000:]

    def test_blogs_bag(self):
        """Classify blog authors using bag-of-words"""
        train, test = self.split_blogs_corpus(BagOfWordsImproved)
        classifier = MaximumEntropy(min_f=200, max_f=800)
        #classifier = MaximumEntropyWithPrior(min_f=200, max_f=800, sigma=0.05)
        classifier.train(train)
        measure(classifier, test)

if __name__ == '__main__':
    # Run all of the tests, print the results, and exit.
    main(verbosity=2)