"""
Name: Hangliang Ren

This program implements the Multinomial Naive Bayes Classifier (MultinomialNB).
It creates MultinomialNB model based on text data.
After creating the model, it can predict which target the input text is belonged to.
"""
import re
from math import log


class MultinomialNB:
    def __init__(self):
        pass

    def train(self, features, targets):
        """
        parameters
            "features": A list of text data for training the model; for example,
            features = ["I love math", "A good game"].
            "targets": A list storing which category each feature belonged to;
            for example, targets = [1, 2], which means "I love math" is belonged
            to category "1", etc.
        post
            Build a MultinomialNB model for prediction, ignoring punctuation,
            with all words lowercase.
        """
        self.features = features
        self.targets = targets
        self.frequency_coll = self._frequency_calculate()
        self.num_unique_words = self._count_unique_words()
    
    def predict(self, features):
        """
        parameters
            "features": A list of text data to predict; for example,
            features = ["I pass the exam", "We fail the game"].
        post
            Return a list, storing the predicted target that each input feature
            belonged to.
        """
        predicted_targets = []
        
        for feature in features:
            results = dict()
            feature_words = self._text_extract(feature)
            
            for target in self.targets:
                log_probability = 0
                for word in feature_words:
                    # calculate log conditional probability with laplace smoothing
                    if word in self.frequency_coll[target]:
                        log_probability += log((self.frequency_coll[target][word] + 1) / (len(feature_words) + self.num_unique_words))
                    else:
                        log_probability += log((0 + 1) / (len(feature_words) + self.num_unique_words))
                results[target] = log_probability

            # predict which target the features is belonged to
            # the feature is belonged to the target with highest probability
            target_belonged = max(results, key=results.get)
            predicted_targets.append(target_belonged)
        
        return predicted_targets

    def _text_extract(self, sentence):
        """
        parameters
            "sentence": A sentence to process; for example,
            sentence = "I love math".
        post
            Return a list, in which each element is a word in input sentence,
            with all lowercase and all punctuation removed.
        """
        text_collect = sentence.split()
        results = []
        for text in text_collect:
            processed_text = re.compile(u'[\u4E00-\u9FA5|\s\w]').findall(text)
            processed_text = "".join(processed_text).lower()
            results.append(processed_text)
        
        return results
    
    def _frequency_calculate(self):
        """
        post
            Return a dictionary storing frequency of each word in training dataset.
        """
        frequency_coll = dict()

        for i in range(len(self.features)):
            feature = self.features[i]
            target = self.targets[i]
            feature_text = self._text_extract(feature)

            if target not in frequency_coll:
                frequency_coll[target] = dict()
            
            # calculate and store frequecy of each word
            for word in feature_text:
                if word in frequency_coll[target]:
                    frequency_coll[target][word] += 1
                else:
                    frequency_coll[target][word] = 1
        
        return frequency_coll
    
    def _count_unique_words(self):
        """
        post
            Return the total number of unique words in training dataset.
        """
        words_collect = []
        for target in self.frequency_coll.keys():
            words_collect += list(self.frequency_coll[target].keys())
        words_collect = tuple(words_collect)
        
        return len(words_collect)