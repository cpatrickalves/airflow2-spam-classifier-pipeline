# AntiSpam Naive bayes Classifier
# This code was developed in the Data Scientist Training (Formação Cientista de Dados) of Data Science Academy (https://www.datascienceacademy.com.br)

# Imports
import sys, os
import warnings
from math import log
import json
import numpy as np
import logging as logger


class NaiveBayesSolver:

    spam_counts = {}
    notspam_counts = {}

    class_counts = {}
    model_vocabulary_size = {}

    def build_word_counts_model(self, files_path):
        """
        This will build a word count model, which will count the word occurrence within documents
        and build class counts from training data.

        There will be two types of models
        1) Binary - just indicating presence or absence of words in doc.
        2) Continuous - In this model, the number of times the word occurs in the doc (frequency) is counted.
        """

        logger.info("Building word count model ...")

        dirs = os.listdir(files_path)
        for class_dir_name in dirs:
            logger.info(f"Processin {class_dir_name} class ...")

            files_to_process = os.listdir(os.path.join(files_path, class_dir_name))
            if not files_to_process:
                logger.error(f"No files found in {os.path.join(files_path, class_dir_name)}")
                return False

            for f in files_to_process:
                document = os.path.join(files_path, class_dir_name, f)
                logger.info(f"Processing file {document}")

                with open(document, 'r', encoding = "latin1") as file:
                    words = file.read().split()
                    distinct_words = sorted(set(words))
                    for word in distinct_words:

                        self.class_counts[class_dir_name]['word_counts'][word]['frequency_count'] = \
                            self.class_counts\
                                .setdefault(class_dir_name,{})\
                                .setdefault('word_counts',{})\
                                .setdefault(word, {})\
                                .setdefault('frequency_count', 0) + words.count(word)

                        self.class_counts[class_dir_name]['word_counts'][word]['presence_count'] = \
                            self.class_counts\
                                .setdefault(class_dir_name,{})\
                                .setdefault('word_counts',{})\
                                .setdefault(word, {})\
                                .setdefault('presence_count', 0) + 1

                    self.class_counts[class_dir_name]['total_count'] = \
                        self.class_counts\
                            .setdefault(class_dir_name, {})\
                            .setdefault('total_count', 0) + 1
                    pass
            pass

        spam_word_counts = self.class_counts["spam"]['word_counts']
        logger.info("Top 10 words most associated with spam: {spam_word_counts}")

        # To get the words that are least associated with spam, we first get the words "not spam"
        notspam_word_counts = self.class_counts["notspam"]['word_counts']
        least_associated_with_spam = {k: v for k, v in notspam_word_counts.items() if k not in spam_word_counts}

        return True


    def train(self, files_path, model_file):

        logger.info("Training the Naive Bayes Algorithm ...")
        status = self.build_word_counts_model(files_path)
        if status:
            self.save_model_to_file(model_file)

    def save_model_to_file(self, file_name):

        logger.info(f"Saving the model to the specified location '{file_name}'")
        if not os.path.exists(os.path.dirname(file_name)):
            os.makedirs(os.path.dirname(file_name))

        with open(file_name, 'w+') as filehandle:
            json.dump({'class_counts': self.class_counts},
                      filehandle, sort_keys=True, indent=4, ensure_ascii=False)


    def load_model_from_file(self, file_name):

        logger.info(f"Loading model '{file_name} ...'")
        try:
            with open(file_name, 'r', encoding = "latin1") as filehandle:
                model = json.load(filehandle, encoding = "ISO-8859-1")
                self.class_counts = model['class_counts']
            self.model_vocabulary_size = len(self.class_counts.get('spam').get('word_counts')) \
                                        + len(self.class_counts.get('notspam').get('word_counts'))
            return True

        except Exception as e:
            logger.exception("Error loading the model")
            return False

    def get_word_presence_class_log_prob(self, word, output_class):
        # Added laplace smoothing, to avoid problems due to invisible words
        # We will get some occasional warnings due to unicode decoding, which functionally does not cause any problems in the system
        # and therefore can be ignored
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return log(float(self.class_counts[output_class]['word_counts'].get(word, {}).get('presence_count', 0) + 1.0) / \
                       (self.class_counts[output_class]['total_count'] + self.model_vocabulary_size))

    def get_word_frequency_class_log_prob(self, word, output_class):

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return log(float(self.class_counts[output_class]['word_counts'].get(word,{}).get('frequency_count',0) + 1.0)/\
                (self.class_counts[output_class]['total_count'] + self.model_vocabulary_size))

    def get_class_prob(self, output_class):
        total_count = 0
        for key in self.class_counts.keys():
            total_count = total_count + self.class_counts[key]['total_count']
        return log(float(self.class_counts[output_class]['total_count'])/\
            (total_count))

    def predict_simple(self, document):
        with open(document, 'r', encoding = "latin1") as f:
            contents = f.read()
            words = set(contents.split())
            max_prob = -sys.maxsize
            max_class = None
            for output_class in ('spam', 'notspam'):
                p = self.get_class_prob(output_class)
                for word in words:
                    p = p + self.get_word_presence_class_log_prob(word, output_class)
                if p > max_prob:
                    max_prob = p
                    max_class = output_class
            return max_class

    def predict_with_word_frequencies(self, document):
        with open(document, 'r', encoding = "latin1") as f:
            contents = f.read()
            words = contents.split()
            max_prob = -sys.maxsize
            max_class = None
            for output_class in ('spam', 'notspam'):
                p = self.get_class_prob(output_class)
                for word in words:
                    p = p + self.get_word_frequency_class_log_prob(word, output_class)
                if p > max_prob:
                    max_prob = p
                    max_class = output_class
            return max_class

    def predict(self, files_path, model_file):
        status = self.load_model_from_file(model_file)
        if status:
            dirs = os.listdir(files_path)
            logger.info("Making predictions with the presence of words in the model ...")
            for class_dir_name in dirs:
                total_test_cases = 0
                correct_predictions = 0
                for f in os.listdir(os.path.join(files_path, class_dir_name)):
                    document = os.path.join(files_path, class_dir_name, f)
                    total_test_cases += 1
                    predicted_class = self.predict_simple(document)
                    if predicted_class == class_dir_name:
                        correct_predictions += 1
                logger.info("Predicting accuracy for the class: %s " % class_dir_name)
                logger.info("Total observations analyzed: %d  " % (total_test_cases))
                logger.info("Correctly classified observations: %d  " % (correct_predictions))
                logger.info("Accuracy: %.2f  " % (float(correct_predictions) / total_test_cases))


            logger.info("Making predictions with the frequency of words model ...")
            for class_dir_name in dirs:
                total_test_cases = 0
                correct_predictions = 0
                for f in os.listdir(os.path.join(files_path, class_dir_name)):
                    document = os.path.join(files_path, class_dir_name, f)
                    total_test_cases += 1
                    predicted_class = self.predict_with_word_frequencies(document)
                    if predicted_class == class_dir_name:
                        correct_predictions += 1
                logger.info("Predicting accuracy for the class: %s " % class_dir_name)
                logger.info("Total observations analyzed: %d  " % (total_test_cases))
                logger.info("Correctly classified observations: %d  " % (correct_predictions))
                logger.info("Accuracy: %.2f " % (float(correct_predictions) / total_test_cases))


if __name__ == "__main__":
    nb = NaiveBayesSolver()

    dataset_directory = sys.argv[1]
    model_file = "models/testmodel"

    nb.train(os.path.join(dataset_directory, "train"), model_file)
    nb.predict(os.path.join(dataset_directory, "test"), model_file)