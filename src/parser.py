__author__ = 'babak'

from bs4 import BeautifulSoup
import glob, os
import re
import numpy as np
import string
import nltk         #Natural Language Toolkit
from nltk.corpus import stopwords # Import the stop word list
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer

#test
class parser():
    def __init__(self,_dataPath):
        self.data_path = _dataPath

    # def parsing(self):
    #     print("parsing")
    #     os.chdir(self.data_path)
    #     products_dict={}
    #     for projectPage in glob.glob("*.html"):
    #         print(projectPage)
    #         soup = BeautifulSoup(open(projectPage),"html.parser")
    #
    #         try:
    #             availibility = soup.find("div",{"id":"availability"}).text
    #             product_name = soup.find("span", {"id": "productTitle"}).text
    #             if "in stock" or "instock" or "available" in availibility.lower() and "unavailable" not in availibility.lower()  :
    #                 products_dict[product_name]= "available"
    #             else:
    #                 products_dict[product_name]= "not available"
    #         except:
    #             try:
    #                 availibility = soup.find("meta",{"itemprop":"availability"})['content']
    #                 product_name = soup.find('h1', {"class":"js-product-heading heading-b product-name product-heading padding-ends"}).text.strip()
    #                 if "in stock" or "instock" or "available" in availibility.lower() and "unavailable" not in availibility.lower():
    #                     products_dict[product_name]= "available"
    #                 else:
    #                     products_dict[product_name]= "not available"
    #
    #             except:
    #
    #                 availibility = soup.find('span', {'id':"ctl00_ctl00_cphMain_cphMain_pdtProduct_spnDisplayedPrice"})["itemprop"]
    #                 product_name = soup.find('h1', {'id':"ctl00_ctl00_cphMain_cphMain_pdtProduct_h1Name"}).text
    #                 if 'price' in availibility.lower():
    #                     products_dict[product_name]= "available"
    #                 else:
    #                     products_dict[product_name]= "not available"
    #
    #             continue
    #         print(availibility.strip().replace('\n',''))
    #     print(products_dict)
    #     print(len(products_dict))

    def html_to_words(self, raw_text):
        """
        Function to convert a raw text to a string of words
        :param raw_text: a single string
        :return: The output is a single string
        """
        textWithoutPunctuation = re.sub("[^a-zA-Z]", " ", raw_text.get_text() )
        #print(textWithoutPunctuation)

        lower_case = textWithoutPunctuation.lower()        # Convert to lower case
        onlyWords = lower_case.split()                     # Split into words

        stops = set(stopwords.words("english"))

        meaningful_words = [w for w in onlyWords if not w in stops]     # Remove stop words

        #print(type(meaningful_words))

        return( meaningful_words )


    def print_counted_words(self,_vectorizer, _train_data_features):

        vocab = _vectorizer.get_feature_names()
        # Sum up the counts of each vocabulary word
        dist = np.sum(_train_data_features, axis=0)

        # For each, print the vocabulary word and the number of times it appears in the training set
        for tag, count in zip(vocab, dist):
            print(count, tag)

    def html_source_processing(self):
        os.chdir(self.data_path)
        products_dict={}
        processed_data = []
        for projectPage in glob.glob("*.html"):
            print(projectPage)
            soup = BeautifulSoup(open(projectPage),"html.parser")

            processed_data = self.html_to_words(soup)


        vectorizer = CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = None, max_features = 5000)

        # fit the model and convert the result to an array
        train_data_features = vectorizer.fit_transform(processed_data).toarray()

        print(train_data_features.shape)
        
        self.print_counted_words(vectorizer, train_data_features)



if __name__ == "__main__":
    pars = parser("../Data/train-data")
    #pars.parsing()

    pars.html_source_processing()
