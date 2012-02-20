#! /usr/bin/python
# -*- encoding: utf-8 -*-

from yahooapi import MAService
import glob
import re
import os

class Parser:

    #A processor for removing the commoner morphological and inflexional endings from words in English
    stopwords=[]
    client = None

    def __init__(self,):
        self.client = MAService()
        self.p = re.compile(r"&.{1,5}?;|\W")
        for file in glob.glob(os.path.dirname(__file__)+'/stopwords/*/*.txt'):
            self.stopwords += [ line.strip() for line in open(file).readlines() ]
        self.stopwords.append('the')

    def clean(self, string):
        """ remove any nasty grammar tokens from string """
        string = self.p.sub(' ',string)
        string = string.lower()
        return string

    def removeStopwords(self,list):
        """ Remove common words which have no search value """
        return [word for word in list if word not in self.stopwords ]

    def tokenise(self, string):
        """ break string up into tokens and stem words """
        words = string.split()
        return [ x["baseform"] for x in self.client.parse(string,results="ma",response="baseform") ]
