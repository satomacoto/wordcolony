#!/usr/bin/env python
# -*- coding: utf-8 -*-

import urllib
#import simplejson
from xml.dom import minidom

class MAService():
    appid = "D4B75Aaxg64RQQVFyySaX9kHXteiRdZnXQVqkJT7fjClojbRutUTDURp_wcF"

    def parse(self,sentence,results="ma,uniq",response="surface,reading,pos,baseform,feature",filter=""):
        # http://developer.yahoo.co.jp/webapi/jlp/ma/v1/parse.html
        url = "http://jlp.yahooapis.jp/MAService/V1/parse"

        form_fields = {"appid": self.appid,
                       "sentence": sentence,
                       "results": results,
                       "response": response,
                       "filter": filter,
                       "ma_response": "",
                       "ma_filter": "",
                       "uniq_response": "",
                       "uniq_filter": "",
                       "uniq_by_baseform": ""}
        form_data = urllib.urlencode(form_fields)
        result = urllib.urlopen(url, form_data)
        ma_result = []
        if result.getcode() == 200:
            dom = minidom.parseString(result.read())
            for word in dom.getElementsByTagName('ma_result')[0].getElementsByTagName('word'):
                func = lambda x: word.getElementsByTagName(x)[0].childNodes[0].data        
                ma_result += [dict([ (x,func(x)) for x in response.split(',') ])]
        
#        return simplejson.dumps(ma_result)
        return ma_result
