#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os

from flask import Flask, render_template, request, abort, jsonify
app = Flask(__name__)


import os
# import simplejson as json
import json
from urllib import urlencode, unquote_plus
from xml.dom import minidom
from src.parser import Parser
from src.scene import analyze
from src.dok_matrix import *
from src.yahooapi import MAService
import src.dok as dok

def tokenize(text, lang='en'):
    '''
    >>> text = 'Hello world. This is a test.'
    >>> tokenize(text, lang='en')
    [['Hello', 'world'], ['This', 'is', 'a', 'test']]
    '''
    if lang == 'en':
        return etokenize(text)
    elif lang == 'ja':
        if not isinstance(text, str):
            text = text.encode('utf-8')
        return jtokenize(text)

def etokenize(text):
    p = Parser()
    sentences = []
    for line in text.split('.'):
        tokens = p.tokenise(line, stem=False)
        if not tokens: continue
        tokens = p.removeStopwords(tokens)
        if not tokens: continue
        sentences.append(tokens)
    return sentences

def jtokenize(text, sep=['.', '。', '．', '\n'], filter=['9']):
    # yahoo api
    p = Parser()
    client = MAService()

    # set the end of sentence
    eos = 'EOS'
    
    for x in sep:
        text = text.replace(x, ' ' + eos + ' ')
    text += eos
    
    # filter
    filter = "|".join(filter)
    
    # get json
    words = client.parse(text, filter=filter)
    
    # split document
    sentences = []
    temp = []
    settoji = ""
    for i in range(len(words)):
        pos = words[i]['pos'].encode('utf-8')
        baseform = words[i]['baseform'].encode('utf-8')
        if baseform == eos:
            sentences += [temp]
            temp = []
        # 接頭辞だったら一時的に保存
        elif pos == '接頭辞':
            settoji = baseform
        # 接尾辞だったら前の語と一緒にする
        elif pos == '接尾辞' and temp[-1]:
            temp[-1] += baseform
        else:
            # 前が接頭辞だったら
            if settoji:
                baseform = settoji + baseform
                settoji = ""
            # ストップワードの除去
            if baseform in p.stopwords: continue
            temp += [baseform]
    
    # 空の除去
    sentences = [ sentence for sentence in sentences if sentence ]
#    sentences = [ [ word for word in sentence] for sentence in sentences ]
    return sentences

def get_scene(sentences):
    tc, cc, td, attr = analyze(sentences)
    return td, attr

def get_bridge(source, target):
    bridge = {}
    return bridge
    
def to_kv(d):
    r = []
    for k, v in d.items():
        r += [{ "key": k, "value": v }]
    return r
    
def to_stv(d):
    r = []
    for (s, t), v in d.items():
        if v > 0:
            r += [{ "source": s, "target": t, "value": v }]
    return r

def output(handler, data, callback=''):
    jsondata = json.dumps(data, ensure_ascii=False)
    handler.response.headers["Content-Type"] = "text/javascript"
    handler.response.out.write("%s(%s)" % (callback, jsondata))

def getxml(nodes, edges):
    '''
    nodes: { k:v, ...
    edges: { (s, t):v, ...
    '''
    doc = minidom.Document()
    
    doc.appendChild(doc.createComment("This is a XML document"))
    
    # graphml
    graphml = doc.createElement('graphml')
    doc.appendChild(graphml)
    
    # key label    
    def set_key(_id, _for, _attrname, _attrtype):
        weight = doc.createElement('key')
        for k, v in [('id', _id), ('for', _for), ('attr.name', _attrname), ('attr.type', _attrtype)]:
            weight.setAttribute(k, v)
        graphml.appendChild(weight)
    set_key('label', 'all', 'label', 'string')
    set_key('weight', 'node', 'weight', 'double')
    
    # graph
    graph = doc.createElement('graph')
    graph.setAttribute('edgedefault', 'directed')
    graphml.appendChild(graph)
    
    def set_node(id, label, weight='1.0'):
        node = doc.createElement('node')
        node.setAttribute('id', id)
        for k, v in [('label', label), ('weight', weight)]:
            data = doc.createElement('data')
            data.setAttribute('key', k)
            data.appendChild(doc.createTextNode(v))
            node.appendChild(data)
        graph.appendChild(node)
    
    def set_edge(source, target, weight='1.0'):
        edge = doc.createElement('edge')
        edge.setAttribute('source', source)
        edge.setAttribute('target', target)
#        data = doc.createElement('data')
#        data.setAttribute('key', 'weight')
#        data.appendChild(doc.createTextNode(weight))
#        edge.appendChild(data)
        graph.appendChild(edge)
    
    # graph
    for k, v in nodes.items():
        set_node(k, k, str(v))
    for (s, t), v in edges.items():
        set_edge(s, t, str(v)) 
    
    return doc.toprettyxml(indent = '    ')

@app.route('/td_attr', methods=['POST', 'GET'])
def main():
    if request.method == 'POST':
        text = request.form['text']
        lang = request.form['lang'] if 'lang' in request.form else 'en'
        output = request.form['output'] if 'output' in request.form else 'json'
        callback = request.form['callback'] if 'callback' in request.form else ''
    else:
        text = request.args.get('text')
        lang = request.args.get('lang') if 'lang' in request.args else 'en'
        output = request.args.get('output') if 'output' in request.args else 'json'
        callback = request.args.get('callback') if 'callback' in request.args else ''    #
    # error
    if len(text) > 10000:
        abort(400)
    #
    tokens = tokenize(text, lang)
    td, attr = get_scene(tokens)
    if output == 'xml':
        _td = {}
        for (s, t), v in td.items():
            if s <> t and v > 0.8:
                _td[s, t] = v
        return getxml(attr, _td)
    else:
        scene = {"td": to_stv(td), "attr": to_kv(attr)}
        res = jsonify({"elements": scene}, callback=callback)
        # return "%s(%s)" % (callback, res.encode('utf-8'))
        app.logger.error(res)
        return res

@app.route('/')
def hello():
    return render_template('d3js.html')

if __name__ == '__main__':
    # Bind to PORT if defined, otherwise default to 5000.
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
