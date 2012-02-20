#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import defaultdict

def combinations(iterable, r):
    # combinations('ABCD', 2) --> AB AC AD BC BD CD
    # combinations(range(4), 3) --> 012 013 023 123
    pool = tuple(iterable)
    n = len(pool)
    if r > n:
        return
    indices = range(r)
    yield tuple(pool[i] for i in indices)
    while True:
        for i in reversed(range(r)):
            if indices[i] != i + n - r:
                break
        else:
            return
        indices[i] += 1
        for j in range(i+1, r):
            indices[j] = indices[j-1] + 1
        yield tuple(pool[i] for i in indices)

def count_item(L):
    S = set(L)
    D = {}
    for x in S:
        D[x] = L.count(x)
    return D

def analyze(events, stopwords=[], selected=[], axis=0):
    '''
    Parameters
    ----------
    events: [['w1','w2',...],[...],...]
    stopwords: ['w1','w2',...]
    selected: ['w1','w2',...]

    Returns
    -------
    wordCount,cooccurrenceCount,termDependency,termAttractiveness

    wordCount : defaultdict(int)
        {'w1': 69,...}
    cooccurrenceCount : defaultdict(int)
        {('w1','w2': 69,...}
    termDependency : defaultdict(float)
        {('w1','w2'): 0.69,...}
    termAttractiveness : defualtdict(float)
        {'w1': 6.9,...}
    '''

    words = []
    if selected:
        events = [ [ word for word in event if not word in selected ] for event in events ]
    if stopwords:
        events = [ [ word for word in event if not word in stopwords ] for event in events ]                    
    
    for x in events:
        words += set(x)
    wordCount = defaultdict(int)
    for x in words:
        wordCount[x] += 1
    
    
    cooccurrences = []
    for s in events:
        if len(s) > 1:
            s = set(s)
            cooccurrences += list(combinations(s, 2))
            cooccurrences += list(zip(s,s))

    cooccurrenceCount = defaultdict(int)
    for x in cooccurrences:
        cooccurrenceCount[x] += 1
    
    termDependency = defaultdict(float)
    for (key1,key2),value in cooccurrenceCount.iteritems():
        termDependency[(key1,key2)] = 1.0 * value / wordCount[key1]
        termDependency[(key2,key1)] = 1.0 * value / wordCount[key2]
        
    termAttractiveness = defaultdict(float)
    if axis == 0:
        for (key1,key2),value in termDependency.iteritems():
            termAttractiveness[key2] += value
    else:
        for (key1,key2),value in termDependency.iteritems():
            termAttractiveness[key1] += value
    
    return wordCount,cooccurrenceCount,termDependency,termAttractiveness

def text2events(text, sep=['\n','．','。','.'], parser=""):
    try:
        if parser == "m":
            import mparser as parser
        elif parser == "y":
            import yparser as parser
        else:
            import eparser as parser
    except ImportError:
        print "ImportError"
        return
    p = parser.Parser()
    events = [text]
    for c in sep:
        tmp = []
        for x in events:
            tmp += x.split(c)
        events = tmp
    events = [ p.clean(event) for event in events ]    
    events = [ p.tokenise(event) for event in events ]    
    events = [ p.removeStopwords(event) for event in events ]
    events = [ event for event in events if event ]
    return events

def scene(text, sep=['\n','．','。','.'], parser="", stopwords=[], selected=[], axis=0, all=False):
    """
    テキストから共起依存度，吸引力を求める．
    
    Parameters
    ----------
    text : str
        解析対象のテキスト
    stopwords : list(str)
        ストップワード
    seleceted : list(str)
        解析するためのワード
    axis : 0 or 1
        軸
    all : bool
        すべて出力するか
        
    Returns
    -------
    wordCount,cooccurrenceCount,termDependency,termAttractiveness

    wordCount : defaultdict(int)
        {'w1': 69,...}
    cooccurrenceCount : defaultdict(int)
        {('w1','w2': 69,...}
    termDependency : defaultdict(float)
        {('w1','w2'): 0.69,...}
    termAttractiveness : defualtdict(float)
        {'w1': 6.9,...}
    
    Examples
    --------
    
    """
    return analyze(text2events(text,sep,parser), stopwords, selected, axis)

if __name__ == "__main__":
    print 0