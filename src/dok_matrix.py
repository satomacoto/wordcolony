#! /usr/bin/python
# -*- encoding: utf-8 -*-

import sys
import math
import bisect
import random
from copy import deepcopy
from collections import defaultdict
from itertools import groupby
import dok

# Dictionary Of Keys based sparse matrix.
# 行列 {(r,c):v, ... }
# ベクトル {k:v,...}

class dok_matrix(defaultdict):
    def __init__(self, arg1=None, axis=0):
        '''
        arg1 : {(a,b):1,...} or {a:{b:1},...}
        '''
        defaultdict.__init__(self,float)
        
        if isinstance(arg1,dict):
            try:
                if len(arg1.keys()[0]) == 2:
                    self.update(arg1)
                elif isinstance(arg1.values()[0], dict):
                    self.update(dok_matrix.setvecs(arg1,axis))
            except:
                raise TypeError('invalid input format')
    
    '''
    def __getitem__(self, key):
        i, j = key
        if isinstance(i,int) or isinstance(i,str) and isinstance(j,int) or isinstance(j,str):
            return defaultdict.get(self, (i,j), 0.)
        else:
            rows,cols = zip(*self.keys())
            if isinstance(i,int) or isinstance(i,str):
                i = [i]
            elif isinstance(i, slice):
                if i.start or i.stop or i.step:
                    rows = sorted(set(rows))
                    i = rows[i.start or 0: i.stop or len(rows):i.step or 1]
                else:                    
                    i = rows
            if isinstance(j,int) or isinstance(j,str):
                j = [j]
            elif isinstance(j, slice):
                if j.start or j.stop or j.step:
                    cols = sorted(set(cols))
                    j = cols[j.start or 0: j.stop or len(cols):j.step or 1]
                else:
                    j = cols
            new = dok_matrix()
            for (r,c),v in self.items():
                if r in i and c in j:
                    new[r,c] = v
            return new
    '''
    
    __neg__ = dok.__neg__
    __add__ = dok.__add__
    __sub__ = dok.__sub__
        
    def __mul__(self, other):
        if isinstance(other, dok_matrix):
            return self._mul_multivector(other)
        elif isinstance(other, float) or isinstance(other, int):
            return dok.product_scalar(self,other)
        elif isinstance(other, dict):
            return self._mul_vector(other,axis=1)
    
    def __rmul__(self, other):
        if isinstance(other, dok_matrix):
            return self._mul_multivector(other)
        elif isinstance(other, float) or isinstance(other, int):
            return dok.product_scalar(self,other)
        elif isinstance(other, dict):
            return self._mul_vector(other,axis=0)
    
    def _mul_vector(self,other,axis=1):
        new = defaultdict(float)
        dd = self.tovecs(axis)
        keys = set(dd.keys()).intersection(set(other.keys()))
        for j in keys:
            ddd = dd[j]
            vv = other[j]
            for i,v in ddd.iteritems():
                new[i] += v * vv
        return new
    
    def _mul_multivector(self, other):
        r = dok_matrix()
        dd1 = self.tovecs(axis=1)
        dd2 = other.tovecs(axis=0)
        keys = set(dd1.keys()).intersection(set(dd2.keys()))
        for key in keys:
            ddd1 = dd1[key]
            ddd2 = dd2[key]
            for k1,v1 in ddd1.iteritems():
                for k2,v2 in ddd2.iteritems():
                    r[(k1,k2)] += v1 * v2
        return r
    
    def __div__(self, other):
        if not (isinstance(other,float) or isinstance(other,int)): raise NotImplementedError
        return self * (1.0/other)
    
    def __rdiv__(self, other):
        if not (isinstance(other,float) or isinstance(other,int)): raise NotImplementedError
        new = dok_matrix()
        eps = 1.e-9
        for k,v in self.items():
            if abs(v) > eps:
                new[k] = other/v
        return new
    
    @property
    def T(self):
        return self.transpose()
        
    def transpose(self):
        new = dok_matrix()
        for (i,j),v in self.items():
            new[j,i] = v
        return new
    
    def tovecs(self, axis=0):
        vecs = defaultdict(lambda:defaultdict(float))
        if axis:
            sub = 0
        else:
            sub = 1
        for k,v in self.items():
            if v:
                vecs[k[axis]][k[sub]] = v
        return vecs
    
    @staticmethod
    def setvecs(vecs, axis=0):
        mat = dok_matrix()
        for i,vec in vecs.items():
            for j,v in vec.items():
                if v:
                    if axis:
                        mat[j,i] = v
                    else:
                        mat[i,j] = v
        return mat
    
    def todense(self):
        raise NotImplementedError
        
    def getkeys(self):
        rows,cols = zip(*self.keys())
        return sorted(set(rows)),sorted(set(cols))
        
    def eliminate_zeros(self):
        for k in self._zeros():
            del self[k]
        return self

    def _zeros(self, eps=1.e-9):
        return [ k for k,v in self.iteritems() if abs(v) < eps ]
        
    def pinv2(self):
        return pinv2(self)

def mrc2dok(mrc,rowIndex,colIndex):
    dok = dok_matrix()
    indexRow = dict(zip(rowIndex.values(),rowIndex))
    indexCol = dict(zip(colIndex.values(),colIndex))
    for i,row in mrc.iteritems():
        for j,val in row.iteritems():
            if val:
                dok[(indexRow[i],indexCol[j])] = val
    return dok

def dok2mrc(dok):
    m = defaultdict(lambda:defaultdict(float))
    rows,cols = dok.getkeys()
    rowIndex = dict(zip(rows,range(len(rows))))
    colIndex = dict(zip(cols,range(len(cols))))
    for (r,c),v in dok.iteritems():
        m[rowIndex[r]][colIndex[c]] = v
    return m,rowIndex,colIndex

#### numpy
#import numpy as np
#
#def dok2array(dok):
#    rows,cols = dok.getkeys()
#    rowIndex = dict(zip(rows,range(len(rows))))
#    colIndex = dict(zip(cols,range(len(cols))))
#    a = np.zeros( (len(rows), len(cols)) )
#    for (r,c),v in dok.iteritems():
#        a[rowIndex[r],colIndex[c]] = v
#    return a,rowIndex,colIndex
####

def svd(dok):
    """
    Compute the singular value decomposition of dok matrix.
    """
    m,r,c = dok2mrc(dok)
    u,w,v = __svd(m,len(r),len(c))
    u = mrc2dok(u,r,r)
    w = mrc2dok(w,r,c)
    v = mrc2dok(v,c,c)
    return u,w,v

def __svd(a,m,n):
    '''
    Compute the singular value decomposition of array.
    m : # of rows
    n : # of cols '''

    # Golub and Reinsch state that eps should not be smaller than the
    # machine precision, ie the smallest number
    # for which 1+e>1.  tol should be beta/e where beta is the smallest
    # positive number representable in the computer.
    eps = 1.e-15  # assumes double precision
    tol = 1.e-64/eps
    assert 1.0+eps > 1.0 # if this fails, make eps bigger
    assert tol > 0.0     # if this fails, make tol bigger
    itmax = 50
    u = deepcopy(a)

    if m < n:
        if __debug__: print 'Error: m is less than n'
        raise ValueError,'SVD Error: m is less than n.'

    e = [0.0]*n  # allocate arrays
    q = [0.0]*n
    v = defaultdict(lambda:defaultdict(float))
 
    # Householder's reduction to bidiagonal form

    g = 0.0
    x = 0.0

    for i in range(n):
        e[i] = g
        s = 0.0
        l = i+1
        for j in range(i,m):
            s += (u[j][i]*u[j][i])
        if s <= tol:
            g = 0.0
        else:
            f = u[i][i]
            if f < 0.0:
                g = math.sqrt(s)
            else:
                g = -math.sqrt(s)
            h = f*g-s
            u[i][i] = f-g
            for j in range(l,n):
                s = 0.0
                for k in range(i,m): s += u[k][i]*u[k][j]
                f = s/h
                for k in range(i,m): u[k][j] = u[k][j] + f*u[k][i]
        q[i] = g
        s = 0.0
        for j in range(l,n): s = s + u[i][j]*u[i][j]
        if s <= tol:
            g = 0.0
        else:
            f = u[i][i+1]
            if f < 0.0:
                g = math.sqrt(s)
            else:
                g = -math.sqrt(s)
            h = f*g - s
            u[i][i+1] = f-g
            for j in range(l,n): e[j] = u[i][j]/h
            for j in range(l,m):
                s=0.0
                for k in range(l,n): s = s+(u[j][k]*u[i][k])
                for k in range(l,n): u[j][k] = u[j][k]+(s*e[k])
        y = abs(q[i])+abs(e[i])
        if y>x: x=y
    # accumulation of right hand gtransformations
    for i in range(n-1,-1,-1):
        if g != 0.0:
            h = g*u[i][i+1]
            for j in range(l,n): v[j][i] = u[i][j]/h
            for j in range(l,n):
                s=0.0
                for k in range(l,n): s += (u[i][k]*v[k][j])
                for k in range(l,n): v[k][j] += (s*v[k][i])
        for j in range(l,n):
            v[i][j] = 0.0
            v[j][i] = 0.0
        v[i][i] = 1.0
        g = e[i]
        l = i
    #accumulation of left hand transformations
    for i in range(n-1,-1,-1):
        l = i+1
        g = q[i]
        for j in range(l,n): u[i][j] = 0.0
        if g != 0.0:
            h = u[i][i]*g
            for j in range(l,n):
                s=0.0
                for k in range(l,m): s += (u[k][i]*u[k][j])
                f = s/h
                for k in range(i,m): u[k][j] += (f*u[k][i])
            for j in range(i,m): u[j][i] = u[j][i]/g
        else:
            for j in range(i,m): u[j][i] = 0.0
        u[i][i] += 1.0
    #diagonalization of the bidiagonal form
    eps = eps*x
    for k in range(n-1,-1,-1):
        for iteration in range(itmax):
            # test f splitting
            for l in range(k,-1,-1):
                goto_test_f_convergence = False
                if abs(e[l]) <= eps:
                    # goto test f convergence
                    goto_test_f_convergence = True
                    break  # break out of l loop
                if abs(q[l-1]) <= eps:
                    # goto cancellation
                    break  # break out of l loop
            if not goto_test_f_convergence:
                #cancellation of e[l] if l>0
                c = 0.0
                s = 1.0
                l1 = l-1
                for i in range(l,k+1):
                    f = s*e[i]
                    e[i] = c*e[i]
                    if abs(f) <= eps:
                        #goto test f convergence
                        break
                    g = q[i]
                    h = pythag(f,g)
                    q[i] = h
                    c = g/h
                    s = -f/h
                    for j in range(m):
                        y = u[j][l1]
                        z = u[j][i]
                        u[j][l1] = y*c+z*s
                        u[j][i] = -y*s+z*c
            # test f convergence
            z = q[k]
            if l == k:
                # convergence
                if z<0.0:
                    #q[k] is made non-negative
                    q[k] = -z
                    for j in range(n):
                        v[j][k] = -v[j][k]
                break  # break out of iteration loop and move on to next k value
            if iteration >= itmax-1:
                if __debug__: print 'Error: no convergence.'
                # should this move on the the next k or exit with error??
                #raise ValueError,'SVD Error: No convergence.'  # exit the program with error
                break  # break out of iteration loop and move on to next k
            # shift from bottom 2x2 minor
            x = q[l]
            y = q[k-1]
            g = e[k-1]
            h = e[k]
            f = ((y-z)*(y+z)+(g-h)*(g+h))/(2.0*h*y)
            g = pythag(f,1.0)
            if f < 0:
                f = ((x-z)*(x+z)+h*(y/(f-g)-h))/x
            else:
                f = ((x-z)*(x+z)+h*(y/(f+g)-h))/x
            # next QR transformation
            c = 1.0
            s = 1.0
            for i in range(l+1,k+1):
                g = e[i]
                y = q[i]
                h = s*g
                g = c*g
                z = pythag(f,h)
                e[i-1] = z
                c = f/z
                s = h/z
                f = x*c+g*s
                g = -x*s+g*c
                h = y*s
                y = y*c
                for j in range(n):
                    x = v[j][i-1]
                    z = v[j][i]
                    v[j][i-1] = x*c+z*s
                    v[j][i] = -x*s+z*c
                z = pythag(f,h)
                q[i-1] = z
                c = f/z
                s = h/z
                f = c*g+s*y
                x = -s*g+c*y
                for j in range(m):
                    y = u[j][i-1]
                    z = u[j][i]
                    u[j][i-1] = y*c+z*s
                    u[j][i] = -y*s+z*c
            e[l] = 0.0
            e[k] = f
            q[k] = x
            # goto test f splitting
        
    d = defaultdict(lambda:defaultdict(float))
    for i in range(n):
        d[i][i] = q[i]
    
    return (u,d,v)

def pythag(a,b):
    """
    ピタゴラスの定理
    """
    absa = abs(a)
    absb = abs(b)
    if absa > absb: return absa*math.sqrt(1.0+(absb/absa)**2)
    else:
        if absb == 0.0: return 0.0
        else: return absb*math.sqrt(1.0+(absa/absb)**2)

def pinv2(d):
    """
    Compute the (Moore-Penrose) pseudo-inverse of a matrix.

    Calculate a generalized inverse of a matrix using its
    singular-value decomposition and including all 'large' singular
    values.
    """
    u,w,v = svd(d)
    return (v*(1./w.T)*u.T).eliminate_zeros()

def kmeans(d,points=3,axis=0,iter=20):
    def init_centroids(dok_vecs):
        centroids = []
        # 1a. choose an initial center c_1 uniformly at random X
        centroids.append(dok_vecs.values()[0])
        
        # 1b. choose the next ceter c_i selecting c_i = x' in X w/probablity D(x')^2/\sum_{x in X} D(x)
        # 1c. repeat step 1b until i == k 
        items = []
        for i in range(1,points):
            for w,v in dok_vecs.items():
                temp = [ dok.squaredistance(v,c) for c in centroids ]
                temp = [ x for x in temp if temp > 0 ]
                items += [(w,min(temp))]
            centroids.append(dok_vecs[dok.weighted_choice_bisect_compile(items)()])
        return centroids

    dok_vecs = d.tovecs(axis)
    centroids = init_centroids(dok_vecs)
    #centroidsを更新する。
    for i in range(iter):
        clusters = []
        for i in range(points):
            clusters += [[]]
        for v in dok_vecs.values():
            temp = [ (dok.squaredistance(v,centroids[i]),i) for i in range(points) ]
            clusters[min(temp)[1]] += [v]
        for i in range(points):
            if len(clusters[i]) > 1:
                centroids[i] = dok.mean(clusters[i])
            else:
                centroids[i] = centroids[i]
    return centroids,clusters

def _kmeans(dok,points=3,axis=0,iter=20):
    zipWith = lambda f, xs, ys : [f(x, y) for x,y in zip(xs, ys)]
    snd  = lambda x: x[1]
    fst  = lambda x: x[0]
    euclid  = lambda x, y : (x-y)**2
    
    def closest(pts, pt):
        closest_ct = pts[0]
        for ct in pts[1:]:
            if distance(pt[1],closest_ct) > distance(pt[1],ct):
                closest_ct = ct
        return closest_ct
    
    def recluster_(centroids,points):
        reclustered = [(closest(centroids,a), a) for a in points]
        reclustered.sort()
        return [map(snd,list(g)) for k, g in groupby(reclustered, fst)]
    
    def recluster(clusters):
        centroids = map(mean, clusters)
        concated_clusters = reduce(lambda a,b: a+b, clusters)
        return recluster_(centroids,concated_clusters)
    
    def part(l,points):
        size = int(math.ceil(len(l)/float(points)))
        return [l[i:i+size] for i in range(0,len(l),size)]

    k = dok.tovecs(axis).items()
    cluster = part(k,points)
    newcluster = recluster(cluster)
    i = 0
    while cluster != newcluster and i < iter:
        cluster = newcluster
        newcluster = recluster(cluster)
        i += 1
    return newcluster


def main():
    A = dok_matrix()
    A['A','a']=1.0
    A['A','b']=1.0
    A['B','a']=1.0
    A['B','b']=3.0
    A['C','a']=2.0
    A['C','b']=3.0
    print A
    print A.pinv2()

#    A = dok_matrix(A)
#    B = dok_matrix(A)
    
#   print b*A
#   print A*b
#   print pinv2(A)
    
#    print A.pinv2()
#    print kmeans(A,points=2)
    
if __name__ == '__main__':
    main()
