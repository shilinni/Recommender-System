import csv
import json
import pandas as pd
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math
import random
from sklearn.metrics.pairwise import pairwise_distances

file_name = "movie_ratings.csv"
pd.options.display.max_rows = 20


class recommender():
    def __init__(self, file_name, k, alpha):
        self.file_name = file_name
        self.k = k
        self.alpha = alpha
        self.U = None
        self.V = None

    def load(self, percent):
        self.percent = percent

        self.df = pd.read_csv(file_name, encoding='latin-1')
        self.allnrows = self.df.shape[0]
        self.nrows = math.floor(self.allnrows * percent)
        self.n = self.df.userId.unique().shape[0]
        self.d = self.df.movieId.unique().shape[0]
        self.unique_users = self.df.userId.unique()
        self.unique_movies = self.df.movieId.unique()
        self.user_dict = {k: v for v, k in enumerate(self.unique_users)}
        self.movie_dict = {k: v for v, k in enumerate(self.unique_movies)}
        self.matrix = np.zeros((self.n, self.d))
        self.vlist = []
        self.fulllist=[]
        self.valid_ur=np.zeros((self.n,1))
        self.valid_mr=np.zeros((self.d,1))
        for row in self.df.itertuples():
            self.matrix[self.user_dict[row.userId], self.movie_dict[row.movieId]] = row.rating
            self.fulllist.append((self.user_dict[row.userId], self.movie_dict[row.movieId], row.rating))
            if row.rating!=0:
                self.valid_ur[self.user_dict[row.userId]]+=1
                self.valid_mr[self.movie_dict[row.movieId]]+=1
        self.vlist=list(self.fulllist)
        np.random.shuffle(self.vlist)
        self.shuffledlist=list(self.vlist)
        self.testvlist =self.vlist[self.nrows:]
        self.vlist=self.vlist[:self.nrows]
        self.MSEs = dict()
        self.uni_trau=0
        self.uni_tram=0

    def m1_train(self, niter, cont=None):
        m = math.sqrt(np.mean(self.vlist, 0)[2] / self.k)
        if not cont:
            #self.U = np.ones((self.n, self.k)) * math.sqrt(m / float(self.k))
            #self.V = np.ones((self.d, self.k)) * math.sqrt(m / float(self.k))
            self.U=np.random.normal(scale=1./self.k, size=(self.n, self.k))
            self.V=np.random.normal(scale=1./self.k, size=(self.d, self.k))
        result = []
        for it in range(niter):  # change to while err<eps
            #print("iteration: ", it)
            np.random.shuffle(self.vlist)
            for i, j, r in self.vlist:
                eij = r - self.U[i, :].dot(self.V[j, :].T)
                self.U[i, :] = self.U[i, :] + 2 * self.alpha * eij * self.V[j, :]
                self.V[j, :] = self.V[j, :] + 2 * self.alpha * eij * self.U[i, :]

            if it%10==0:
                mse=self.MSE()
                result.append(mse)
                print("iteration= {}, MSE_train= {}", it, mse)
        return result

    def MSE(self):
        e_sum = 0
        for i, j, r in self.vlist:
            eij = r - self.U[i, :].dot(self.V[j, :].T)
            e_sum += pow(eij, 2)
        return e_sum/self.nrows

    def MSE_on_test_m1(self):
        e_sum = 0
        for i, j, r in self.testvlist:
            eij = r - self.U[i, :].dot(self.V[j, :].T)
            e_sum += pow(eij, 2)
        return e_sum / len(self.testvlist)

    def MSE_on_test_m2(self):
        e_sum = 0
        for i, j, r in self.testvlist:
            rt1 = self.valid_ur[i] / self.d
            rt2 = self.valid_mr[j] / self.n
            lt1 = rt1 / (rt1 + rt2)
            lt2 = rt2 / (rt1 + rt2)
            rp = lt1 * self.U[i, :].dot(self.U[i, :].T) + lt2 * self.V[j, :].dot(self.V[j, :].T)
            eij = r - rp
            e_sum += pow(eij, 2)
        return e_sum / len(self.testvlist)

    def m2_train(self,niter,cont=None):
        #if self.uni_tram==0 or self.uni_trau==0:
        #    self.uni_trau=len(set([x[0] for x in self.vlist]))
        #    self.uni_tram = len(set([x[1] for x in self.vlist]))
        m = math.sqrt(np.mean(self.vlist, 0)[2] / self.k)
        if not cont:
            self.U = np.random.normal(scale=1. / self.k, size=(self.n, self.k))#.astype('Float64')
            self.V = np.random.normal(scale=1. / self.k, size=(self.d, self.k))#.astype('Float64')
        result = []
        for it in range(niter):  # change to while err<eps
            # print("iteration: ", it)
            np.random.shuffle(self.vlist)
            for i, j, r in self.vlist:
                #rp=self.U[i, :].dot(self.V[j, :].T)+self.U[i, :].dot(self.U[i, :].T)+self.V[j, :].dot(self.V[j, :].T)
                #eij = r - rp/3
                #self.U[i, :] = self.U[i, :] + 2 /3 *self.alpha * eij * (self.V[j, :]+2*self.U[i,:])
                #self.V[j, :] = self.V[j, :] + 2 /3* self.alpha * eij * (2*self.V[j, :]+self.U[i,:])
                rt1=self.valid_ur[i]/self.d
                rt2=self.valid_mr[j]/self.n
                lt1=rt1/(rt1+rt2)
                lt2=rt2/(rt1+rt2)
                rp=lt1*self.U[i, :].dot(self.U[i, :].T)+lt2*self.V[j, :].dot(self.V[j, :].T)
                eij = r - rp
                self.U[i, :] = self.U[i, :] + 4*lt1 *self.alpha * eij * self.U[i,:]
                self.V[j, :] = self.V[j, :] + 4*lt1* self.alpha * eij * self.V[j, :]

            #if it % 10 == 0:
                #mse = self.MSE()
                #result.append(mse)
                #print("iteration= {}, MSE_train= {}", it, mse)
        return result

    def recommendations(self, nitems, userId, silence=False, recall=False):
        if userId in self.user_dict:
            i = self.user_dict[userId]
            result = []
            for j in range(self.d):
                # if recall and no actual value, skip this one
                if not recall or self.matrix[i, j] != 0:
                    result.append((userId, self.unique_movies[j], self.U[i, :].dot(self.V[j, :].T)))
            result.sort(key=lambda x: x[2], reverse=True)
            if not silence:
                df = pd.DataFrame(result, columns=['userId', 'movieId', 'rating'])
                print(df.head(nitems))

            return result[:nitems]  # for recall use
        return []

    def recall(self, nitems, likes=None):
        if not likes:
            likes = 3.5
        result = dict()
        m_predicted = self.U.dot(self.V.T)

        for user, i in self.user_dict.items():
            total_likes = 0
            rec_likes = 0
            rlimit = self.recommendations(nitems, user, silence=True, recall=True)[-1][2]
            for j in range(self.d):
                if self.matrix[i, j] >= likes:
                    total_likes += 1
                    if m_predicted[i, j] >= rlimit:
                        rec_likes += 1
            if total_likes == 0:
                result[user] = 1
            else:
                result[user] = float(rec_likes) / total_likes
        return result

    def precision(self, nitems, likes=None):
        if not likes:
            likes = 3.5
        result = dict()
        m_predicted = self.U.dot(self.V.T)
        for user, i in self.user_dict.items():
            total_likes = 0
            rec_likes = 0
            ritems = self.recommendations(nitems, user, silence=True, recall=True)
            for iid, jid, r in ritems:
                j = self.movie_dict[jid]
                if self.matrix[i, j] >= likes:
                    rec_likes += 1
            result[user] = float(rec_likes) / len(ritems)
        return result

#don't call anything before this.
def k_fold(r,kf,m1,iters):

    r.load(1 - 1. / kf)
    testlen = math.floor(r.allnrows / kf)
    for i in range(kf):
        r.testvlist = r.shuffledlist[i * testlen:(i + 1) * testlen]
        r.vlist = r.shuffledlist[:i * testlen] + r.shuffledlist[(i + 1) * testlen:]
        if m1:
            r.m1_train(10, False)
            r.MSEs[(10, i, kf)] = r.MSE_on_test_m1()
        else:
            r.m2_train(10,False)
            r.MSEs[(10, i, kf)] = r.MSE_on_test_m2()

        for iter in range(iters-1):
            if m1:
                r.m1_train(10, True)
                r.MSEs[((iter + 2) * 10, i, kf)] = r.MSE_on_test_m1()
            else:
                r.m2_train(10, True)
                r.MSEs[((iter + 2) * 10, i, kf)] = r.MSE_on_test_m2()

#don't call anything before this.
def k_fold_newuser(r,kf,m1,iters):
    r.load(1 - 1. / kf)
    testlen = math.floor(r.allnrows / kf)
    for i in range(kf):
        r.testvlist = r.fulllist[i * testlen:(i + 1) * testlen]
        r.vlist = r.fulllist[:i * testlen] + r.fulllist[(i + 1) * testlen:]

        ur1user = set([i for i, j, r in r.vlist])
        l=[]
        for x in r.testvlist:
            if x[0] not in ur1user:
                l.append(x)
        r.testvlist=list(l)

        if m1:
            r.m1_train(10, False)
            r.MSEs[(10, i, kf)] = r.MSE_on_test_m1()
        else:
            r.m2_train(10,False)
            r.MSEs[(10, i, kf)] = r.MSE_on_test_m2()

        for iter in range(iters-1):
            if m1:
                r.m1_train(10, True)
                r.MSEs[((iter + 2) * 10, i, kf)] = r.MSE_on_test_m1()
            else:
                r.m2_train(10, True)
                r.MSEs[((iter + 2) * 10, i, kf)] = r.MSE_on_test_m2()

def p1():
    r1 = recommender(file_name, 10, 0.001)
    r2 = recommender(file_name, 10, 0.001)
    kf = 4
    iters = 4
    k_fold(r1, kf, True, iters)
    k_fold(r2, kf, False, iters)
    for i in range(kf):
        err1 = []
        err2 = []
        for j in range(iters):
            err1.append(r1.MSEs[(j * 10 + 10, i, kf)])
            err2.append(r2.MSEs[(j * 10 + 10, i, kf)][0])
        il4 = [x * 10 + 10 for x in range(iters)]
        plt.plot(il4, err1, 'r--', il4, err2, 'b--')
        plt.show()

def p2():
    r1 = recommender(file_name, 10, 0.001)
    r2 = recommender(file_name, 10, 0.001)
    ks = [2, 3, 4]
    iters = 2
    err1 = []
    err2 = []
    for kf in ks:
        k_fold(r1, kf, True, iters)
        k_fold(r2, kf, False, iters)
        e1 = 0
        e2 = 0
        for i in range(kf):
            e1 += (r1.MSEs[(20, i, kf)])
            e2 += (r2.MSEs[(20, i, kf)][0])
        err1.append(e1 / kf)
        err2.append(e2 / kf)
    plt.plot(ks, err1, 'r--', ks, err2, 'b--')
    plt.show()


def p3():
    # test new user
    kf = 4
    iters = 3
    r1 = recommender(file_name, 10, 0.001)
    r2 = recommender(file_name, 10, 0.001)
    k_fold_newuser(r1, kf, True, iters)
    k_fold_newuser(r2, kf, True, iters)
    err1 = []
    err2 = []
    for i in range(kf):
        err1.append(r1.MSEs[(30, i, kf)])
        err2.append(r2.MSEs[(30, i, kf)])
    plt.plot([1, 2, 3, 4], err1, 'r--', [1, 2, 3, 4], err2, 'b--')
    plt.show()

