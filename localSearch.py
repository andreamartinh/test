#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 11:40:31 2020

@author: amartinh
"""

#from dft import DFT
import numpy as np
import pandas as pd
import sys
import time
from random import randint
from itertools import combinations , product
import itertools
import argparse
import pickle

import random

import os
import time
from datetime import datetime
from threading import Timer




def read_pairwise_preference(men_prefs_file, women_prefs_file):
    '''
        Helper function to read the input pickle file.
    '''
    with open(men_prefs_file, 'rb') as input_file:
        df_men = pickle.load(input_file)
    with open(women_prefs_file, 'rb') as input_file:
        df_women = pickle.load(input_file)
    return len(df_men), df_men, df_women



def generateAgents(wNumber,mNumber):
    women = []
    men = []
    for i in range(wNumber):
        if i < 10:
            women.append("w"+"0"+str(i))
        else:
            women.append("w"+str(i))
        
    for i in range(mNumber):
        if i < 10:
            men.append("m"+"0"+str(i))
        else:    
            men.append("m"+str(i))
    return women, men
    


################################
### Local Search
################################
def generateRandomMatching(women,men):
   
    randomMatching = []
    for i in range(len(women)):
        w = random.choice(women)
        women.remove(w)
        
        m = random.choice(men)
        men.remove(m)
        
        couple = (m,w)
        randomMatching.append(couple)

  
    return randomMatching

def convertMatchinIntoDict(matching):
    matchingMen = {}
    matchingWomen = {}
    
    for man,woman in matching :
        matchingMen[man] = woman
        matchingWomen[woman] = man
    return matchingMen,matchingWomen

def sortArrays(pairs,betas):
    '''
    
    sortedIndex = np.argsort(betas)
    sortedPairs = []
    sortedBetas = []
    for i in sortedIndex:
        sortedPairs.append(pairs[i])
        sortedBetas.append(betas[i])
    '''
    sortedPairs, sortedBetas =  zip(*sorted(zip(betas, pairs)))
    
    return list(sortedPairs), list(sortedBetas)

def calculateAlpha(women, men, matchingMen,matchingWomen,dfMen,dfWomen):
    
            
        
    
    pairs = []
    betas = []
   
    for m in men:
        for w in women:
            if matchingMen[m] != w:
                pairs.append((m,w))
                
                currentManMatching = matchingMen[m]
                currentWomanMatching = matchingWomen[w]
                
                
                manProb = dfMen[w+currentManMatching][m]
                womanProb = dfWomen[m+currentWomanMatching][w]
                betas.append(1-(manProb*womanProb))
    
    return pairs, betas
  
def swapping(women,men,blockingPair,randomMatching):

    newMatching = []


    
    for m,w in randomMatching:

        if m == blockingPair[0]:
            currentManMatch = w
            
        if w == blockingPair[1]:
            currentWomanMatch = m
        
        if m != blockingPair[0] and  w != blockingPair[1]:
            newMatching.append((m,w))
    

    blockingManIndex = int((blockingPair[0])[1:])
    blockingMan = men[blockingManIndex]
    
    blockingWomanIndex = int((blockingPair[1])[1:])
    blockingWoman = women[blockingWomanIndex]
    
    
    
    newMatching.append((blockingMan ,blockingWoman))
    newMatching.append((currentWomanMatch,currentManMatch))

    currentMatchingMen,currentMatchingWomen = convertMatchinIntoDict(newMatching)

    
    
    
    return currentMatchingMen, currentMatchingWomen  

def matchingFromDict(matchingMen):
    matching = []
    for m,w in matchingMen.items():
        matching.append((m,w))
    return matching



def localSearch(women, men, iterations,dfMen,dfWomen,alpha_cut):
    #allMatchings = generateAllMatchings(women,men)
    
    start = time.time()
    globalAlpha = 0
    previous_global_alpha = 0
    globalAlphas = []
    allGlobalAlphas = []
    
    for k in range(iterations):
        
        allGlobalAlphas.append(globalAlpha)
        # 1) Get Randon Matchin
        tempWomen = women.copy()
        tempMen = men.copy()
        
        
        matching = generateRandomMatching(tempWomen ,tempMen)
     


        
        matchingMen,matchingWomen = convertMatchinIntoDict(matching)
        #print(matchingMen)
        
        # Calculate Alpha
        pairs,betas = calculateAlpha(women, men, matchingMen,matchingWomen,dfMen,dfWomen)
        


        sortedBetas, sortedPairs = zip(*sorted(zip(betas, pairs)))
        sortedBetas, sortedPairs = list(sortedBetas), list(sortedPairs )
        #print(sortedPairs)
        
        localAlpha = np.product(sortedBetas)
        #print("first Alpha:", localAlpha )
        localMatching = matchingMen
         
    
        while len(sortedPairs) > 0:
           
            blockingPair = sortedPairs.pop(0)
            blockingBeta = sortedBetas.pop(0)
            
            matchingMen,matchingWomen  = swapping(women,men,blockingPair,matching)
            pairs, betas = calculateAlpha(women, men, matchingMen,matchingWomen,dfMen,dfWomen)

           
            alpha =  np.product(betas)

    

            
            if alpha > localAlpha:
             
                sortedBetas, sortedPairs = zip(*sorted(zip(betas, pairs)))
                sortedBetas, sortedPairs = list(sortedBetas), list(sortedPairs )
               
                localAlpha = alpha
                localMatching = matchingMen
                matching = matchingFromDict(matchingMen)
        
        
        if localAlpha > globalAlpha:
            globalAlpha = localAlpha
            globalMatch = localMatching 
            

            if round(globalAlpha,4)  == round(alpha_cut,4):
                end = time.time()
                timeFound = end - start
                return globalAlphas, globalAlpha, globalMatch, timeFound, k, allGlobalAlphas
                

        
        if globalAlpha > previous_global_alpha:
            previous_global_alpha = globalAlpha

    end = time.time()
    timeFound = end - start
    return  globalAlphas, globalAlpha, globalMatch, timeFound, k, allGlobalAlphas
            
     
#if __name__ == "__main__":
allTotalK =[]    
allSolBool = []
allTimes = []
allAlphaSolutions = []
allSolutions = []

#number_agents = int(sys.argv[1])
#n = int(sys.argv[2])
#m = int(sys.argv[3])
#p = int(sys.argv[4])

number_agents = 13
n = 0
m = 1
p = 720

largeK = []
allK = []

women, men  = generateAgents(number_agents, number_agents)
count = 0
totalk = 0

t= time.time()
for i in range(n,m):
    
    totalK = []
    solBool =[]
    times = []
    alphaSolutions = []
    solutions = []
    
    for j in range(1):
        print( "Example:", i, "Number", j )
        df_alphaSolutions =   pd.read_pickle("data/Pairwise/Results/AplhaSolutions/"+str(number_agents) + "m" + str(number_agents) + "w"+".pkl")
        df_matchingSolution = pd.read_pickle("data/Pairwise/Results/MatchingSolutions/"+str(number_agents) + "m" + str(number_agents) + "w"+".pkl")

        

        alpha_cut = float(df_alphaSolutions[0][i])

        matchingSolution = df_matchingSolution[0][i]
        matchingSolution_d ={}
        
        #print("alpha_cut", alpha_cut, round(alpha_cut,4))
        #print("matchingSolution ",matchingSolution )
        
        for couple in matchingSolution:
            #print(couple)
            key = couple[0:3]
            item = couple[3:]
            matchingSolution_d[key] = item
            
        
        df_men =  pd.read_pickle("data/Probabilities/"+str(number_agents)+ "Agents/" +str(number_agents) + "m" + str(number_agents) + "w_men"+ str(i)+".pkl")
        df_women =  pd.read_pickle("data/Probabilities/" +str(number_agents)+"Agents/" + str(number_agents) + "m" + str(number_agents) + "w_women" + str(i)+ ".pkl")
        

        
        
        #if answer is not found it will stop after 100 iterations    
        k = 30
        globalAlphas, globalAlpha, globalMatch, timeFound, k,  allGlobalAlphas = localSearch(women, men, p ,df_men, df_women,alpha_cut)
        
        globalMatchTouples = set()
        for key,value in globalMatch.items():
            globalMatchTouples.add((key,value))
            
        
        globalMatchLP = set()
        for pair in df_matchingSolution[0][i]:
            globalMatchLP.add((pair[0:3],pair[3:]))
         
        totalK.append(k)    
        solBool.append(globalMatchLP  == globalMatchTouples)
        times.append(timeFound)
        solutions.append(globalMatch)
        alphaSolutions.append(globalAlpha)
    
        #print("k",k)

        if globalMatchLP  != globalMatchTouples:
            #print(globalMatchLP  == globalMatchTouples)
            count += 1
    
    allTotalK.append(totalK)
    allSolBool.append(solBool)
    allSolutions.append(solutions)
    allTimes.append(times)
    allAlphaSolutions.append(alphaSolutions)
'''
resultsTotalK = pd.DataFrame(allTotalK)
resultsSolBool = pd.DataFrame(allSolBool)
resultsTimesDF = pd.DataFrame(allTimes)
resultsAplhaSolutionsDF = pd.DataFrame(allAlphaSolutions)
resultsSolutionsDF = pd.DataFrame(allSolutions)

fileName = str(number_agents ) + "m"+ str(number_agents ) + "w"+"Range"+str(n) +str(m)

resultsTotalK.to_pickle("data/localSearch/Results/TotalK/"+  fileName +".pkl", protocol=2)
resultsSolBool.to_pickle("data/localSearch/Results/SolutionBoolean/"+  fileName +".pkl",protocol=2)
resultsTimesDF.to_pickle("data/localSearch/Results/Times/"+  fileName +".pkl",protocol=2)
resultsAplhaSolutionsDF.to_pickle("data/localSearch/Results/AplhaSolutions/"+  fileName +".pkl",protocol=2)
resultsSolutionsDF.to_pickle("data/localSearch/Results/MatchingSolutions/"+  fileName +".pkl",protocol=2)
'''
  
#print("Results were saved")
print("count",count)
t2 = time.time()
print(t2-t)



 