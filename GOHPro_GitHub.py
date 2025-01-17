import argparse
import warnings
import math
import numpy as np
import pandas as pd
from math import factorial
from datetime import datetime
from collections import OrderedDict
import random
import networkx as nx
from scipy.stats import pearsonr, spearmanr
import os
from functools import reduce
import scipy.spatial.distance as sd
from operator import itemgetter


PPMatrix=[]
PGMatrix=[]
PDMatrix=[]
PCMatrix=[]
GoList=[]
Domainlist=[]
Complexlist=[] 
proteinlist=[]
TrainProteinList=[]
TestProteinList=[]
NList=[]
N2List=[] 
CPList=[]
PCList=[]
PGList=[]
PDList=[]
DPList=[]
PGONum=[]
WeightC=[]
WeightD=[]
WeightP=[] 
WeightG=[]
WeightE=[]
TenFold=[] 
CurrentSet=[]
ComplexScores=[]
α=0.1
β=0.1
ε=0.00001
γ=0.7
GOType='F'
CAFAStr=''
#CAFAStr='_CAFA3'
Species='Saccharomyces_cerevisiae'
#Species='Homo_sapiens'
def combination(m, n):
    #return math.factorial(m) // (math.factorial(n) * math.factorial(m - n))
    if n > m:
        return 0
    if n == 0 or n == m:
        return 1
    result = 1
    for i in range(1, n + 1):
        result *= (m - i + 1)
        result //= i
    return result 
 
def HDP_Complex(N,M,n,k):
    return combination(M,k)*combination(N-M,n-k)/combination(N,n)

def getscore_complex(N,M,n,k):
    scores=0
    counter = 1
    while counter<=k:
          scores=scores+HDP_Complex(N,M,n,counter)
          counter=counter+1
    return scores


def Weight_Complex():
   global ComplexScores
   ComplexScores=[0 for _ in range(len(Complexlist))]
   N=len(proteinlist)
   M=len(GoList)
   MaxScore=0
   MinScore=1  
   for i in range(0,len(Complexlist)):
      L=len(CPList[i])    
      r=0
      for k in range(0,len(CPList[i])):
        if (k>=0)and (PGONum[int(CPList[i][k])]>0):
          r=r+1   
      if N>=M:    
        a=getscore_complex(N,M,L,r) 
      else:
        a=0 
      if a<0:
       print(N,M,L,r)      
      ComplexScores[i]=a
   global WeightC    
   WeightC = [[0 for _ in range(N)] for _ in range(N)]
   for i in range(0,N-1):
     for j in range(i+1,N):
       score_a=0
       score_b=0
       score_common=0
       intersection =list(set(PCList[i]) & set(PCList[j]))
       for k in range(0,len(PCList[i])):
          score_a=score_a+ComplexScores[int(PCList[i][k])]
       for k in range(0,len(PCList[j])):
          score_b=score_b+ComplexScores[int(PCList[j][k])]
       for k in range(0,len(intersection)):
          score_common=score_common+ComplexScores[int(intersection[k])]           
       score_a=len(PCList[i])
       score_b=len(PCList[j])
       score_common=len(intersection)
       if (score_a*score_b*score_common>0):
         value=(score_common*score_common)/(score_a*score_b)
       else:  
         value=0
       if value<0:
         print(value)       
       WeightC[i][j]=value
       WeightC[j][i]=value
       if WeightC[i][j]>MaxScore:
           MaxScore=WeightC[i][j]
       if WeightC[i][j]<MinScore:
           MinScore=WeightC[i][j]  
   for i in range(0,N-1):
     for j in range(i+1,N):
       WeightC[i][j]=(WeightC[i][j]-MinScore)/(MaxScore-MinScore)
       WeightC[j][i]=WeightC[i][j]

def Weight_Domain():
  listlen=len(proteinlist)
  global WeightD
  global  PDList 
  WeightD = [[0 for _ in range(listlen)] for _ in range(listlen)]
  N_CPList=[[0 for j in range(0)] for i in range(listlen)]
  PDList=[[0 for j in range(0)] for i in range(listlen)]
  PDList2=[[0 for j in range(0)] for i in range(listlen)] 
  
  for i in range(0,listlen): 
     for j in range(0,len(Domainlist)):
       if PDMatrix[i][j]==1:
          PDList[i].append(j)
       for k in range(len(NList[i])):
          if PDMatrix[int(NList[i][k])][j]==1:
            PDList2[i].append(j) 
            break           
  M=len(Domainlist)
  MaxScore=0
  MinScore=1  
  for i in range(0,listlen-1):
    #print(i+1,'/',len(proteinlist),end='\r')
    if PGONum[i]==0:
      continue
    for j in range(i+1,listlen): 
      if PGONum[j]==0:
       continue 
      a1=len(PDList[i])
      b1=len(PDList[j])
      intersection1 =list(set(PDList[i]) & set(PDList[j]))
      Common1=len(intersection1)
      a2=len(PDList2[i])
      b2=len(PDList2[j])
      intersection2 =list(set(PDList2[i]) & set(PDList2[j]))
      Common2=len(intersection2)
      value1=0
      if  (a1>0) and (b1>0):
        value1=(Common1*Common1)/(a1*b1)
      value2=0
      if  (a2>0) and (b2>0):
        value2=(Common2*Common2)/(a2*b2)  
      value1=(1-β)*value1+β*value2
      scores=value1           
      WeightD[i][j]=scores
      WeightD[j][i]=scores
      if scores>MaxScore:
         MaxScore=scores
      if scores<MinScore:
         MinScore=scores        
  for i in range(0,listlen-1):
   for j in range(i+1,listlen):
       WeightD[i][j]=(WeightD[i][j]-MinScore)/(MaxScore-MinScore)
       WeightD[j][i]=WeightD[i][j] 

 
def LoadPPI():
  global TrainProteinList
  global proteinlist
  global TestProteinList
  TempList=[]
  CAFAList=[]
  with open(Species+'_GO2024'+'.txt', 'r') as file:
    for line in file:
      line = line.strip()
      beginstr,endstr,types=line.split('\t')
      if types!=GOType:
         continue      
      if beginstr not in TempList:
         TempList.append(beginstr)

  if CAFAStr!='':
    TrainProteinList=[]
    with open(Species+'_GO2024'+CAFAStr+'.txt', 'r') as file:
     for line in file:
      line = line.strip()
      beginstr,endstr,types=line.split('\t')
      if types!=GOType:
         continue      
      if beginstr not in TrainProteinList:
         TrainProteinList.append(beginstr)    
    CAFAList=list(set(TrainProteinList) - set(TempList))
    TempList=list(set(TempList + TrainProteinList))

  PList=[]
  with open(Species+'_PPI2024.txt', 'r') as file:
    for line in file:
      line = line.strip()
      beginstr,endstr=line.split('\t')
      #if not ((beginstr in TempList) and (endstr in TempList)):
      #  continue
      if beginstr not in PList:
         PList.append(beginstr)
      if endstr not in PList:
         PList.append(endstr)        
    list1 = list(set(PList))
    list2 = list(set(TempList))
    proteinlist = [item for item in list1 if item in list2] #求两个列表的交集
    listlen=len(proteinlist)

    file.seek(0)
    global PPMatrix
    PPMatrix = [[0 for _ in range(listlen)] for _ in range(listlen)]
    global NList
    NList=[[0 for j in range(0)] for i in range(listlen)]
    global N2List
    N2List=[[0 for j in range(0)] for i in range(listlen)]
    for line in file:
        line = line.strip()
        beginstr,endstr=line.split('\t')
        if (beginstr not in TempList) or (endstr not in TempList):
          continue
        Ipos=proteinlist.index(beginstr)
        JPos=proteinlist.index(endstr)
        PPMatrix[Ipos][JPos]=1
        PPMatrix[JPos][Ipos]=1  
        NList[Ipos].append(JPos)
        NList[JPos].append(Ipos)
        
    for i in range(0,listlen):
      for j in range(0,len(NList[i])):
         IPos=int(NList[i][j])
         for k in range(0,len(NList[IPos])):
            IPos2=int(NList[IPos][k])
            if (IPos2!=i) and (IPos2 not in N2List[i]):
              N2List[i].append(IPos2)
  if CAFAStr!='':
    TestProteinList=list(set(proteinlist) - set(TrainProteinList))
    #print(len(TrainProteinList),len(TestProteinList))
  with open(Species+'_GO2024'+'.txt', 'r') as file:
    for line in file:
      line = line.strip()
      beginstr,endstr,types=line.split('\t')
      if types!=GOType:
         continue
      if beginstr not in proteinlist:
         continue 
      global GoList
      if endstr not in GoList:
         GoList.append(endstr)         
  if CAFAStr!='':
   with open(Species+'_GO2024'+CAFAStr+'.txt', 'r') as file_CAFA:
    for line in file_CAFA:
      line = line.strip()
      beginstr,endstr,types=line.split('\t')
      if types!=GOType:
         continue
      if beginstr not in proteinlist:
         continue 
      if endstr not in GoList:
         GoList.append(endstr)  
         
  global PGMatrix
  PGMatrix=[[0 for j in range(len(GoList))] for i in range(listlen)]
  global PGList
  PGList=[[0 for j in range(0)] for i in range(listlen)]
  global PGONum
  PGONum=[0 for i in range(listlen)] 
  with open(Species+'_GO2024'+'.txt', 'r') as file:
   for line in file:
      line = line.strip()
      beginstr,endstr,types=line.split('\t')
      try:
          Ipos=proteinlist.index(beginstr)
      except ValueError:
          Ipos=-1
      if types!=GOType:
         continue    
      if Ipos==-1:
         continue     
      JPos=GoList.index(endstr)
      PGMatrix[Ipos][JPos]=1
      PGList[Ipos].append(JPos) 
      PGONum[Ipos]=PGONum[Ipos]+1 
  if CAFAStr!='':
   with open(Species+'_GO2024'+CAFAStr+'.txt', 'r') as file:#只考虑在CAFA集合中出现而没有在Go2024中出现的蛋白质，也就说，CAFA和Go2024交集的蛋白质用Go2024的功能
    for line in file:
      line = line.strip()
      beginstr,endstr,types=line.split('\t')
      if beginstr not in CAFAList:
        continue
      try:
          Ipos=proteinlist.index(beginstr)
      except ValueError:
          Ipos=-1
      if types!=GOType:
         continue    
      if Ipos==-1:
         continue     
      JPos=GoList.index(endstr)
      PGMatrix[Ipos][JPos]=1
      PGList[Ipos].append(JPos) 
      PGONum[Ipos]=PGONum[Ipos]+1         
      
def LoadMultiData():
   #Loading Domain
   with open(Species+'_Domain2024.txt', 'r') as file:
    for line in file:
        line = line.strip()
        beginstr,endstr=line.split('\t')
        global Domainlist          
        if endstr not in Domainlist:
          Domainlist.append(endstr)      
    PSize=len(proteinlist)
    listlen=len(Domainlist)
    global PDMatrix
    global DPList
    PDMatrix=[[0 for j in range(listlen)] for i in range(PSize)]
    DPList=[[0 for j in range(0)] for i in range(len(Domainlist))]  
    
    file.seek(0)
    for line in file:
        line = line.strip()
        beginstr,endstr=line.split('\t')
        try:
          Ipos=proteinlist.index(beginstr)
        except ValueError:
          Ipos=-1
        if Ipos>=0:
          JPos=Domainlist.index(endstr)
          PDMatrix[Ipos][JPos]=1
          DPList[JPos].append(Ipos) 
     
  #Loadin Complex
   with open(Species+'_Protein_Complex.txt', 'r') as file:
    for line in file:
        line = line.strip()
        beginstr,endstr=line.split('\t')
        global Complexlist 
        if beginstr not in Complexlist:
           Complexlist.append(beginstr) 
 
    PSize=len(proteinlist)
    listlen=len(Complexlist)
    global PCMatrix
    global CPList
    global PCList
    PCMatrix=[[0 for j in range(listlen)] for i in range(PSize)]
    CPList=[[0 for j in range(0)] for i in range(listlen)]
    PCList=[[0 for j in range(0)] for i in range(PSize)]
    file.seek(0)
    for line in file:
        line = line.strip()
        beginstr,endstr=line.split('\t')
        try:
          Ipos=proteinlist.index(endstr)
        except ValueError:
          Ipos=-1

        if Ipos>=0:
          JPos=Complexlist.index(beginstr)
          PCMatrix[Ipos][JPos]=1
          CPList[JPos].append(Ipos) 
          PCList[Ipos].append(JPos) 
              

def max_path_product(G, source, target):
    all_paths = list(nx.all_simple_paths(G, source, target))
    if not all_paths:
        return 0    
    products = [reduce(lambda x, y: x * y, [G[u][v]['weight'] for u, v in zip(path, path[1:])], 1) for path in all_paths]
    max_product=max(products)
    return max_product

def GOHPro():
   global γ
   if GOType=='P':
     γ=0.8
   elif GOType=='C':  
     γ=0.9  
   else:
     γ=0.6
   #print('Weighting based on complex:', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
   Weight_Complex()
   #print('Weighting based on domain:', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
   Weight_Domain()
   G=nx.DiGraph()
   LP=len(proteinlist)
   LG=len(GoList)
   Matrix_PP=np.zeros((LP,LP))
   Matrix_PG=np.asarray(PGMatrix)
   Matrix_GG =np.zeros((LG,LG))
   Temp_GG =np.zeros((LG,LG))
   HengVector=np.zeros(len(proteinlist))
   ShuVector=np.zeros(len(proteinlist))
   TempMatrix_PG=np.zeros((LP,LG))
   CurrentMatrix_PG=np.zeros((LP,LG))
   PNeighborList=[[0 for j in range(0)] for i in range(LP)]
   GNeighborList=[[0 for j in range(0)] for i in range(LG)]
   CurrentMatrix=np.zeros((LP,LP))
   MatrixC=np.asarray(WeightC)
   MatrixD=np.asarray(WeightD)
   CurrentMatrix=MatrixC*γ+MatrixD*(1-γ)
   AVGN=sum(PGONum)/len(PGONum)
   AVGValue=math.ceil(AVGN)
   with open('Total_DAG.txt', 'r') as file:  
     for line in file:
      line = line.strip()
      beginstr,endstr,types,relation=line.split('\t')
      if types!=GOType:
         continue       
      if  relation=='is_a':        
        G.add_edge(beginstr,endstr,weight=0.4)
      else:
        G.add_edge(beginstr,endstr,weight=0.3)       
   
   #print('Get GG Matrix:', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))     
   for i in range(0,LG):    
      print(i+1,'/',LG,end='\r')
      for j in range(0,LG):
        if i==j:
          Matrix_GG[i][j]=1
          continue
        if GoList[i] in G.nodes() and GoList[j] in G.nodes():
            sp=max_path_product(G, GoList[i],GoList[j])
            Matrix_GG[i][j]=sp

   for i in range(LP-1):
      for j in range(i+1,LP):
        Matrix_PP[i][j]=CurrentMatrix[i][j]
        Matrix_PP[j][i]=Matrix_PP[i][j]
   
   for i in range(LP):
     for j in range(LP):
       if Matrix_PP[i][j]!=0:
         PNeighborList[i].append(j)
   for i in range(LG):
     for j in range(LG):
       if Matrix_GG[i][j]!=0:
         GNeighborList[i].append(j)   

   for i in range(LP):  
     sumi=0
     sumj=0
     for j in range(LP):
        sumi+=Matrix_PP[i][j]
        sumj+=Matrix_PP[j][i]
     HengVector[i]=sumi
     ShuVector[i]=sumj
  
   for i in range(LP):
      print(i+1,'/',LP,end='\r')
      for j in range(LP):
        sumi=HengVector[i]
        sumj=ShuVector[j]
        if sumi*sumj!=0:
          Matrix_PP[i][j]=Matrix_PP[i][j]/math.sqrt(sumi*sumj)
        else:
          Matrix_PP[i][j]=0   
   
   for i in range(LG):
      print(i+1,'/',LG,end='\r')
      for j in range(LG):
        sumi=0
        sumj=0
        for k in range(LG):
          sumi+=Matrix_GG[i][k]
          sumj+=Matrix_GG[k][j]
        if sumi*sumj!=0:
          Matrix_GG[i][j]=Matrix_GG[i][j]/math.sqrt(sumi*sumj)
        else:
          Matrix_GG[i][j]=0
  
   KNum=len(GoList)
   predictnum=0
   AvgPrecision=np.zeros(len(GoList))
   AvgRecall=np.zeros(len(GoList))
   AvgFPR=np.zeros(len(GoList))
   AvgPrecision0=0
   AvgRecall0=0
   AvgFPR0=0
   MatchOne=0  #match at least one function  
   PredictMatch=0 # number of protein with predicted functions perfect matched
   BenchbarkMatch=0 # number of protein with Benchbark functions perfect matched
   Benchbark=[]
   pridictedlist=[]
   Scores=dict()
   nums=0
   #wfile = open('temp.txt', "w")
   #print('Running GOHPro,γ=',γ, datetime.now().strftime('%Y-%m-%d %H:%M:%S'))  
   Fmax=0
   #for i in range(2):
   for i in range(len(proteinlist)):
       print(i+1,'/',len(proteinlist),',GOHPro',end='\r')
       if PGONum[i]==0:
         continue
       if (CAFAStr!='') and (proteinlist[i] not in TestProteinList):
         continue
       count=0
       predictnum=predictnum+1
       InitMatrix_PG=np.copy(PGMatrix)
       
       maxvalue=0
       maxindex=-1
       for  l in range(LP):
           if maxvalue<Matrix_PP[i][l]:
             maxvalue=Matrix_PP[i][l]
             maxindex=l 
       
       for k in range(LG):
          sumk=0
          for  l in range(len(PNeighborList[i])):
            if (proteinlist[int(PNeighborList[i][l])] in TestProteinList):
              continue                
            for m in range(len(GNeighborList[k])):
              sumk+=Matrix_PP[i][int(PNeighborList[i][l])]*PGMatrix[int(PNeighborList[i][l])][int(GNeighborList[k][m])]*Matrix_GG[k][int(GNeighborList[k][m])]

          InitMatrix_PG[i][k]=sumk
       CurrentMatrix_PG=np.copy(InitMatrix_PG)
       if (CAFAStr!=''):
          for j in range(len(proteinlist)):
             if (i==j) or (proteinlist[j]  in TestProteinList):
               CurrentMatrix_PG[j,:]=0
       while count<200:
        count=count+1
        #TempMatrix_PG=α*np.dot(np.dot(Matrix_PP, CurrentMatrix_PG), Matrix_GG)+(1-α)*InitMatrix_PG
        TempMatrix_PG=α*Matrix_PP@CurrentMatrix_PG@Matrix_GG+(1-α)*InitMatrix_PG
        norm = np.linalg.norm(TempMatrix_PG-CurrentMatrix_PG, ord=2)
        if norm>ε:
          CurrentMatrix_PG=np.copy(TempMatrix_PG)
        else:
          break     
       Benchbark=[]
       Scores=dict()
       prob=np.zeros((LG))
       
       for k in range(0,len(GoList)):
         if PGMatrix[i][k]==1:
           Benchbark.append(k)
         GoScore=0
         if CurrentMatrix_PG[i][k]>0:
           GoScore=CurrentMatrix_PG[i][k]
         Scores[k]=GoScore
         prob[k]=GoScore
       temp=np.sum(prob)
       if temp!=0:          
          prob_normalized = prob / temp 
       else:
          prob_normalized = prob
       ind_sorted_desc = np.flip(np.argsort(prob_normalized))
       pridictedlist0=[]
       if maxindex==-1:
           SelectedLen=AVGValue
       else:
           SelectedLen=PGONum[maxindex]
       for k in range(len(prob)):
            if k<SelectedLen:
              pridictedlist0.append(ind_sorted_desc[k])      
            else:
              break         
       Ipos=len(pridictedlist0)
       intersection=list(set(Benchbark)&set(pridictedlist0))
       matchnum=len(intersection)
       Pre=0       
       if Ipos!=0:
          Pre=matchnum * 1.0 / Ipos
       Rec=matchnum * 1.0 / PGONum[i]  
       AvgRecall0= AvgRecall0 +Rec
       AvgPrecision0= AvgPrecision0 +Pre
       if matchnum>0:
          MatchOne=MatchOne+1
       pridictedlist=[]
       for k in range(0,KNum):
           pridictedlist=ind_sorted_desc[:k+1]
           intersection=list(set(Benchbark)&set(pridictedlist))
           matchnum=len(intersection)  
           Pre=matchnum * 1.0 / (k+1)
           Rec=matchnum * 1.0 / PGONum[i]  
           AvgRecall[k]= AvgRecall[k] +Rec
           AvgPrecision[k]= AvgPrecision[k] +Pre 
           AvgFPR[k]=AvgFPR[k]+(k+1-matchnum) * 1.0 / (k+1)
   AUPR=0
   AUROC=0
   for k in range(0,KNum):   
      AvgPrecision[k]= AvgPrecision[k] / predictnum
      AvgRecall[k]= AvgRecall[k] / predictnum
      F_measure=2*AvgPrecision[k]*AvgRecall[k]/(AvgRecall[k]+AvgPrecision[k])
      AvgFPR[k]=AvgFPR[k]/ predictnum
      if k>0:  
          AUPR= AUPR + abs(AvgRecall[k] - AvgRecall[k - 1]) * abs(AvgPrecision[k] + AvgPrecision[k - 1]) * 0.5;
          AUROC=AUROC+abs(AvgFPR[k] - AvgFPR[k - 1]) * abs(AvgRecall[k] + AvgRecall[k - 1]) * 0.5;
      s='GOHPro'+'\t'+str(k+1)+'\t'+format(AvgPrecision[k],'.3f')+'\t'+format(AvgRecall[k],'.3f')
      #wfile.write(s+'\n') 
        #WL.append(s)
   AvgPrecision0= AvgPrecision0 / predictnum
   AvgRecall0= AvgRecall0 / predictnum
   F_measure0=2*AvgPrecision0*AvgRecall0/(AvgRecall0+AvgPrecision0)
   AvgFPR0=AvgFPR0/ predictnum
   s='GOHPro'+'\t'+str(predictnum)+'\t'+format(AvgPrecision0,'.3f')+'\t'+format(AvgRecall0,'.3f')+'\t'+format(F_measure0,'.3f')+'\t'+format(AUPR,'.3f')+'\t'+format(AUROC,'.3f')+'\t'+format(MatchOne/ predictnum,'.3f')
   print(s)
   #wfile.close()
   return AvgRecall,AvgPrecision,AvgFPR,AUPR,AUROC
   #return s
 
def main():

    print("Loading Data:"+Species+'_'+GOType+CAFAStr, datetime.now().strftime('%Y-%m-%d %H:%M:%S')) 
    LoadPPI()    
    LoadMultiData() 
    GOHPro()
    print('Over', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))    
    
if __name__ == "__main__":
	main()
   
