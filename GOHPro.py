import warnings
import math
import numpy as np
from math import factorial
from datetime import datetime
import random
import networkx as nx
import os
from functools import reduce
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from joblib import Parallel, delayed
from collections import Counter
import time

PPMatrix=[]
PGMatrix=np.empty_like([])
PDMatrix=np.empty_like([])
PCMatrix=np.empty_like([])
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
TenFold=[] 
train_index=[]
test_index=[]
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
        if (CAFAStr!='') and (CPList[i][k] in test_index):
          continue
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
   return WeightC
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
  return WeightD



def ordered_unique(lst):
    seen = set()
    return [x for x in lst if not (x in seen or seen.add(x))]
 
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
      if beginstr not in PList:
         PList.append(beginstr)
      if endstr not in PList:
         PList.append(endstr)        

    list1 = ordered_unique(PList)  
    list2 = ordered_unique(TempList)   
    proteinlist = [item for item in list1 if item in list2]  
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
  #PGMatrix = np.zeros((listlen, len(GoList))) 
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
   with open(Species+'_GO2024'+CAFAStr+'.txt', 'r') as file:
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
  if (CAFAStr!=''):
      global train_index
      global test_index
      train_index=[]
      test_index=[]
      for i in range(len(proteinlist)):
        if proteinlist[i] in TestProteinList:
          test_index.append(i)
        else:
          train_index.append(i)       
      
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
   Weight_Complex()
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
   AvgPrecision=0
   AvgRecall=0
   Benchbark=[]
   pridictedlist=[]
   Scores=dict()
   nums=0

   Fmax=0

   all_predictions = np.zeros((len(proteinlist), len(GoList)))

   def process_one_protein(i):
       print(i+1,'/',len(proteinlist),',GOHPro',end='\r')
       Scores=np.zeros(LG)
       if PGONum[i]==0:
         return Scores
       if (CAFAStr!='') and (proteinlist[i] not in TestProteinList):
         return Scores
       count=0
       InitMatrix_PG=np.copy(PGMatrix)   
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
        TempMatrix_PG=α*Matrix_PP@CurrentMatrix_PG@Matrix_GG+(1-α)*InitMatrix_PG
        norm = np.linalg.norm(TempMatrix_PG-CurrentMatrix_PG, ord=2)
        if norm>ε:
          CurrentMatrix_PG=np.copy(TempMatrix_PG)
        else:
          break     
       Scores=CurrentMatrix_PG[i]          
       return Scores
   results = Parallel(n_jobs=-1)(delayed(process_one_protein)(i) for i in range(LP))
   for i, y_score in enumerate(results):
        for j in range (len(y_score)):
          all_predictions[i, j] = y_score[j]      
   PredictResults=[]
   for i in range(LP):
        if (CAFAStr!='') and (i not in test_index):
          continue
        maxvalue=0
        maxindex=-1
        for  l in range(LP):
           if maxvalue<Matrix_PP[i][l]:
             maxvalue=Matrix_PP[i][l]
             maxindex=l 
        if maxindex==-1:
           SelectedLen=AVGValue
        else:
           SelectedLen=PGONum[maxindex]
        Benchbark=[]
        prob=np.zeros((len(GoList)))
        for k in range(0,len(GoList)):
          if PGMatrix[i][k]==1:
            Benchbark.append(k)
          prob[k]=all_predictions[i, k]             
        ind_sorted_desc = np.flip(np.argsort(prob)) 
        pridictedlist=[]  
        for k in range(len(prob)):  
            if k<SelectedLen:
              pridictedlist.append(ind_sorted_desc[k])      
            else:
              break 
        Ipos=len(pridictedlist)    
        intersection=list(set(Benchbark)&set(pridictedlist))
        matchNum=len(intersection)
        Rec=matchNum * 1.0 / PGONum[i]
        AvgRecall= AvgRecall +Rec
        Pre=0        
        if Ipos!=0:
          Pre=matchNum * 1.0 / Ipos  
        AvgPrecision= AvgPrecision +Pre 
   if (CAFAStr!=''):
       predictnum=len(test_index)
   else:
       predictnum=LP       
   AvgPrecision= AvgPrecision / predictnum
   AvgRecall= AvgRecall / predictnum
   F_measure=2*AvgPrecision*AvgRecall/(AvgRecall+AvgPrecision)
  
   if (CAFAStr!=''):
      return F_measure,all_predictions[test_index, : ]
   else:
      return F_measure,all_predictions


def parse_go_data(file_path):
    G = nx.DiGraph()
    with open(file_path, 'r') as f:
        for line in f:
            go1, go2,types, relationship = line.strip().split('\t')
            if types!=GOType:
               continue  
            if go1 not in GoList or  go2 not in GoList:
               continue                
            if relationship == 'is_a' or relationship == 'part_of':
                G.add_edge(go1, go2)
    return G

def calculate_ic(G, true_go_terms): 
    all_go_terms = set()
    for terms in true_go_terms.values():
        valid_terms = [term for term in terms if term in G.nodes()]
        all_go_terms.update(valid_terms)
    term_counts = {term: 0 for term in all_go_terms}
    for terms in true_go_terms.values():
        valid_terms = [term for term in terms if term in G.nodes()]
        for term in valid_terms:
            ancestors = nx.descendants(G, term) | {term}
            for ancestor in ancestors:
                if ancestor in term_counts:
                    term_counts[ancestor] += 1
    num_proteins = len(true_go_terms)
    ic = {}
    for term, count in term_counts.items():
        prob = (count + 1e-10) / num_proteins  
        ic[term] = -np.log(prob)  
    return ic
 
def semantic_similarity(G, ic, pred_terms, true_terms):
    valid_pred_terms = [term for term in pred_terms if term in G.nodes()]
    valid_true_terms = [term for term in true_terms if term in G.nodes()]
    if not valid_pred_terms or not valid_true_terms:
        return 0
    sum_max_ic = 0
    for pred_term in valid_pred_terms:
        max_ic = 0
        for true_term in valid_true_terms:
            common_ancestors = nx.lowest_common_ancestors.all_pairs_lowest_common_ancestor(G, [(pred_term, true_term)])
            for _, ancestor in common_ancestors:
                if ancestor in ic:
                    max_ic = max(max_ic, ic[ancestor])
        sum_max_ic += max_ic
    return sum_max_ic / len(valid_pred_terms)

def calculate_smin(best_threshold,all_predictions, true_go_terms, G, ic, GOType):
    binary_preds = (all_predictions >= best_threshold).astype(int)
    ru = 0.0  
    mi = 0.0  
    num_proteins = len(true_go_terms)
    for i, (protein, true_terms) in enumerate(true_go_terms.items()):
        pred_terms = [term for j, term in enumerate(G.nodes()) if binary_preds[i][j] == 1]
        sim = semantic_similarity(G, ic, pred_terms, true_terms)        
        true_terms = list(true_terms)
        if not true_terms:
            continue
        true_ic = sum(ic.get(term, 0) for term in true_terms) / len(true_terms)
        ru += (1 - sim) * true_ic        
        pred_terms = list(pred_terms)
        if not pred_terms:
            continue
        pred_ic = sum(ic.get(term, 0) for term in pred_terms) / len(pred_terms)
        mi += (1 - sim) * pred_ic   
    ru /= num_proteins
    mi /= num_proteins
    smin = np.sqrt(ru**2 + mi**2)
    return smin
    
def align_predictions(all_predictions, G):
    go_terms_in_G = set(G.nodes())
    valid_indices = [i for i, term in enumerate(GoList) if term in go_terms_in_G]
    all_predictions = all_predictions[:, valid_indices]
    return all_predictions
    
def Evaluations(all_predictions,true_go_terms):
    all_labels = np.copy(PGMatrix)
    if  (CAFAStr!=''):
       all_labels=all_labels[test_index, :] 
    

    y_true = all_labels.flatten()
    y_score = all_predictions.flatten()
    y_true = np.nan_to_num(y_true, nan=0)
    y_score = np.nan_to_num(y_score, nan=0)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    precision, recall, _ = precision_recall_curve(y_true, y_score)

    #--------------------------------------------------------------------------------------------------
    all_predictions_z = (all_predictions - np.mean(all_predictions)) / np.std(all_predictions)
    all_predictions_normalized = (all_predictions_z - np.min(all_predictions_z)) / (np.max(all_predictions_z) - np.min(all_predictions_z))
    sorted_indices = np.argsort(all_predictions_normalized, axis=1) 
    sorted_scores = np.array([all_predictions_normalized[i][indices] for i, indices in enumerate(sorted_indices)])
    sorted_labels = np.array([all_labels[i][indices] for i, indices in enumerate(sorted_indices)])
    sorted_scores = np.nan_to_num(sorted_scores, nan=0)
    sorted_labels = np.nan_to_num(sorted_labels, nan=0)
    
    thresholds = np.linspace(0, 1, 1000)[::-1]  
    best_fmax = 0.0
    best_threshold = 0.0
    for threshold in thresholds:
      binary_preds = (sorted_scores >= threshold).astype(int)     
      tp = np.sum((binary_preds == 1) & (sorted_labels == 1))
      fp = np.sum((binary_preds == 1) & (sorted_labels == 0))
      fn = np.sum((binary_preds == 0) & (sorted_labels == 1))      
      precision1 = tp / (tp + fp) if (tp + fp) > 0 else 0.0
      recall1 = tp / (tp + fn) if (tp + fn) > 0 else 0.0     
      f1 = 2 * precision1 * recall1 / (precision1 + recall1) if (precision1 + recall1) > 0 else 0.0    
      if f1 > best_fmax:
            best_fmax = f1
            best_threshold = threshold
    #------------------------------------------------------------------------------------   
    y_true_micro = sorted_labels.flatten()
    y_score_micro = sorted_scores.flatten()
    micro_auc = roc_auc_score(y_true_micro, y_score_micro)
    micro_aupr = average_precision_score(y_true_micro, y_score_micro)

    best_smin=0
    
    G = parse_go_data('Total_DAG.txt')
    all_predictions = align_predictions(all_predictions, G)
    ic = calculate_ic(G, true_go_terms)
    best_smin = calculate_smin(best_threshold,all_predictions, true_go_terms, G, ic, 'GOType')
       
    return {
        "fmax": best_fmax,
        "smin": best_smin,
        "auroc": micro_auc,
        "aupr": micro_aupr,
        "roc_curve": (fpr, tpr),
        "pr_curve": (precision, recall)
    }

def plot_curves_with_metrics(method_results):
    true_go_terms = {}
    for i in range(len(proteinlist)):
         if  (CAFAStr!='') and i not in (test_index):
           continue
         protein_id =proteinlist[i]
         associated_go_terms = [GoList[j] for j in range(len(GoList)) if PGMatrix[i][j] == 1]
         true_go_terms[protein_id] = associated_go_terms
     
    plt.figure(figsize=(6, 5)) 

    metrics = {} 
    for name, pred in method_results.items():

        result = Evaluations(pred['Predictions'],true_go_terms)
        Avg_F_Measure=pred['F_measure']
        precision, recall = result["pr_curve"]
        metrics[name] = {
            "F_measure": Avg_F_Measure,
            "Fmax": result["fmax"],
            "Smin": result["smin"],
            "AUROC": result["auroc"],
            "AUPR": result["aupr"],
            "precision": precision,
            "recall": recall
        } 
        fpr, tpr = result["roc_curve"]
        plt.plot(fpr, tpr, label=f'{name} (AUC={result["auroc"]:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=0.5)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc='lower right')
    plt.tight_layout()

    output_file = Species+'_'+GOType+CAFAStr+'_ROC.jpg'
    plt.savefig(output_file, dpi=720)
    plt.close()

    plt.figure(figsize=(6, 5))
    #plt.subplot(1, 2, 2)
    for name, pred in metrics.items():
        precision=pred["precision"]
        recall =pred["recall"]
        plt.plot(recall, precision, label=f'{name} (AUPR={pred["AUPR"]:.3f})')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('PR Curves')
    plt.legend(loc='upper right')

    plt.tight_layout()
    #plt.show()
    output_file = Species+'_'+GOType+CAFAStr+'_PR.jpg'
    plt.savefig(output_file, dpi=720)
    plt.close()

    print("\nMetrics Summary:")
    print("{:<15} {:<10} {:<10} {:<10} {:<10} {:<10}".format("Method", "F_measure","Fmax","Smin", "AUROC", "AUPR"))
    for name, data in metrics.items():
        print("{:<15}  {:<10.3f} {:<10.3f} {:<10.3f} {:<10.3f}".format(
            name, data["Fmax"], data["Smin"], data["AUROC"], data["AUPR"]))                 
      
def main():  
    print("Loading Data:"+Species+'_'+GOType+CAFAStr, datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    LoadPPI()    
    LoadMultiData()
    F_measure1,all_predictions1=GOHPro()              
    method_results = {}
    method_results['GOHPro']={"F_measure": F_measure1, "Predictions": all_predictions1}
    plot_curves_with_metrics(method_results)   
    print('Over', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))    
    
if __name__ == "__main__":
	main()
   