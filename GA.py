import random
import math
import numpy as np
import matplotlib.pyplot as plt
# Kianoush Aqabakee
# Apache License 2.0
class Genetic_algorithm:
    def __init__(self,MaxIt=51,nPop=20
                 ,pc=10,pm=10,gamma=0.2,mu=0.4,beta=8,VarMin=-10,VarMax= 10):
        self.MaxIt=MaxIt
        self.nPop=nPop
        self.pc=pc
        self.pm=pm
        self.gamma=gamma
        self.mu=mu
        self.beta=beta
        self.VarMin=VarMin
        self.VarMax=VarMax
    def RouletteWheelSelection(self,P):
        r=np.random.rand();
        C=np.zeros([self.nPop]);
        c0=0;
        for k1 in range(0,self.nPop):
            C[k1]=c0+P[k1];
            c0=C[k1];
        flag=1;
        for k1 in range(0,self.nPop):
                if (r<=C[k1] and flag==1):
                    i=[k1];
                    flag=0;
                else:
                    i=np.random.randint(0,self.nPop);
        return i
    def Crossover(self,i1,i2):
        x1=np.copy(self.popPosition[i1])
        x2=np.copy(self.popPosition[i2])
        alpha=np.random.uniform(-self.gamma,1+self.gamma,x1.size);
        y=np.zeros([2,x1.size]);
        x1=np.array(x1)
        x2=np.array(x2)
        alpha=np.array(alpha)
        y[0]=alpha*x1+(1-alpha)*x2;
        y[1]=alpha*x2+(1-alpha)*x1;

        y=np.clip(y,self.VarMin,self.VarMax)
        return y
    def Mutate(self,index):
        x=np.copy(self.popPosition[index])
        nVar=x.size;
        nmu=int(np.ceil(self.mu*nVar));
        sigma=0.1*(self.VarMax-self.VarMin);
        j=np.random.randint(0,nVar,nmu);
        y=x;
        y[j]=y[j]+sigma*np.random.randint(nmu+1)
        y=np.clip(y,self.VarMin,self.VarMax)
        
        return y
    def GA(self,fobj,n_params,initial_value=[]):#'dont have'):
        nm=round(self.pm*self.nPop)
        nc=2*round(self.pc*self.nPop/2)
        ## Initializations
        maxCosts=-100000;
        minCosts=100000;
        popPosition_r = np.random.rand(self.nPop,initial_value.shape[0])
        self.popPosition=self.VarMin+popPosition_r*(self.VarMax-self.VarMin)
        if(any(initial_value)):#'dont have'):
            self.popPosition[0,:]=initial_value
        self.popCost=np.asarray([fobj(ind) for ind in self.popPosition])

        ## Main Loop
        popcPosition=np.zeros([round(nc/2),n_params]);
        popcCost=np.zeros([round(nc/2),2]);
        popmPosition=np.zeros([nm,n_params])
        popmCost=np.zeros([nm]);
        popnew=np.zeros([self.nPop+round(nc/2)+nm,n_params]);
        popnewCost=np.zeros([self.nPop+round(nc/2)+nm]);
        Sort_popnew=np.zeros([popnew.shape[0],n_params]);
        Sort_popnewCost=np.zeros(popnew.shape[0]);
        for it in range(1,self.MaxIt):
            # Selection Probabilities
            Pr=np.zeros([self.nPop]);
            P=np.zeros([self.nPop]);
            sumPr=0;
            for i in range (0,self.nPop):
                Pr[i]=np.exp(-self.beta*self.popCost[i]/np.max(self.popCost));
                sumPr=Pr[i]+sumPr;
            for i in range (0,self.nPop):
                P[i]=Pr[i]/sumPr;
            popc=np.zeros([round(nc/2),2]);
            for k in range(0,round(nc/2)):
                # Select Parents Indices
                 i1=self.RouletteWheelSelection(P);
                 i2=self.RouletteWheelSelection(P);
                 while (i1==i2):
                     i2=self.RouletteWheelSelection(P);
                     
            # Apply Crossover
            popcPosition=self.Crossover(i1,i2);
            # Mutation
            for k in range(0,nm):
                i=np.random.randint(self.nPop)
                popmPosition[k]=self.Mutate(i);
            # Merge
            popnew=np.concatenate((self.popPosition,popcPosition,popmPosition))
            popnewCost=np.asarray([fobj(ind) for ind in popnew])

            index=np.argsort(popnewCost)
            Sort_popnewCost=np.sort(popnewCost)
            Sort_popnew=np.array([popnew[ind] for ind in index])
            # Truncation
            self.popPosition=Sort_popnew[0:self.nPop][:]
            self.popCost=Sort_popnewCost[0:self.nPop][:]
            print('Best unfitness in iteration %d: %f' % (it, self.popCost[0]))
        #print(fobj(self.popPosition[0]))
        yield self.popPosition[0],self.popCost[0]

