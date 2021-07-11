import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from GA import Genetic_algorithm
# Kianoush Aqabakee
# Apache License 2.0
class RBFnn():
    def __init__(self, number_of_centers):
        self.c_num=number_of_centers
    def sigma_estimator(self):
        self.sigma=np.zeros([self.input_num,self.c_num])
        dist_center=np.zeros([self.input_num,self.c_num,self.c_num])
        for uu in range(self.input_num):
            for k in range(0,self.c_num):
                for l in range(k,self.c_num):
                    dist_center[uu][k][l]=np.abs(self.centers[uu][l]-self.centers[uu][k])
        p=self.c_num*(self.c_num-1)/2
        Alpha=0.3 # Heuristically
        for uu in range(self.input_num):
            self.sigma[uu]+=Alpha/p*np.sum(dist_center[uu],axis=0)
    def H_estimator(self,inputdata):
        h=np.zeros([self.input_num,self.c_num,inputdata.shape[1]])
        for uu in range(self.input_num):
            for k in range(0,self.c_num):
                R=0.5*self.sigma[uu][k]**2*np.identity((inputdata[uu]-self.centers[uu][k]).shape[0])
                h[uu][k]=np.exp(-np.dot(np.transpose((inputdata[uu]-self.centers[uu][k])),R,(inputdata[uu]-self.centers[uu][k])))
        H=np.ones(h.shape[2])
        for uu in range(self.input_num):
            H=np.multiply(H,h[uu])
        return(np.clip(H.T,0,10**3))
    def Train(self, inputdata, targetdata, number_of_inputs):
        self.inputdata=inputdata
        self.targetdata=targetdata
        if number_of_inputs==1:
            inputdata=inputdata.reshape(1,inputdata.size)
        self.input_num=number_of_inputs
        self.centers=np.zeros([number_of_inputs,self.c_num])
        for i in range(number_of_inputs):
            XX=np.array([inputdata[i,:],inputdata[i,:]]).T
            temp=KMeans(self.c_num,max_iter=40).fit(XX).cluster_centers_
            self.centers[i,:]=temp[:, 1]
        self.sigma_estimator()
        H=self.H_estimator(inputdata)
        self.weights=np.dot(np.linalg.pinv(H), targetdata)  #  pinv is pseudu of H * Y
        ga=Genetic_algorithm(MaxIt=51)
        U_=np.reshape(self.sigma,[self.sigma.size])
        result = list(ga.GA(self.error_RBF,n_params=U_.shape[0],initial_value= U_ ,))
        #result = list(ga.GA(targetdata,inputdata,self.error_RBF,
        #                    n_params=U_.shape[0],initial_value=np.zeros(U_.shape)  ,))
        U=result[0][:]
        self.sigma=np.reshape(U[0],[self.input_num,int(U[0].size/self.input_num)])
        H=self.H_estimator(inputdata)
        self.weights=np.dot(np.linalg.pinv(H), targetdata)  #  pinv is pseudu of H * Y
    def error_RBF(self,ind):
        x=self.inputdata
        t=self.targetdata
        t=np.reshape(t,[1,t.size])
        if self.input_num==1:
            x=np.reshape(x,[1,x.size])
            ind=np.reshape(ind,[1,ind.size])
        try:
            U=ind[0][:]
        except:
            U=ind
        self.sigma=np.reshape(U,[self.input_num,int(U.size/self.input_num)])
        self.ooo2=x
        H=self.H_estimator(x)
        self.ooo=H
        self.weights=np.dot(np.linalg.pinv(H), t[0,:])
        y=self.predict(x)-t
        y=y[0][:]
        return(0.5*np.sqrt(np.sum(y**2)/x.size))
    def predict(self, inputdata):
        if self.input_num==1:
            inputdata=inputdata.reshape(1,inputdata.size)
        H=self.H_estimator(inputdata)
        return(np.dot(H,self.weights))

inputdata=np.linspace(-3*np.pi,3*np.pi,300)
targetdata=np.sinc(inputdata*0.5)*np.sin(inputdata)

rbfQ=RBFnn(20)
rbfQ.Train(inputdata,targetdata,1)
y_hat=rbfQ.predict(inputdata)
plt.plot(inputdata,targetdata)
plt.plot(inputdata,y_hat)
plt.show()
##########
inputdata=np.linspace(-3*np.pi,3*np.pi,5000)
targetdata=np.sinc(inputdata*0.5)*np.sin(inputdata)

y_hat=rbfQ.predict(inputdata)
plt.plot(inputdata,targetdata,linewidth=2.0)
plt.plot(inputdata,y_hat,'r',linewidth=2.0)
plt.legend(['Actual data','Predicted data'])
plt.show()
