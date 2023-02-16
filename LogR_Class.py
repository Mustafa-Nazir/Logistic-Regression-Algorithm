import numpy as np

class LogisticRegression():
    def __init__(self ,  rate = 0.001 , model_type = "BGA" , iteration = 1000):
        self.__rate = rate
        self.__model_type = model_type
        self.__iteration = iteration
    
    #properties
    def getRate(self):
        return self.__rate
    def setRate(self,value):
        self.__rate = value
        
    def getModelType(self):
        return self.__model_type
    
    def getIteration(self):
        return self.__iteration
    def setIteration(self,value):
        self.__iteration = value
        
    #Methods
    def __exp(self,x,q):
        dot = np.dot(x, q)
        hq = 1 / (1 + np.exp(-1*dot))
        return hq
    
    #Bacth Gradient Ascent
    def __BGA(self):
        #the theta values of the equation
        self.__Q = np.zeros((self.__shape[1]+1 , 1))
        for i in range(self.__iteration):
            #for every Q
            for j in range(self.__shape[1]+1):
                hq = self.__exp(self.__mX , self.__Q)
                y = self.__y.reshape(-1,1)
                _mX_j = self.__mX[:,j].reshape(-1,1)
                f = (y - hq) * _mX_j
                f = self.__rate * f.sum()
                self.__Q[j] += f
    
    #Multi Gradient Ascent     
    def __MBGA(self):
        k = self.__y.shape[1]
        #the theta values of the equation
        self.__Q = np.zeros((k,self.__shape[1]+1))
        #for every K element
        for i in range(k):
            Q = self.__Q[i].reshape(-1,1)
            for j in range(self.__iteration):
                #for every Q
                for l in range(self.__shape[1]+1):                  
                    y = self.__y[:,i].reshape(-1,1)
                    _mX_l = self.__mX[:,l].reshape(-1,1)
                    hq = self.__exp(self.__mX , Q)
                    f = (y - hq) * _mX_l 
                    f = self.__rate * f.sum()
                    Q[l] += f
            self.__Q[i] = Q.reshape(1,-1)
                            
    #train the model according to model type
    def __train(self):
        if(self.__model_type == "BGA"):
            self.__BGA()
        elif(self.__model_type == "MBGA"):
            self.__MBGA()
             
    def train(self, x ,y):
        self.__x = x
        self.__y = y
        self.__shape = self.__x.shape
        #modified X
        self.__mX = np.concatenate([np.ones((self.__shape[0],1)) , self.__x] , axis = 1)
              
        #train the model
        self.__train()
        
    def predict(self, x):
        _mX = np.concatenate([np.ones((x.shape[0],1)) , x] , axis = 1)
        if(self.__model_type == "MBGA"):
            return self.__exp(_mX , self.__Q.T)
        return self.__exp(_mX , self.__Q)
        
        
        