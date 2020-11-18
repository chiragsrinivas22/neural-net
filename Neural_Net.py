import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import random

dataset=pd.read_csv('pre_processed.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#splitting the dataset into  train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1)

#takes in a list and finds the Sigmoid of each value in the list and returns a list containing the sigmoids
def sigmoid(a):
	result=[]
	for i in a:
		result.append(1/(1+np.exp(-i)))
	return(result)

#takes in a list and finds the Relu of each value in the list and returns a list containing the relu values
def relu(a):
	result=[]
	for i in a:
		if(i<0):
			result.append(0)
		else:
			result.append(i)
	return(result)

#finds the square of the difference between true value and predicted value for a data point in the training set
def squared_error(y_true,y_prediced):
	return np.square(y_true - y_prediced)

#finds the Sigmoid derivative of a number
def sigmoid_derivative(a):
	return(a*(1-a))

#finds the Relu derivative of a number 
def relu_derivative(a):
	result=[]
	for i in a:
		if(i<0):
			result.append(0)
		else:
			result.append(1)
	return(result)

def training_accuracy(y_train,y_train_obs):
	for i in range(len(y_train_obs)):
		if(y_train_obs[i]>0.6):
			y_train_obs[i]=1
		else:
			y_train_obs[i]=0
	cm=[[0,0],[0,0]]
	fp=0
	fn=0
	tp=0
	tn=0
		
	for i in range(len(y_train)):
		if(y_train[i]==1 and y_train_obs[i]==1):
			tp=tp+1
		if(y_train[i]==0 and y_train_obs[i]==0):
			tn=tn+1
		if(y_train[i]==1 and y_train_obs[i]==0):
			fp=fp+1
		if(y_train[i]==0 and y_train_obs[i]==1):
			fn=fn+1
	cm[0][0]=tn
	cm[0][1]=fp
	cm[1][0]=fn
	cm[1][1]=tp
	print('Training:')
	print('Training Accuracy -',(tp+tn)/(tp+tn+fp+fn))

class NN:
	def __init__(self):
		self.no_of_input_nodes=9
		self.no_of_hidden_1_nodes=5
		self.no_of_output_nodes=1
		self.bias1_value=0
		self.bias2_value=0
		self.learning_rate=0.05

		#setting the initial weights using He initialization

		#the weights1 matrix is the weight matrix that stores the weights of the edges between input layer and hidden layer 1
		self.weights1=[]
		for i in range(self.no_of_input_nodes):
			l=[]
			for j in range(self.no_of_hidden_1_nodes):
				value=random.uniform(-np.sqrt(6/(self.no_of_input_nodes+self.no_of_hidden_1_nodes)),np.sqrt(1.414*(6/(self.no_of_input_nodes+self.no_of_hidden_1_nodes))))
				l.append(value)
			self.weights1.append(l)
        
		#the weights2 matrix is the weight matrix that stores the weights of the edges between hidden layer 1 and output layer
		self.weights2=[]
		for i in range(self.no_of_hidden_1_nodes):
			l=[]
			for j in range(self.no_of_output_nodes):
				value=random.uniform(-np.sqrt(6/(self.no_of_hidden_1_nodes+self.no_of_output_nodes)),np.sqrt(1.414*(6/(self.no_of_hidden_1_nodes+self.no_of_output_nodes))))
				l.append(value)
			self.weights2.append(l)
        
		#the bias1 matrix stores the bias for the hidden layer 1
		self.bias1_matrix=[self.bias1_value for i in range(self.no_of_hidden_1_nodes)]

		#the bias2 matrix stores the bias for the output layer
		self.bias2_matrix=[self.bias2_value for i in range(self.no_of_output_nodes)]

	def forward_propagation(self,input_row):
		self.input_row=input_row

		#the x0 matrices for the last 2 layers will store the input x0 as 1
		self.x0_hidden_layer_1=np.array([1 for i in range(self.no_of_hidden_1_nodes)])
		self.x0_output_layer=np.array([1 for i in range(self.no_of_output_nodes)])

		#since we use Relu as the activation function for our hidden layer,we find the Relu values of output of hidden 1 layer
		self.output_of_hidden1=relu(np.dot(input_row,self.weights1) + self.bias1_matrix)

		#since we use Sigmoid as the activation function for our output layer,we find the Sigmoid value of output of output layer node.
		self.output=sigmoid(np.dot(self.output_of_hidden1,self.weights2) + self.bias2_matrix)[0]

		result=self.output
		return(result)
	
	def back_propagation(self,y_train):

		#changing the weights of weights2 and bias2 matrices
		a=2*(self.y_actual - self.output)
		b=sigmoid_derivative(self.output)
		c=np.array(self.output_of_hidden1)
		result1=a*b*c
		result1.shape=(self.no_of_hidden_1_nodes,1)
		d_weights2=result1
		self.weights2+=(self.learning_rate*d_weights2)
		self.bias2_matrix+=(self.learning_rate*(a*b*self.x0_output_layer))
		
		#changing the weights of weights1 and bias1 matrices
		a=np.dot(2*(self.y_actual - self.output) * sigmoid_derivative(self.output), self.weights2.T)* relu_derivative(self.output_of_hidden1)
		self.input_row.shape=(self.no_of_input_nodes,1)
		d_weights1=self.input_row*a
		self.weights1+=(self.learning_rate*d_weights1)
		a=np.dot(2*(self.y_actual - self.output) * sigmoid_derivative(self.output), self.bias2_matrix.T)* relu_derivative(self.output_of_hidden1)
		self.bias1_matrix+=(self.learning_rate*a*self.x0_hidden_layer_1)

	def fit(self,X,Y):

		#doing SGD for the training set with an epoch value = 300
		epoch=300

		while(epoch>=0):
			y_train_obs=[]
			index=0

			for input_row in X:
				self.y_actual=Y[index]
				result=self.forward_propagation(input_row)

				if(result>0.6):
					y_train_obs.append(1)
				else:
					y_train_obs.append(0)

				#we back propagate only if the square of the error between predicted and true value is greater than or equal to 0.1
				if(np.square(Y[index] - result)>=0.1):
					self.back_propagation(Y)

				index+=1
			epoch-=1

		return(y_train_obs)

	def predict(self,X):

		yhat=[]

		#for every data point in the test set, we store the predicted value in a list "yhat" and return it
		for input_row in X:
			yhat.append(self.forward_propagation(input_row))

		return(yhat)

	def CM(y_test,y_test_obs):

		for i in range(len(y_test_obs)):
			if(y_test_obs[i]>0.6):
				y_test_obs[i]=1
			else:
				y_test_obs[i]=0
		cm=[[0,0],[0,0]]
		fp=0
		fn=0
		tp=0
		tn=0
		
		for i in range(len(y_test)):
			if(y_test[i]==1 and y_test_obs[i]==1):
				tp=tp+1
			if(y_test[i]==0 and y_test_obs[i]==0):
				tn=tn+1
			if(y_test[i]==1 and y_test_obs[i]==0):
				fp=fp+1
			if(y_test[i]==0 and y_test_obs[i]==1):
				fn=fn+1
		cm[0][0]=tn
		cm[0][1]=fp
		cm[1][0]=fn
		cm[1][1]=tp
		p= tp/(tp+fp)
		r=tp/(tp+fn)
		f1=(2*p*r)/(p+r)
		print()
		print('Testing:')
		print("Confusion Matrix : ")
		print(cm)
		print(f"Precision : {p}")
		print(f"Recall : {r}")
		print(f"F1 SCORE : {f1}")
		print('Test Accuracy -',(tp+tn)/(tp+tn+fp+fn))


net=NN()
y_train_obs=net.fit(X_train,y_train)
y_test_obs=net.predict(X_test)
training_accuracy(y_train,y_train_obs)
NN.CM(y_test,y_test_obs)
