import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score

'''
Design of a Neural Network from scratch

*************<IMP>*************
Mention hyperparameters used and describe functionality in detail in this space
- carries 1 mark
'''

dataset = pd.read_csv('LBW_Dataset.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X)
X = imputer.transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

def sigmoid(a):
	result=[]
	for i in a:
		result.append(1/(1+np.exp(-i)))
	return(result)
	
def relu(a):
	result=[]
	for i in a:
		if(i<0):
			result.append(0)
		else:
			result.append(i)
	return(result)

def squared_error(y_train,y_train_obs):
	return np.square(y_train - y_train_obs)

def sigmoid_derivative(a):
	return(a*(1-a))

def relu_derivative(a):
	result=[]
	for i in a:
		if(i<0):
			result.append(0)
		else:
			result.append(1)
	return(result)

def relu_derivative_for_column_matrix(a):
	result=[]
	for i in a:
		if(i[0]<0):
			result.append([0])
		else:
			result.append([1])
	return(result)

class NN:
	def __init__(self):
		self.no_of_input_nodes=9
		self.no_of_hidden_1_nodes=7
		self.no_of_hidden_2_nodes=7
		self.no_of_output_nodes=1
		self.bias1_value=0.15
		self.bias2_value=0.15
		self.bias3_value=0.15
		self.learning_rate=0.05

		self.bias1=[self.bias1_value for i in range(self.no_of_hidden_1_nodes)]
		self.bias2=[self.bias2_value for i in range(self.no_of_hidden_2_nodes)]
		self.bias3=[self.bias3_value for i in range(self.no_of_output_nodes)]

		self.weights1 = np.random.rand(self.no_of_input_nodes,self.no_of_hidden_1_nodes)*np.sqrt(1/(self.no_of_input_nodes+self.no_of_hidden_2_nodes))
		self.weights1 = np.insert(self.weights1,0,self.bias1,axis=0)

		self.weights2 = np.random.rand(self.no_of_hidden_1_nodes,self.no_of_hidden_2_nodes)*np.sqrt(1/(self.no_of_hidden_1_nodes+self.no_of_output_nodes))
		self.weights2 = np.insert(self.weights2,0,self.bias2,axis=0)

		self.weights3 = np.random.rand(self.no_of_hidden_2_nodes,self.no_of_output_nodes)*np.sqrt(1/(self.no_of_hidden_2_nodes+self.no_of_output_nodes))
		self.weights3 = np.insert(self.weights3,0,self.bias3,axis=0)

	def forward_propagation(self,input_row):
		result=[]

		input_row=np.insert(input_row,0,1)								
		self.input_row=input_row

		self.output_of_hidden1=relu(np.dot(input_row,self.weights1))
		self.output_of_hidden1.insert(0,1)								

		self.output_of_hidden2=relu(np.dot(self.output_of_hidden1,self.weights2))
		self.output_of_hidden2.insert(0,1)	

		self.output_before_sigmoid=np.dot(self.output_of_hidden2,self.weights3)
		self.output=sigmoid(self.output_before_sigmoid)	

		result.append(self.output[0])
		return(result)

	def back_propagation(self,y_train):
		#changing the weight3 matrix 
		a=2*(self.y_actual - self.output[0])
		b=sigmoid_derivative(self.output[0])
		c=np.array(self.output_of_hidden2)
		result=a*b*c
		result.shape=(self.no_of_hidden_2_nodes+1,1)
		d_weights3=result
		self.weights3+=(self.learning_rate*d_weights3)

		#changing the weight2 matrix
		a=2*(self.y_actual - self.output[0])
		b=sigmoid_derivative(self.output[0])
		c=self.weights3
		d=relu_derivative(self.output_of_hidden2)
		e=np.array(self.output_of_hidden1)
		e.shape=(self.no_of_hidden_1_nodes+1,1)
		result=a*b*c*d*e
		d_weights2=result[:,1:self.no_of_hidden_2_nodes+1]
		self.weights2+=(self.learning_rate*d_weights2)

		#changing weights1 matrix
		a=2*(self.y_actual - self.output[0])
		b=sigmoid_derivative(self.output[0])
		c=self.weights3
		d=relu_derivative(self.output_of_hidden2)
		e=self.weights2
		a_b_c_d_e=np.dot(a*b*c*d,e)
		f=relu_derivative(self.output_of_hidden1)
		temp=np.dot(f,a_b_c_d_e)
		g=self.input_row
		g.shape=(self.no_of_input_nodes+1,1)
		d_weights1=g*temp
		self.weights1+=(self.learning_rate*d_weights1)


	''' X and Y are dataframes '''
	
	def fit(self,X,Y):
		'''
		Function that trains the neural network by taking x_train and y_train samples as input
		'''
		index=0
		for input_row in X:
			self.y_actual=Y[index]
			result=self.forward_propagation(input_row)
			if(np.square(Y[index] - result[0])>=0.15):
				self.back_propagation(Y)
			index+=1
	def predict(self,X):

		"""
		The predict function performs a simple feed forward of weights
		and outputs yhat values 

		yhat is a list of the predicted value for df X
		"""
		yhat=[]
		for input_row in X:
			yhat.append(self.forward_propagation(input_row)[0])
		return(yhat)


	def CM(y_test,y_test_obs):
		'''
		Prints confusion matrix 
		y_test is list of y values in the test dataset
		y_test_obs is list of y values predicted by the model

		'''
		for i in range(len(y_test_obs)):
			if(y_test_obs[i]>0.6):
				y_test_obs[i]=1
			else:
				y_test_obs[i]=0
		print(y_test_obs)
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

		# p= tp/(tp+fp)
		# r=tp/(tp+fn)
		# f1=(2*p*r)/(p+r)
		
		print("Confusion Matrix : ")
		print(cm)
		print("\n")
		# print(f"Precision : {p}")
		# print(f"Recall : {r}")
		# print(f"F1 SCORE : {f1}")
		print(accuracy_score(y_test, y_test_obs))

	def CM1(y_test,y_test_obs):
		'''
		Prints confusion matrix 
		y_test is list of y values in the test dataset
		y_test_obs is list of y values predicted by the model

		'''
		for i in range(len(y_test_obs)):
			if(y_test_obs[i]>0.6):
				y_test_obs[i]=1
			else:
				y_test_obs[i]=0
		print(y_test_obs)
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

		# p= tp/(tp+fp)
		# r=tp/(tp+fn)
		# f1=(2*p*r)/(p+r)
		
		print("Confusion Matrix : ")
		print(cm)
		print("\n")
		# print(f"Precision : {p}")
		# print(f"Recall : {r}")
		# print(f"F1 SCORE : {f1}")
		print(accuracy_score(y_test, y_test_obs))


net=NN()
net.fit(X_train,y_train)
y_test_obs=net.predict(X_test)
y_train_obs=net.predict(X_train)
NN.CM(y_test,y_test_obs)
NN.CM1(y_train,y_train_obs)



