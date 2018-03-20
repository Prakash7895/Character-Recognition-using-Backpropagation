import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
import os,sys,select
from pathlib import Path
import random
import math

# number of neurons in each layer
i_p=900
o_p=62
hidden=481

#output list
dig=[chr(48+i) for i in range(10)]
caps=[chr(65+i) for i in range(26)]
small=[chr(97+i) for i in range(26)]
OutPut=dig+caps+small

#sigmoid function
def sigm(x):
	v=1+math.exp(-x)
	return 1/v

#to calculate target value for each neuron in outer layer
def target(i,label):
	if(OutPut[i]==label):
		return 1
	else:
		return 0

#load the images from folder directory and writes the pixel value alongwith the label to a text file(Input_to_NN.txt)

#The given link(http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/EnglishHnd.tgz) contain images of size 1200x900, 
#so to reduce size to 30x30, set fx=40 and fy=30 in function load_images(folder) below.
def load_images(folder):
	file=open("Input_to_NN.txt","w+")
	for filename in os.listdir(folder):
		p=os.path.join(folder,filename)
		print("Dir : ",p)
		if os.path.isdir(p):
			for f in os.listdir(p):
				img=cv.imread(os.path.join(p,f),0)
				w,h=np.shape(img)
#				print("ORIGINAL w : ",w," h : ",h)
				img0=cv.resize(img,(0,0),fx=1/40,fy=1/30)
				w,h=np.shape(img0)
#				print("ORIGINAL w : ",w," h : ",h)
				file.write(filename)
				for i in range(w):
					for j in range(h):
						if(img0[i][j]==255):
							file.write("0")
						else:
							file.write("1")
				file.write("\r\n")
	file.close()

def check_and_delete(filename):
	pathname = os.path.dirname(sys.argv[0])
	full_path =os.path.abspath(pathname)
	filepath=os.path.join(full_path,filename)
	for f in os.listdir(full_path):
		p=os.path.join(full_path,f)
		if(p==filepath):
			os.remove(filepath)
			print("Previous file removed")

#print('sys.argv[0] =', sys.argv[0])
pathname = os.path.dirname(sys.argv[0])
full_path =os.path.abspath(pathname)
#print("FULL : ",full_path)
flag=False
for f in os.listdir(full_path):
	if(f=="Input_to_NN.txt"):
		flag=True
		break

if(flag==False):
	home = os.path.join(full_path,"Dataset")
	#dict['label']=[images]
	#print(home)
	load_images(home)

flag=False
for f in os.listdir(full_path):
	if(f=="Weights.txt"):
		flag=True
		break

inputLayer=[0 for i in range(i_p)]
hiddenLayer=[0 for i in range(hidden)]
outputLayer=[0 for i in range(o_p)]
hiddenbias=0.005
outputbias=0.004

if(flag==False):
	n=0.75
	#initial value to weights
	innerWeights=[[int(random.randint(1,10))/1000 for i in range(i_p)] for j in range(hidden)]
	outerWeights=[[int(random.randint(1,10))/1000 for i in range(hidden)] for j in range(o_p)]

	epochS=1
	epoch=0
	while(epoch <= epochS):
		print("EPOCH : ",epoch)
		#in each epoch it asks user to either increase epochS or not. if user does not reply it increases the epochS
		if(epoch>=0 and epoch==epochS):
			print("You have five seconds to answer!")
			print("Enter 'N' or 'n' to stop epochs here else 'Y' or 'y' : ")
			i, o, e = select.select( [sys.stdin], [], [], 5 )
			if (i):
				name=sys.stdin.readline().strip()
				print("You said", name)
				if(name=="Y" or name=="y"):
					epochS+=1
					print("Epoch increased , New Epoch : ",epochS)
				elif(name=="N" or name=="n"):
					print("Epoch constant, Epoch : ",epochS)
					break;
			else:
				print("You said nothing!")
				epochS+=1
				print("Epoch increased , New Epoch : ",epochS)

		file=open("Input_to_NN.txt","r")
		for line in file:
			label=line[0]
			for i in range(i_p):
				inputLayer[i]=int(ord(line[i+1])-48)
			
			cont=0
			while(True):
				for i in range(hidden):
					s=hiddenbias
					for j in range(i_p):
						s=s+innerWeights[i][j]*inputLayer[j]
					hiddenLayer[i]=sigm(s)
					
				for i in range(o_p):
					s=outputbias
					for j in range(hidden):
						s=s+outerWeights[i][j]*hiddenLayer[j]
					outputLayer[i]=sigm(s)

				error=0.0

				for i in range(o_p):
					error+=(0.5*(target(i,label)-outputLayer[i])**2)
				cont+=1
				if(error > 0.10):
					print("Label : ",label,end=' ')
					print("Count : ",cont,"  Error : ",error)
					
				if(error <= 0.10):
					break
				#Error in outer weights
				dEt_By_doutw=[[0 for i in range(hidden)] for j in range(o_p)]
				
				for i in range(o_p):
					diff=-(target(i,label)-outputLayer[i])*outputLayer[i]*(1-outputLayer[i])
					for j in range(hidden):
						dEt_By_doutw[i][j]=diff*hiddenLayer[j]

				#Error in inner Weights
				dEt_By_dinw=[[0 for i in range(i_p)] for j in range(hidden)]
				dEt_By_douth=[0 for j in range(hidden)]
				
				for i in range(hidden):
					s=0
					for k in range(o_p):
						s=s+(outputLayer[k]-target(k,label))*outputLayer[k]*(1-outputLayer[k])*outerWeights[k][i]
					dEt_By_douth[i]=s
				
				for i in range(hidden):
					diff=dEt_By_douth[i]*hiddenLayer[i]*(1-hiddenLayer[i])
					for j in range(i_p):
						dEt_By_dinw[i][j]=diff*inputLayer[j]
				
				#Updating weights
				for i in range(o_p):
					for j in range(hidden):
						outerWeights[i][j]-=(n*dEt_By_doutw[i][j])
				
				for i in range(hidden):
					for j in range(i_p):
						innerWeights[i][j]-=(n*dEt_By_dinw[i][j])

		#after each epoch it writes the current weights value to text file(Weights_for_epoch_(epochValue).txt)
		if(epoch>=0):
			print("Writing Weights.......!")
			filename="Weights_for_epoch_"+str(epoch)+".txt"
			epochweight=open(filename,"w+")
			epochweight.write("InnerWeights")
			epochweight.write("\r\n")
			for i in range(hidden):
				for j in range(i_p):
					epochweight.write(str(innerWeights[i][j]))
					epochweight.write(" ")
				epochweight.write("\r\n")
			epochweight.write("OuterWeights")
			epochweight.write("\r\n")
			for i in range(o_p):
				cnt1=0
				for j in range(hidden):
					epochweight.write(str(outerWeights[i][j]))
					epochweight.write(" ")
					cnt1+=1
				print("Length of "+str(i)+" : ",cnt1)
				epochweight.write("\r\n")
			#check_and_delete("Weights_for_epoch_"+str(epoch-5)+".txt")
		epoch+=1

	#these are final weights before u stop epoch, written in text file(Weights.txt)
	weight=open("Weights.txt","w+")
	weight.write("InnerWeights")
	weight.write("\r\n")
	for i in range(hidden):
		for j in range(i_p):
			weight.write(str(innerWeights[i][j]))
			weight.write(" ")
		weight.write("\r\n")
	weight.write("OuterWeights")
	weight.write("\r\n")
	for i in range(o_p):
		for j in range(hidden):
			weight.write(str(outerWeights[i][j]))
			weight.write(" ")
		weight.write("\r\n")

# get the weights value from text file
weight=open("Weights.txt","r")
innerW=0
outerW=0
innerWeights=[]
outerWeights=[]

for line in weight:
	if('InnerWeights' in line):
		innerW=1
		outerW=0
	elif('OuterWeights' in line):
		outerW=1
		innerW=0
	elif(innerW==1 and outerW==0):
		weig=line.split()
		for i in range(len(weig)):
			weig[i]=float(weig[i])
		innerWeights.append(weig)
	elif(innerW==0 and outerW==1):
		weig=line.split()
		for i in range(len(weig)):
			weig[i]=float(weig[i])
		outerWeights.append(weig)

#This for testing purpose
#Put a image from mnist dataset in InputImage folder.
home = os.path.join(full_path,"InputImage")

for f in os.listdir(home):
	p=os.path.join(home,f)
	inputImage=cv.imread(p,0)
	w,h=np.shape(inputImage)
	#print("Input Image  : ")
	#print("w : ",w," h : ",h)
	inputImage=cv.resize(inputImage,(0,0),fx=1/40,fy=1/30)
	w,h=np.shape(inputImage)
	#print("Input Image  : ")
	#print("w : ",w," h : ",h)
	for i in range(w):
		for j in range(h):
			if(inputImage[i][j]==255):
				#print("0",end='')
				inputLayer[i*30+j]=0
			else:
				#print("1",end='')
				inputLayer[i*30+j]=1
		#print()

	for i in range(hidden):
		s=hiddenbias
		for j in range(i_p):
			s=s+innerWeights[i][j]*inputLayer[j]
		hiddenLayer[i]=sigm(s)

	for i in range(o_p):
		s=outputbias
		for j in range(hidden):
			s=s+outerWeights[i][j]*hiddenLayer[j]
		outputLayer[i]=sigm(s)

	print("FileName : ",f," Character : ",OutPut[outputLayer.index(max(outputLayer))]," Value : ",max(outputLayer))
