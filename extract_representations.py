import os, sys, json, gzip
import numpy as np
from config import *
import random

k_mer = 4
stride = 1
bases = ['A', 'T', 'C', 'G']

tempPath = './tempFiles/'
inputPath = './Data/Extracted_Sequences/'
outputPath = './Data/Baseline_Representations/'

train_files = "./input_files_train.json"
test_files = "./input_files_test.json"

nodeDict1 = {v:k+1 for k,v in nodeDict.items()}

input_files = {c:[] for c in classes}

for f in os.listdir(inputPath):
	input_files[f].append(os.path.join(inputPath, f))

input_files_train = json.load(open(train_files, 'r'))
input_files_test = json.load(open(test_files, 'r'))

def extract_global_n2v(input_files, embedDict, classes, bases):
	nf = []
	representations = {c:[] for c in classes}

	for className,files in input_files.items():
		c = 0
		for input_file in files:
			c += 1
			input_file = os.path.join(inputPath, input_file)

			with open(input_file, 'r') as file:
				for line in file:
					temp = []
					read = line.replace('\n', '').split(': ')[-1]
					read =  "".join([c if c.isalpha() else random.choice(bases) for c in read])
					for i in range(0, len(read), stride):
						mer = read[i:i+k_mer]
						if len(mer) != k_mer:
							continue
						if mer in embedDict:
							temp.append(embedDict[mer])
						else:
							print(mer + "not found!!")
							nf.append(mer)
							temp.append(np.zeros(128,))
					temp = np.array([np.array(i) for i in temp])
					representations[className].append(np.reshape(temp.mean(0), (1, -1)))
	return representations

def extract_contextualized_n2v(input_files, embedDict, nodeDict, classes, bases):
	nf = []
	representations = {c:[] for c in classes}

	for className,files in input_files.items():
		c = 0
		for input_file in files:
			c += 1
			input_file = os.path.join(inputPath, input_file)
			print(className, input_file)
			with open(input_file, 'r') as file:
				for line in file:
					temp = []
					read = line.replace('\n', '').split(': ')[-1]
					read =  "".join([c if c.isalpha() else random.choice(bases) for c in read])
					for i in range(0, len(read), stride):
						mer = read[i:i+k_mer]
						if len(mer) != k_mer:
							continue
						if mer in nodeDict:
							temp.append(embedDict[nodeDict[mer]])
						else:
							nf.append(mer)
							temp.append(embedDict[1296,...])
					temp = np.array([np.array(i) for i in temp])
					representations[className].append(np.reshape(temp.mean(0), (1, -1)))
	return representations
	
def combine_features(file_paths, outputPath):
	if not os.path.exists(outputPath):
		os.makedirs(outputPath)
	files = {}
	for inputPath in file_paths:
		for f in os.listdir(inputPath):
			if f not in files.keys():
				files[f] = []
			files[f].append(os.path.join(inputPath, f))
	for fileName, paths in files.items():
		outputFile = os.path.join(outputPath, fileName)
		temp = [np.load(p) for p in paths]
		temp = np.hstack(temp)
		np.save(outputFile, temp)

n2v_path = os.path.join(outputPath, 'global_n2v_v2')
if not os.path.exists(n2v_path):
	os.makedirs(n2v_path)

print(input_files_train)
n2v_rep = extract_global_n2v(input_files_train, embedDict, classes, bases)
for k,v in n2v_rep.items():
	print(k, classes.index(k))
	ofName = os.path.join(n2v_path, '%d_embeddings_train.npy'%(classes.index(k)))
	np.save(ofName, np.vstack(v))

n2v_path = os.path.join(outputPath, 'contextualized_n2v_v2')
if not os.path.exists(n2v_path):
	os.makedirs(n2v_path)

n2v_rep = extract_contextualized_n2v(input_files_train, contextualizedEmbedDict, nodeDict1, classes, bases)
for k,v in n2v_rep.items():
	print(k, classes.index(k))
	ofName = os.path.join(n2v_path, '%d_embeddings_train.npy'%(classes.index(k)))
	np.save(ofName, np.vstack(v))

global_n2v = './Data/Representations/global_n2v/'
contextualized_n2v = './Data/Representations/contextualized_n2v/'
mg2v = './Data/Representations/mg2v_encoder_representations/'

combine_features([global_n2v, contextualized_n2v], './Data/Representations/global_and_contextualized_representations/')