import os, sys, json, gzip
import numpy as np
from config import *
import random, gc

k_mer = 4
stride = 1
bases = ['A', 'T', 'C', 'G']

tempPath = './tempFiles/'

inputPath = "./Data/Metagenomes/Sequences/"
outputPath = "./Data/Metagenomes/Representations/"


if not os.path.exists(outputPath):
	os.makedirs(outputPath)

nodeDict1 = {v:k+1 for k,v in nodeDict.items()}

def extract_global_n2v(input_files, embedDict, bases, n2v_path):
	nf = []
	representations = {c:[] for c in input_files.keys()}
	print("a", representations)
	for className,files in input_files.items():
		c = 0
		for input_file in files:
			c += 1
			print(className, input_file)
			lineNo = 0
			with open(input_file, 'r') as file:
				for line in file:
					lineNo += 1
					print("\r Processed: %06d."%lineNo, end="")
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
							# print(mer + "not found!!")
							nf.append(mer)
							temp.append(np.zeros(128,))
							# sys.exit(0)
					if not temp:
						temp.append(np.zeros(128,))
					temp = np.array([np.array(i) for i in temp])
					representations[className].append(np.reshape(temp.mean(0), (1, -1)))
			ofName = os.path.join(n2v_path, '%s_embeddings.npy'%(className))
			np.save(ofName, np.vstack(representations[className]))
			del(representations[className])
			gc.collect()
	return representations

def extract_contextualized_n2v(input_files, embedDict, nodeDict, bases, n2v_path):
	nf = []
	representations = {c:[] for c in input_files.keys()}

	for className,files in input_files.items():
		c = 0
		for input_file in files:
			c += 1
			print(className, input_file)
			lineNo = 0
			with open(input_file, 'r') as file:
				for line in file:
					lineNo += 1
					print("\r Processed: %06d."%lineNo, end="")
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
							print(mer + "not found!!")
							nf.append(mer)
							temp.append(np.zeros(128,))
							sys.exit(0)
					if not temp:
						temp.append(embedDict[1296,...])
					temp = np.array([np.array(i) for i in temp])
					representations[className].append(np.reshape(temp.mean(0), (1, 128)))
			ofName = os.path.join(n2v_path, '%s_embeddings.npy'%(className))
			np.save(ofName, np.vstack(representations[className]))
			del(representations[className])
			gc.collect()

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


input_files = {}

for f in os.listdir(inputPath):
	input_files[f.split('.')[0]] = os.path.join(inputPath, f)
	
n2v_path = os.path.join(outputPath, 'contextualized_n2v')

if not os.path.exists(n2v_path):
	os.makedirs(n2v_path)
n2v_rep = extract_contextualized_n2v(input_files, contextualizedEmbedDict, nodeDict1, bases, n2v_path)

n2v_path = os.path.join(outputPath, 'global_n2v')

if not os.path.exists(n2v_path):
	os.makedirs(n2v_path)
n2v_rep = extract_global_n2v(input_files, embedDict, bases, n2v_path)

global_n2v = os.path.join(outputPath, 'global_n2v')
contextualized_n2v = os.path.join(outputPath, 'contextualized_n2v')

combine_features([global_n2v, contextualized_n2v], os.path.join(outputPath, 'global_and_contextualized_representations'))