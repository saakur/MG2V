import json, os, sys, random, time
from config import *
import numpy as np
import multiprocessing as mp
from functools import partial


bases = ['A', 'T', 'C', 'G']

counter = None


def chunks(l, n):
	return [l[i:i+n] for i in range(0, len(l), n)]

def process_read(nodeDict, embedDict, reads, outFileName):
	p_reads = []
	for read in reads:
		temp = []
		if not read.isalpha():
			read =  "".join([c if c.isalpha() else random.choice(bases) for c in read])
		prevNode = None
		for i in range(0, len(read), stride):
			mer = read[i:i+k_mer]
			if mer not in embedDict:
				continue
			else:
				temp.append(nodeDict[mer])
		pad = lambda a,i : a[0:i] if len(a) > i else a + [0] * (i-len(a))
		temp = pad(temp, MAX_LENGTH)
		p_reads.append(np.stack(temp))

	count = len(p_reads)
	np.save(outFileName, np.reshape(np.stack(p_reads), (count, -1, 1)))

def prep_fastq_pretrain(filepath, embedDict, nodeDict):
	fcount = 0
	output_path = './Temp/'
	if not os.path.exists(output_path):
		os.makedirs(output_path)
	nodeDict = {v:k+1 for k,v in nodeDict.items()}
	input_files = [os.path.join(filepath, f) for f in os.listdir(filepath) if 'fastq' in f]

	for file in input_files:
		print(file)
		realFake = 0
		embeddings = []
		if 'meta' in file.lower():
			realFake = 1
			continue
		else:
			realFake = 0
		reads = []
		fcount = 0
		start_time = time.time()
		tot = 0
		with open(file, 'r') as f:
			count = 0
			for line in f:
				if not line[0].isalpha():
					continue
				else:
					reads.append(line.replace('\n', ''))
					count += 1
					tot += 1
				if count%100000 == 0:

					num_process = 10#int(os.cpu_count()*0.5)
					
					total = len(reads)
					chunk_size = int(total / num_process)
					slices = chunks(reads, chunk_size)
					jobs = []
					
					for s in slices:
						fcount += 1
						outFileName = os.path.join(output_path, '%d_tmp_%d.npy'%(realFake, fcount))
						j = mp.Process(target=process_read, args=(nodeDict, embedDict, s, outFileName))
						jobs.append(j)
					for j in jobs:
						j.start()
					for j in jobs:
						j.join()

					reads = []
					count = 0
			total = len(reads)
			chunk_size = int(total / num_process)
			slices = chunks(reads, chunk_size)
			jobs = []
			
			for s in slices:
				fcount += 1
				outFileName = os.path.join(output_path, '%d_tmp_%d.npy'%(realFake, fcount))
				j = mp.Process(target=process_read, args=(nodeDict, embedDict, s, outFileName))
				jobs.append(j)
			for j in jobs:
				j.start()
			for j in jobs:
				j.join()

			reads = []
			count = 0
		end_time = time.time()
		print("Done in %s seconds"%str(start_time - end_time), tot)
		start_time = time.time()
	return output_path, fcount



def list2Dict(embeddings, labels, perc=1.0):
	assert embeddings.shape[0] == labels.shape[0]
	if perc == 1.0:
		idx = list(range(embeddings.shape[0]))
	else:
		idx = random.sample(list(range(embeddings.shape[0])), int(perc*embeddings.shape[0]))	
	embedDict = {k:embeddings[k] for k in range(embeddings.shape[0]) if k in idx}
	labelDict = {k:labels[k] for k in range(embeddings.shape[0]) if k in idx}
	return embedDict, labelDict

