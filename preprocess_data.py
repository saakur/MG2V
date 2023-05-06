import os, sys, json, gzip
import numpy as np
from pyunpack import Archive
import shutil

tempPath = './tempFiles/'
inputPath = './Data/Metagenomes/'
outputPath = './Data/Extracted_Sequences/'
max_host = 1000

seqFiles = {f.split('.')[0]: os.path.join(inputPath, f) for f in os.listdir(inputPath) if '.fastq' in f and '.gz' not in f}
blastRARFiles = {f.split('.')[0]: os.path.join(inputPath, f) for f in os.listdir(inputPath) if '.rar' in f}

for currID, rarFile in blastRARFiles.items():

	if not os.path.exists(tempPath):
		os.makedirs(tempPath)

	count = 0
	lineNo = 0
	pathogenSeqIds = {}
	seqFile = seqFiles[currID]
	nonhost_seqs = []

	seqMode = 'r'

	if not os.path.exists(rarFile):
		RaiseError("RAR File not found!!!")
	if not os.path.exists(seqFile):	
		RaiseError("Metagenome Sequence File not found!!!")
	
	Archive(rarFile).extractall(tempPath)
	
	blastFiles = [os.path.join(tempPath, f) for f in os.listdir(tempPath) if '.blast' in f]
	
	for blastFile in blastFiles:
		pathogenName = '_'.join(blastFile.split(os.sep)[-1].split('_')[-1].split('.')[:-1]).replace(currID, '').replace('btrehalosi', 'B_Trehalosi').replace('P_multocidaa', 'P_multocida').replace('M_haemolytica', 'M_hemolytica')
		
		print("Processing ID %s and Pathogen %s"%(currID, pathogenName))
		if pathogenName not in pathogenSeqIds:
			pathogenSeqIds[pathogenName] = []

		with open(blastFile, 'r') as file:
			for line in file:
				line = line.replace("\n", "")
				if not line.startswith('>'):
					continue
				seqID = line.split(' ')[1]
				pathogenSeqIds[pathogenName].append(seqID)

		seqID = None
		reads = []
		readIDS = []

		lineNo = 0
		
		with open(seqFile, seqMode) as file:
			for line in file:
				lineNo += 1
				line = line.replace("\n", "")
				if line.startswith('@'):
					seqID = line.split(' ')[0].replace('@', '').replace(' ', '')
					if seqID in pathogenSeqIds[pathogenName]:
						count += 1
						readIDS.append(lineNo + 1)
				if lineNo in readIDS:
					reads.append((seqID, line))
			nonhost_seqs += readIDS
		if len(set(pathogenSeqIds[pathogenName])) != len(reads):
			with open('./Errors.txt', 'a') as ef:
				ef.write("No Match!" + ", " +  currID + ", " + pathogenName + ", " + str(len(set(pathogenSeqIds[pathogenName]))) + "\n")

		if reads:
			outputFile = os.path.join(outputPath, '%s_%s.txt'%(currID, pathogenName))
			with open(outputFile, 'w') as of:
				for seqID, read in reads:
					of.write(seqID + ": " + read + "\n")
	# Host Reads
	reads = []
	lineNo = 0

	with open(seqFile, seqMode) as file:
		for line in file:
			lineNo += 1
			line = line.replace("\n", "")
			if line.startswith('@'):
				seqID = line.split(' ')[0].replace('@', '').replace(' ', '')
				validID = lineNo + 1
			if lineNo not in nonhost_seqs and lineNo == validID and np.random.uniform() < 0.5 and len(reads) < max_host:
				reads.append((seqID, line))
	outputFile = os.path.join(outputPath, '%s_%s.txt'%(currID, 'host'))
	with open(outputFile, 'w') as of:
		for seqID, read in reads:
			of.write(seqID + ": " + read + "\n")

	shutil.rmtree(tempPath)
print("fin")	