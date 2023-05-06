import os, sys, json, gzip
import numpy as np
from pyunpack import Archive
import shutil

inputPath = "./Data/Metagenomes/"
outputPath = "./Data/Metagenomes/Sequences/"

seqFiles = {f.split('.')[0]: os.path.join(inputPath, f) for f in os.listdir(inputPath) if 'fastq' in f and '72hr' in f}
seqMode = 'r'

look_up = {'CM008177': 'host', 'NC_020515':'B_trehalosi', 'NC_010519':'H_somni', 'NC_014760':'M_bovis', 'NZ_CP008918': 'P_multocida', 'NZ_CP007519': 'T_pyogenes', 'NC-110909': 'M_hemolytica', 'AB770484': 'PI-3', 'JF714967': 'BVDV2', 'NC_003045': 'BCV', 'NC_038272': 'BRSV', 'NZ_CP007519': 'T_pyogenes', 'NC_005261': 'BHV5', 'NC_001847': 'BHV1', 'NZ_CP006944': 'M_varigena'}

def extract_metagenomes(seqFiles, outputPath):
	for currID, seqFile in seqFiles.items():
		# Host Reads
		reads = []
		lineNo = 0
		print(currID, seqFile)

		outputFile = os.path.join(outputPath, '%s_%s.txt'%(currID, 'all'))
		if os.path.exists(outputFile):
			print("Existing file: %s. Skipping..."%(outputFile))
			continue
		with open(seqFile, seqMode) as file:
			for line in file:
				lineNo += 1
				line = line.replace("\n", "")
				if line.startswith('@'):
					seqID = line.split(' ')[0].replace('@', '').replace(' ', '')
					validID = lineNo + 1
				if lineNo == validID:
					reads.append((seqID, line))


		if not os.path.exists(outputPath):
			os.makedirs(outputPath)

		with open(outputFile, 'w') as of:
			for seqID, read in reads:
				of.write(seqID + ": " + read + "\n")

	print("fin")	

def extract_labels(inputPath):
	inputFiles = [os.path.join(inputPath, f) for f in os.listdir(inputPath) if f.endswith('.txt')]
	for inputFile in inputFiles:
		currID = inputFile.split(os.sep)[-1].split('.')[0]
		labels = []
		with open(inputFile, 'r') as file:
			for line in file:
				lineID = line.split(':')[0].split('.')[0]
				if lineID in look_up:
					labels.append(classes_simulated.index(look_up[lineID]))
				elif lineID[:2] == 'NC':
					labels.append(classes_simulated.index('M_hemolytica'))
				else:
					print("unknown sequence: %s"%line)
					sys.exit(0)
		outputFile = os.path.join(inputPath, '%s_%s.npy'%(currID, 'labels'))
		np.save(outputFile, np.array(labels))
extract_metagenomes(seqFiles, outputPath)
