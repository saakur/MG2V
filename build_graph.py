import os, sys, json, itertools, random
import numpy as np
from tqdm import tqdm

def weight_scale(prevWeight, newWeight):
	# Update previous weight by current weight
	weight = newWeight/np.linalg.norm([prevWeight, newWeight]) + prevWeight
	return 2 * max(weight - 1, 1) ** .5 + min(weight, 2) - 2

bases = ['A', 'T', 'C', 'G']
kmer_ids = {}
idx = itertools.count()

input_path = sys.argv[1]
output_path = sys.argv[2]


k_mer = int(sys.argv[3])
stride = 1


if not os.path.exists(input_path):
	raise ValueError('Input Path does not exist.')

if not os.path.exists(output_path):
	os.makedirs(output_path)

input_files = [os.path.join(input_path, f) for f in os.listdir(input_path) if 'fastq' in f]

graph = {}
tot_count = 0
num = 0
for file in input_files:
	count = 0
	num += 1
	bar = tqdm()
	with open(file, 'r') as f:
		for line in f:
			if not line[0].isalpha():
				continue
			else:
				read = line.replace('\n', '')
				read =  "".join([c if c.isalpha() else random.choice(bases) for c in read])
				prevNode = None
				for i in range(0, len(read), stride):
					mer = read[i:i+k_mer]
					if len(mer) != k_mer:
						continue
					if mer not in graph:
						graph[mer] = {}
					if mer not in kmer_ids:
						kmer_ids[mer] = next(idx)

					if prevNode is not None:
						if mer not in graph[prevNode]:
							graph[prevNode][mer] = 1
						else:
							graph[prevNode][mer] = weight_scale(graph[prevNode][mer], 1)
					prevNode = mer
				count += 1
				bar.update(1)
	tot_count += count
	bar.close()

node_idx_file = os.path.join(output_path, "node_indices.json")
with open(node_idx_file, 'w') as of:
	of.write(json.dumps(kmer_ids, indent=4, sort_keys=True))

output_file = os.path.join(output_path, "edgelist_%d_mer_nseq_%d.edgelist"%(k_mer, tot_count))
with open(output_file, 'w') as f:
	for n1, n2Dict in graph.items():
		for n2, wt in n2Dict.items():
			f.write('%d %d %f\n'%(kmer_ids[n1], kmer_ids[n2], wt))
