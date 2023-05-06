import json
import numpy as np

# Hyperparameters
MAX_LENGTH = 1500
embed_size = 128 #node2vec size
num_layers = 4
d_model = embed_size
dff = 512
num_heads = 8
EPOCHS = 100
dropout_rate = 0.1
BUFFER_SIZE = 20000
BATCH_SIZE = 64

# K_mer size and stride for extracting features from the genome sequences
k_mer = 4
stride = 1

# Should be true if you are running the pre-training for the first time
preProcess = True

# Change below to True when you want to extract the embedding layer weights for extracting features
save_embedding_weights = False

# Paths for loading/saving checkpoints and embedding files
checkpoint_path = "./Files/E2E/checkpoints/"
pre_train_checkpoint_path = "./Files/Pretraining_Graph/checkpoints/"
embedFile = "./Files/Pretraining_Graph/Embeddings/edgelist_4_mer_nseq_2151436.edgelist"
contextualized_embedFile = "./Data/Pretraining/contextualized_embeddings.npy"
nodeIDFile = "./Files/Pretraining_Graph/node_indices.json"

embedFile = "./Files/Pretraining_Graph/Embeddings/edgelist_4_mer_nseq_2151436.edgelist"
nodeIDFile = "./Files/Pretraining_Graph/node_indices.json"

with open(nodeIDFile, 'rb') as f:
	temp = json.load(f)
	nodeDict = {v:k for k,v in temp.items()}

contextualizedEmbedDict = np.load(contextualized_embedFile)
embedDict = {}

with open(embedFile, 'r') as file:
	for i,line in enumerate(file):
		line = line.replace('\n', '')
		if i > 0:
			l = line.split()
			embedding = np.array([float(i) for i in l[1:]])
			nodeID = int(l[0])
			if nodeID in nodeDict:
				embedDict[nodeDict[nodeID]] = embedding
			else:
				print("%d node not found in ID file."%(nodeID))

input_vocab_size = len(nodeDict.keys())*2
target_vocab_size = len(nodeDict.keys())*2

# Classes used in the experiments. Example given for what is used in the paper.
classes = ['host', 'H_somni', 'M_bovis', 'M_hemolytica', 'P_multocida', 'T_pyogenes', 'B_trehalosi']
numClasses = len(classes)