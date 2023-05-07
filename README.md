# MG2V
Official implementation for the MG2V framework proposed in the following papers:
  
  "Metagenome2Vec: Building Contextualized Representations for Scalable Metagenome Analysis", ICDMW, 2021. Paper: https://arxiv.org/abs/2111.08001
  
  "Scalable Pathogen Detection from Next Generation DNA Sequencing with Deep Learning", Under Review at Transactions on Big Data. Preprint: https://arxiv.org/abs/2212.00015
  

**How to Run the Code**

The entire process was broken down into individual steps to enable tractable computation on an academic budget. All model paths are set in `config.py`.  There are three steps to running the code:

1. First, you need to build and extract the graph-based representations from the metagenome(s) used in the pre-training stage. This step does not require a GPU although you will need decent amount of RAM (>= 64GB) to extract features for k-mers of size more than 5. To do that, you will need to run the following:

    * Extract the sequence reads from your fastq files into a text file using `extract_sequences.py`
    * Build the graph and extract the edgelist using `build_graph.py`
    * Extract node2vec representations for the edgelist from the previous step using `n2v_main.py`. Default parameters were used. 

2. Second, you will need to pre-train a transformer to extract contextualized representations. This is the most computationally expensive part of the whole process and usually takes about 10 hours to converge on a single Titan RTX GPU. Below are the steps to pre-train your model:
    * Ensure that the `preprocess` flag in `config.py` is set to *True* if running for the first time. This extracts k-mers and tokenizes them for use in the embedding layer.
    * Ensure that the `save_embedding_weights` flag is set to *True* to save the embeddings to a file.
    * Run `pre_train.py` to pre-train your transformer.

3. Finally, you can extract MG2V representations for sequences using `extract_metagenome_representations.py` if you have extracted all the sequence reads into a text file, as shown in the example code `extract_sequences.py`. 
    * Note that you can extract all three types of representations - MG2V, NG2V, and CN2V by changing parameters in `extract_metagenome_representations.py`.

Once representations are extracted, you can train your favorite machine learning model for classifying them.

If this is useful, remember to cite us!
