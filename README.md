# bert-dna
Incorporating Pre-training of Deep Bidirectional Transformers and Convolutional Neural Networks to Interpret DNA Sequences

Recently, language representation models have drawn a lot of attention in natural language processing (NLP) field due to their remarkable results. Among them, Bidirectional Encoder Representations from Transformers (BERT) has proven to be a simple, yet powerful language model that achieved novel state-of-the-art performance. BERT adopted the concept of contextualized word embeddings to capture the semantics and context of the words in which they appeared. In this study, we present a novel technique namely BERT-DNA by incorporating BERT-base multilingual model in bioinformatics to interpret the information of DNA sequences. We treated DNA sequences as sentences and transformed them into fixed-length meaningful vectors where 768- vector represents each nucleotide. We observed that our BERT-base features improved more than 5-10% in terms of sensitivity, specificity, accuracy, and MCC compared to the current state-of-the-art features in bioinformatics. Moreover, advanced experiments show that deep learning (as represented by convolutional neural networks) hold potential in learning BERT features better than other traditional machine learning techniques. In conclusion, we suggest that BERT and deep convolutional neural networks could open a new avenue in bioinformatic modeling using sequence information.

### Dependencies
- Python 3
- Tensorflow 1.x: https://www.tensorflow.org/
- BERT: https://github.com/google-research/bert

### Prediction step-by-step:
### Step 1
Use "extract_seq.py" file to generate JSON files
- *python extract_seq.py*

### Step 2
Use command line in "bert2json.txt" to train BERT model and extract features

### Step 3
Use "jsonl2csv.py" to transfrom JSON to CSV files:
- *python jsonl2csv.py json_file csv_file*

### Step 4
Use 6mAtraining.py to train CNN model on CSV files
