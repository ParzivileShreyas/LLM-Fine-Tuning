# Full Implementation of fine-tuning an LLM on Hugging Face

I used the gemma-2b-it model (due to its size and ease of usage in typical hardware setups) and a publicly available dataset from Hugging Face, 
such as the "pubmed_qa" dataset (which contains medical reports, history, and observations that could be useful for impression generation).

# STEP BY STEP PROCESS :-

# Step 1: Setup the Environment
You need to install the following libraries:

1. transformers for working with the pre-trained model.
2. datasets for loading Hugging Face datasets.
3. torch for PyTorch-based fine-tuning.

# Step 2: Load the Dataset
We will use the "pubmed_qa" dataset from Hugging Face as a substitute for your medical dataset. 
It contains reports in the medical domain and can be useful for generating impressions.

# Step 3: Preprocess the Data
For fine-tuning, you need to tokenize the text. We will concatenate the Report Name, History, and Observation fields to form the input for the model.

# Step 4: Fine-tuning the GPT-2 Model
We will now use the "gpt2" model from Hugging Face. Since GPT-2 is an auto-regressive model (it generates text step-by-step), we fine-tune it similarly to how we did with bloom-2b.

# Step 5: Model Evaluation
Once the model is fine-tuned, you can evaluate its performance using metrics such as :
1. Calculate Perplexity - Perplexity can be derived from the loss.
2. ROUGE Score - We can use the rouge-score library to compute the ROUGE score for generated impressions.

# Step 6: Text Analysis
Install Dependencies
We will need the following Python libraries for this task:
1. nltk for stop word removal, stemming, and lemmatization.
2. spacy for more advanced lemmatization.
3. sentence-transformers for generating text embeddings.
4. scikit-learn for calculating cosine similarity.
5. plotly for interactive visualizations.

# Step 7: Perform Text Preprocessing (Stop Word Removal, Stemming, and Lemmatization)
first preprocess the text data by removing stop words, applying stemming, and lemmatizing the text.

# Step 8: Convert Preprocessed Text into Embeddings
Convert the preprocessed text into embeddings using the Sentence Transformers library. These embeddings will be used to calculate the similarity between word pairs.

# Step 9: Calculate Cosine Similarity and Identify Top Word Pairs
We will calculate cosine similarity between the text embeddings to identify the top 100 most similar word pairs.

# Step 10: Create Interactive Visualization Using Plotly
Create an interactive visualization using Plotly to display the top 100 word pairs and their similarity scores.

# Step 11: Run the Visualization
After running the above code, you will see an interactive graph where:
1. Nodes represent the words.
2. Edges represent the similarity between word pairs.
3. Hovering over a node will show the word it represents.
4. Zooming and Panning allows you to explore the word pairs and connections.


