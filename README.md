# README

## Project Title
### Screening Task: Semantic NLP Filtering for Deep Learning Papers in Virology/Epidemiology

## Overview
This project implements a semantic natural language processing (NLP) solution to filter and classify academic papers that discuss deep learning applications within virology and epidemiology. Using advanced transformer-based NLP models, the project semantically screens research papers to accurately identify and classify those that meet specific criteria in these domains.

### Purpose
The goal is to create a refined dataset that excludes irrelevant papers and accurately captures those most pertinent to deep learning in virology and epidemiology, surpassing traditional keyword filtering techniques by leveraging semantic understanding.

## Methodology

1. **Loading and Cleaning the Dataset**
    - The initial dataset includes around 11,450 records containing research paper metadata.
    - Records missing either title or abstract are filtered out, resulting in a clean dataset that ensures sufficient content for semantic analysis.

2. **NLP Technique for Filtering**
    #### **Semantic Embeddings**: 
    
    - This solution applies semantic embeddings to filter research papers based on relevance to virology/epidemiology and deep learning topics. After evaluating multiple embedding models, the **HuggingFace/SmolLM2-1.7B** model was selected due to its superior accuracy and ability to capture nuanced semantic meaning.
    
    - This model, trained with **1.7 billion parameters**, consistently outperformed lighter-weight Sentence Transformer models such as **all-MiniLM-L6-v2** and **paraphrase-MiniLM-L6-v2**, each trained with approximately **66 million parameters**. While these lighter models offer speed and efficiency, they reached a maximum similarity threshold of only **0.67**, limiting their effectiveness for high-accuracy filtering. By contrast, the HuggingFace model achieved a maximum threshold of **0.95**, significantly enhancing the accuracy of this filtering approach.
    
    - Though larger and requiring slightly longer for embedding generation on personal computers, the HuggingFace model was deemed optimal for this project due to its precision. In environments with robust GPU resources, such as Google Colab, embedding generation time is significantly reduced, making this approach feasible with larger datasets. This method is particularly effective for topics with varied terminology yet consistent context, supporting **contextual filtering** over simple keyword matching.
3. **Semantic Filtering with Cosine Similarity**

    **Relevance Embeddings:**
    - To ensure accurate filtering, two main relevance embeddings are generated:

        - **Virology/Epidemiology Relevance Embedding**: Captures terms and contexts related to infectious disease modeling, virus detection, epidemiology, and virology.
        - **Deep Learning Relevance Embedding**: Focuses on neural network-based techniques, such as convolutional neural networks (CNNs), recurrent neural networks (RNNs), and other machine learning models applied in research.
        
    - Using these embeddings, each paper’s similarity score is computed via cosine similarity:
        
        - **Initial Virology/Epidemiology Filtering**: Papers are first filtered by their relevance to virology and epidemiology based on a similarity threshold of 0.90.
        - **Deep Learning Relevance Filtering**: Further filtering is applied to assess each paper’s relevance to deep learning using a similarity threshold of 0.80.
        
    These threshold values are adjustable based on the desired specificity and the degree of relevance required. Adjusting these thresholds can help refine the selection, making it more inclusive or exclusive depending on specific research needs.


4. **Method Classification and Extraction**
    - **Classification**: Relevant papers are classified into “Text Mining” or “Computer Vision” categories, based on the method type used, using method-specific embeddings.
    - **Extraction of Specific Methods**: To enhance the relevance of results, a keyword-based approach is used to extract specific methods and techniques such as **CNNs**, **transformers**, and other model types. This method leverages a predefined list of keywords directly tied to the query parameters, ensuring that the extraction process aligns with the methods initially used to query the dataset. This alignment allows for efficient identification and categorization of research papers that mention these specific deep learning techniques within virology and epidemiology contexts.



## Why This Approach is More Effective Than Keyword-Based Filtering
Keyword filtering often captures a wide range of irrelevant records, especially for complex research topics where terminology may vary widely. The semantic approach used here:
   - **Understands Context**: By capturing the meaning behind the words, this method identifies relevant papers even if specific keywords are not present.
   - **Improves Accuracy**: Reduces false positives (irrelevant papers) and false negatives (relevant papers lacking specific keywords), ensuring a high-quality, focused dataset.

## Resulting Dataset Statistics


The final dataset generated after applying semantic filtering, method classification, and specific method extraction contains **154 research papers**. The steps and filters applied have significantly refined the dataset to focus on high-relevance research, capturing deep learning applications within virology and epidemiology. Here are the key characteristics of this final dataset:

### To retrieve a broader range of relevant papers, lowering the similarity thresholds can capture additional entries with slightly lower similarity scores, enabling a more inclusive selection based on specific research needs. This involves adjusting the virology/epidemiology similarity threshold (currently 0.90) and the deep learning similarity threshold (currently 0.80) as needed..

1. **Total Records**: 154 entries, each with a non-empty abstract and content relevant to virology/epidemiology and deep learning techniques.
2. **Key Columns**:
   - **Virology_Similarity**: Cosine similarity score indicating each paper's relevance to virology/epidemiology topics.
   - **Deep_Learning_Similarity**: Cosine similarity score indicating relevance to deep learning methods within the virology/epidemiology context.
   - **Method_Type**: Classifies each paper as either "Text Mining," "Computer Vision," or "Other" based on semantic similarity to specific NLP and computer vision research contexts.
   - **Methods_Used**: Extracted specific techniques, such as "neural network" and "computer vision," to provide a focused view of methods applied in each study.

3. **Additional Information**:
   - Out of 154 entries, **15 records** have “Not Specified” in the `Methods_Used` column, indicating the absence of the predefined keywords for specific methods.
   - The dataset retains key bibliographic information, such as `PMID`, `Title`, `Authors`, `DOI`, and `Abstract`, making it comprehensive for further analysis and reference.
   - With 154 records and 18 columns, including two floating-point columns for similarity scores, this dataset is structured for efficient processing.



        | #   | Column                    | Non-Null Count | Dtype   |
        |-----|----------------------------|----------------|---------|
        | 0   | PMID                       | 154 non-null   | int64   |
        | 1   | Title                      | 154 non-null   | object  |
        | 2   | Authors                    | 154 non-null   | object  |
        | 3   | Citation                   | 154 non-null   | object  |
        | 4   | First Author               | 154 non-null   | object  |
        | 5   | Journal/Book               | 154 non-null   | object  |
        | 6   | Publication Year           | 154 non-null   | int64   |
        | 7   | Create Date                | 154 non-null   | object  |
        | 8   | PMCID                      | 113 non-null   | object  |
        | 9   | NIHMS ID                   | 6 non-null     | object  |
        | 10  | DOI                        | 149 non-null   | object  |
        | 11  | Abstract                   | 154 non-null   | object  |
        | 12  | Title_Abstract             | 154 non-null   | object  |
        | 13  | Paper_Embedding            | 154 non-null   | object  |
        | 14  | Virology_Similarity        | 154 non-null   | float32 |
        | 15  | Deep_Learning_Similarity   | 154 non-null   | float32 |
        | 16  | Method_Type                | 154 non-null   | object  |
        | 17  | Methods_Used               | 154 non-null   | object  |



The final dataset, stored in `deep_learning_virology_methods_extracted.csv`, provides a well-curated collection of papers that focus specifically on relevant deep learning applications in virology and epidemiology, optimized for downstream research and analysis.


## Final Note
This semantic NLP-based screening process significantly enhances the relevance and quality of the dataset compared to simple keyword-based filtering, providing a robust solution to researchers and analysts focusing on deep learning applications in virology and epidemiology.

## Usage Guide

1. **Install Required Libraries**
   ```bash
   pip install transformers pandas scikit-learn torch tqdm
   ```
   - These libraries include tools for handling data, generating embeddings, and calculating similarity scores.

2. **Run the Jupyter Notebook**
    - The Jupyter notebook provided contains code cells for each stage:
        - Loading and filtering the data.
        - Generating embeddings for semantic filtering.
        - Calculating cosine similarity scores and applying thresholds.
        - Classifying by method type and extracting specific techniques.

3. **Adjust Thresholds**
    - Users may adjust the similarity thresholds (currently set at 0.90 for virology and 0.80 for deep learning) to broaden or narrow the selection as needed.
    
4. **Output Files**
    - Each stage generates a CSV file summarizing filtered, classified, and extracted data. These files are suitable for additional analysis or presentation of findings.

