from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
import numpy as np
from huggingface_hub import login
import pandas as pd
from typing import List, Dict
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import re 
from langchain.schema import Document  
import matplotlib.pyplot as plt  
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForCausalLM
from rouge_score import rouge_scorer
from bert_score import score
from nltk.translate.bleu_score import sentence_bleu
import nltk
import json, os
nltk.download('wordnet')

class BM25Retriever:
  def __init__(self, texts: List[str]):
      self.texts = texts
      tokenized_texts = [text.split() for text in texts]
      self.bm25 = BM25Okapi(tokenized_texts)
  
  def get_relevant_documents(self, query: str, k: int = 3 ) -> List[str]:
      tokenized_query = query.split()
      doc_scores = self.bm25.get_scores(tokenized_query)
      top_k_indices = np.argsort(doc_scores)[-k:][::-1]
      return [self.texts[i] for i in top_k_indices]
  

class SemanticRetriever:  
    def __init__(self, texts):    
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2") 
        print("Hugging face Dense vector embedding details: ", self.embeddings)
        documents = [Document(page_content=text) for text in texts]   
        self.vectorstore = FAISS.from_documents(documents, self.embeddings)  

    def get_relevant_documents(self, query, k=3):    
        docs = self.vectorstore.similarity_search(query, k=k)  
        return [doc.page_content for doc in docs] 

class EnhancedHybridRetriever:
    def __init__(self, texts: List[str]):
        self.texts = texts
        self.bm25_retriever = BM25Retriever(texts)
        self.semantic_retriever = SemanticRetriever(texts)
        self.tfidf = TfidfVectorizer()
        self.tfidf.fit(texts)

    def expand_query(self, query: str) -> str:
        """Expands query using WordNet synonyms"""
        expanded_terms = set()
        for word in query.split():
            expanded_terms.add(word)
            for syn in wordnet.synsets(word):
                for lemma in syn.lemmas():
                    expanded_terms.add(lemma.name())
        return ' '.join(list(expanded_terms))

    def calculate_dynamic_weights(self, query: str) -> Tuple[float, float]:
        """Calculate weights based on query specificity"""
        query_vec = self.tfidf.transform([query]).toarray()[0]
        query_specificity = np.count_nonzero(query_vec) / len(query_vec)
        if query_specificity > 0.5:
            return 0.7, 0.3  # More weight to BM25 for specific queries
        return 0.3, 0.7  # More weight to semantic for general queries

    def enhanced_reciprocal_rank_fusion(
        self, 
        rankings: List[List[str]], 
        weights: List[float], 
        k: float = 60
    ) -> Dict[str, float]:
        scores = {}
        for rank_list, weight in zip(rankings, weights):
            for rank, doc in enumerate(rank_list):
                if doc not in scores:
                    scores[doc] = 0
                scores[doc] += weight * (1 / (rank + k))
        return scores

    def get_relevant_documents(self, query: str, k: int = 3) -> List[str]:
        expanded_query = self.expand_query(query)
        bm25_weight, semantic_weight = self.calculate_dynamic_weights(query)
        bm25_docs = self.bm25_retriever.get_relevant_documents(expanded_query, k)
        semantic_docs = self.semantic_retriever.get_relevant_documents(expanded_query, k)
        fusion_scores = self.enhanced_reciprocal_rank_fusion(
            [bm25_docs, semantic_docs],
            [bm25_weight, semantic_weight]
        )
        sorted_docs = sorted(fusion_scores.items(), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in sorted_docs[:k]]

def average_precision(retrieved_chunks, ground_truth_chunk):   
    relevant_retrieved = 0  
    precision_at_k = 0.0  
    for k, chunk in enumerate(retrieved_chunks, start=1):  
        if chunk == ground_truth_chunk:  
            relevant_retrieved += 1  
            precision_at_k += relevant_retrieved / k  
 
    if relevant_retrieved == 0:  
        return 0.0  
    return precision_at_k / relevant_retrieved 

def calculate_ndcg(retrieved_docs: List[str], relevant_docs: List[str], k: int = 3) -> float:
  relevance = []
  for doc in retrieved_docs[:k]: 
      similarity = 1 if doc in relevant_docs else 0
      relevance.append(similarity)          
  dcg = sum([(2**rel - 1) / np.log2(i + 2) for i, rel in enumerate(relevance)])
  ideal_relevance = sorted(relevance, reverse=True)
  idcg = sum([(2**rel - 1) / np.log2(i + 2) for i, rel in enumerate(ideal_relevance)])
  return dcg / idcg if idcg > 0 else 0

def count_words(text):  
    if pd.isna(text):
        return 0  
    return len(str(text).split())

def evaluate_retrievers(df: pd.DataFrame, texts: List[str]) -> Dict[str, Dict[str, float]]:
    metrics = {
        'bm25': {'ap': [], 'ndcg': [], 'recall': []},
        'semantic': {'ap': [], 'ndcg': [], 'recall': []},
        'hybrid': {'ap': [], 'ndcg': [], 'recall': []}
    }
    retrievers = {
        'bm25': BM25Retriever(texts),
        'semantic': SemanticRetriever(texts),
        'hybrid': EnhancedHybridRetriever(texts)
    }
    for idx, row in df.iterrows():
        query = row['question']
        ground_truth = row['context']

        for name, retriever in retrievers.items():
            retrieved_docs = retriever.get_relevant_documents(query)
            ap = average_precision(retrieved_docs, ground_truth)
            ndcg = calculate_ndcg(retrieved_docs, [ground_truth])
            recall = 1.0 if ground_truth in retrieved_docs else 0.0
            metrics[name]['ap'].append(ap)
            metrics[name]['ndcg'].append(ndcg)
            metrics[name]['recall'].append(recall)

    results = {}
    for name in metrics:
        results[name] = {
            metric: np.mean(values) 
            for metric, values in metrics[name].items()
        }
    return results


def main():
    login(token="**********************")
    df = pd.read_parquet("hf://datasets/PatronusAI/HaluBench/data/test-0000-of-00001.parquet")
    df.rename(columns={'passage': 'context'}, inplace=True)
    cleaned_rows = df.drop_duplicates(subset=['context', 'question'], keep='first')  
    cleaned_rows['word_count'] = cleaned_rows['context'].apply(count_words)
    texts = cleaned_rows['context'].unique().tolist()
    bm25_retriever = BM25Retriever(texts)
    semantic_retriever = SemanticRetriever(texts) 
    hybrid_retriever = EnhancedHybridRetriever(texts)
    bm25_chunks = []
    semantic_chunks = []
    hybrid_chunks = []

    for index, row in cleaned_rows.iterrows():
        query = row['question']
        bm25_rel_docs = bm25_retriever.get_relevant_documents(query)
        semantic_rel_docs = semantic_retriever.get_relevant_documents(query)
        hybrid_rel_docs = hybrid_retriever.get_relevant_documents(query)
        bm25_chunks.append(bm25_rel_docs)
        semantic_chunks.append(semantic_rel_docs)
        hybrid_chunks.append(hybrid_rel_docs)
    cleaned_rows['bm25_chunks'] = bm25_chunks
    cleaned_rows['semantic_chunks'] = semantic_chunks
    cleaned_rows['hybrid_chunks'] = hybrid_chunks
    evaluation_results = evaluate_retrievers(cleaned_rows, texts)
    print("\nRetriever Evaluation Results:")
    for retriever, metrics in evaluation_results.items():
        print(f"\n{retriever.upper()} Retriever:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
    cleaned_rows.to_csv('retriever_results_enhanced.csv', index=False)

    with open('evaluation_results.json', 'w') as f:
        json.dump(evaluation_results, f, indent=2)
    return cleaned_rows, evaluation_results

if __name__ == "__main__":
    result_df, eval_results = main()