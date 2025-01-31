from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
import numpy as np
import torch
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

import pandas as pd


dataframe = pd.read_csv('retriever_results_v3.csv')
dataframe["source_ds"].value_counts()

# filter the dataframe by source_ds column name
df_factual = dataframe[dataframe['source_ds'] == 'DROP'].reset_index(drop=True)

# filter the dataframe by label column where 25 rows are 'FAIL' and 25 rows are 'PASS'
# df_factual = pd.concat([
#     df_factual[df_factual['label'] == 'FAIL'].sample(25),
#     df_factual[df_factual['label'] == 'PASS'].sample(25)
# ]).reset_index(drop=True)

df_factual = df_factual[df_factual['label'] == 'PASS'].sample(25).reset_index(drop=True)


class LlamaQAPipeline:
    def __init__(self):
        #self.model_name = "meta-llama/Meta-Llama-3.1-70B-Instruct"
        self.model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )

    def generate_prompt(self, context: str, question: str) -> str:
        prompt = f"""[INST] You are a precise and helpful assistant. When responding:
            - Provide a single, clear answer without repetition
            - Don't restate the question or context. DO NOT REPEAT THE PROMPT IN THE RESPONSE, dont write any code.
            - Search if you can find the relevantanswer in the provided context.
            - If uncertain, say "The context doesn't provide sufficient information to answer the question"
            - Avoid unnecessary formatting tokens in the response
            - Be direct and concise while maintaining a friendly tone, avoid long explanations
            - Only provide the answer to the question
            

        Context: {context}

        Question: {question}

        Answer: [/INST]"""
        return prompt

    def generate_answer(self, context: str, question: str) -> str:
        prompt = self.generate_prompt(context, question)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        prompt_length = inputs.input_ids.shape[1]

        outputs = self.model.generate(
            inputs.input_ids,
            # max_length=16384,
            max_new_tokens = 8132,
            temperature=0.8,
            top_p=0.9,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        generated_tokens = outputs[0][prompt_length:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        response = response.replace("[/INST]", "").strip()

        return response

qa_pipeline = LlamaQAPipeline()
data = df_factual.to_dict('records')

bm25_predictions = []
semantic_predictions = []
hybrid_predictions = []
ground_truths = []
results = pd.DataFrame()
i = 1
for item in data:
    print(f"******************* executing row number {i} ***************************")
    bm25_prediction = qa_pipeline.generate_answer(item['bm25_chunks'], item['question'])
    bm25_predictions.append(bm25_prediction)

    semantic_prediction = qa_pipeline.generate_answer(item['semantic_chunks'], item['question'])
    semantic_predictions.append(semantic_prediction)

    hybrid_prediction = qa_pipeline.generate_answer(item['hybrid_chunks'], item['question'])
    hybrid_predictions.append(hybrid_prediction)

    ground_truths.append(item['answer'])
    results = pd.concat([results, pd.DataFrame({
        'question': [item['question']],
        'answer': [item['answer']],
        'context': [item['context']],
        'source': [item['source_ds']],
        'label': [item['label']],
        'bm25_context': [item['bm25_chunks']],
        'bm25_prediction': [bm25_prediction],
        'semantic_context': [item['semantic_chunks']],
        'semantic_prediction': [semantic_prediction],
        'hybrid_context': [item['hybrid_chunks']],
        'hybrid_prediction': [hybrid_prediction]
        
    }, index = [0])])
    i += 1

results.to_excel("generated_output_drop_25pass.xlsx", index=False)
x = 2