from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
import nltk
import heapq
import streamlit as st
from rouge_score import rouge_scorer
from transformers import BartTokenizer, BartForConditionalGeneration
from newspaper import Article

nltk.download('stopwords')
nltk.download('punkt')

def calculate_rouge(original_text, generated_summary):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(original_text, generated_summary)
    return scores

def calculate_overall_accuracy(rouge_scores):
    overall_accuracy = (rouge_scores['rouge1'].fmeasure + rouge_scores['rouge2'].fmeasure + rouge_scores['rougeL'].fmeasure) / 3
    return overall_accuracy * 100

def get_summary(text):
    # Use BART model for summarization
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')

    inputs = tokenizer("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(**inputs, max_length=150, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return summary

def scrape_and_summarize(url):
    article = Article(url)
    article.download()
    article.parse()
    text = article.text
    summary = get_summary(text)
    return text, summary

def main():
    st.title("News Article Summarizer")

    url = st.text_input("Enter the news article URL:")
    
    if st.button("Generate Summary"):
        if url:
            try:
                article_text, summary = scrape_and_summarize(url)
                rouge_scores = calculate_rouge(article_text, summary)
                overall_accuracy = calculate_overall_accuracy(rouge_scores)
                
                st.subheader("Summary:")
                st.write(summary)
                st.subheader("Rouge Scores:")
                st.write(f"Rouge-1: {rouge_scores['rouge1'].fmeasure * 100:.2f}%")
                st.write(f"Rouge-2: {rouge_scores['rouge2'].fmeasure * 100:.2f}%")
                st.write(f"Rouge-L: {rouge_scores['rougeL'].fmeasure * 100:.2f}%")
                st.subheader("Overall Accuracy:")
                st.write(f"{overall_accuracy:.2f}%")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

if __name__ == '__main__':
    main()
