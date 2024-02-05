from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize
import nltk
import heapq
import streamlit as st

nltk.download('stopwords')
nltk.download('punkt')

def calculate_accuracy(original_text, generated_summary):
    original_words = set(original_text.lower().split())
    summary_words = set(generated_summary.lower().split())
    shared_words = original_words.intersection(summary_words)
    accuracy = len(shared_words) / len(original_words)
    return accuracy

def get_summary(text):
    sentences = sent_tokenize(text)

    stop_words = set(stopwords.words("english"))
    words = word_tokenize(text)
    filtered_words = [word.lower() for word in words if word.isalnum() and word.lower() not in stop_words]

    word_freq = FreqDist(filtered_words)

    sentence_scores = {}
    for sentence in sentences:
        for word in word_tokenize(sentence.lower()):
            if word in word_freq.keys():
                if sentence not in sentence_scores:
                    sentence_scores[sentence] = word_freq[word]
                else:
                    sentence_scores[sentence] += word_freq[word]

    summary_sentences = heapq.nlargest(5, sentence_scores, key=sentence_scores.get)
    summary = ' '.join(summary_sentences)
    
    accuracy = calculate_accuracy(text, summary)

    return summary, accuracy

def main():
    st.title("Text Summarizer")

    text = st.text_area("Enter your text here:")
    
    if st.button("Generate Summary"):
        if text:
            summary, accuracy = get_summary(text)
            st.subheader("Summary:")
            st.write(summary)
            st.subheader("Accuracy:")
            st.write(f"{accuracy * 100:.2f}%")

if __name__ == '__main__':
    main()
