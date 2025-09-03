import pandas as pd
import joblib
import os
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from dotenv import load_dotenv
from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate

load_dotenv()

def preprocess_data(df):
    """Clean and preprocess the data."""
    # Remove null values
    df = df.dropna(subset=['text', 'category'])
    
    # Basic text cleaning
    df['text'] = df['text'].str.strip()
    df['category'] = df['category'].str.strip()
    
    return df

def train_classification_model(df):
    """Train a classification model using TF-IDF and SVM."""
    try:
        # Preprocess data
        df = preprocess_data(df)
        
        # Split data
        X = df['text']
        y = df['category']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Vectorize text
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)
        
        # Train model
        model = SVC(kernel='linear', random_state=42, probability=True)
        model.fit(X_train_tfidf, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test_tfidf)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        return {
            'model': model,
            'vectorizer': vectorizer,
            'accuracy': accuracy,
            'classification_report': report,
            'test_data': (X_test, y_test, y_pred)
        }
    except Exception as e:
        raise e

def save_model(model_data, model_dir='models'):
    """Save the trained model and vectorizer."""
    try:
        os.makedirs(model_dir, exist_ok=True)
        
        model_path = os.path.join(model_dir, 'classification_model.pkl')
        vectorizer_path = os.path.join(model_dir, 'vectorizer.pkl')
        
        joblib.dump(model_data['model'], model_path)
        joblib.dump(model_data['vectorizer'], vectorizer_path)
        
        return True
    except Exception as e:
        raise e

@st.cache_resource
def load_model(model_dir='models'):
    """Load the saved model and vectorizer."""
    try:
        model_path = os.path.join(model_dir, 'classification_model.pkl')
        vectorizer_path = os.path.join(model_dir, 'vectorizer.pkl')
        
        if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
            return None
        
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        
        return {'model': model, 'vectorizer': vectorizer}
    except Exception as e:
        raise e

def classify_ticket(text, model_data=None):
    """Classify a ticket using the trained model."""
    try:
        if model_data is None:
            model_data = load_model()
            if model_data is None:
                # Fallback to LLM-based classification
                return classify_with_llm(text)
        
        # Vectorize the input text
        text_tfidf = model_data['vectorizer'].transform([text])
        
        # Predict category
        prediction = model_data['model'].predict(text_tfidf)[0]
        
        # Get prediction probabilities if available
        if hasattr(model_data['model'], 'predict_proba'):
            probabilities = model_data['model'].predict_proba(text_tfidf)[0]
            classes = model_data['model'].classes_
            confidence = max(probabilities)
        else:
            confidence = None
        
        return {
            'category': prediction,
            'confidence': confidence,
            'method': 'ml_model'
        }
    except Exception as e:
        # Fallback to LLM-based classification
        return classify_with_llm(text)

@st.cache_resource
def _init_llm():
    return OpenAI(temperature=0)

def classify_with_llm(text):
    """Fallback classification using LLM when ML model is not available."""
    try:
        llm = _init_llm()
        
        prompt = PromptTemplate(
            input_variables=["text"],
            template="""
            Classify the following support ticket into one of these categories:
            - HR Support: Issues related to human resources, payroll, benefits, leave, employee relations
            - IT Support: Technical issues, software problems, hardware issues, access problems
            - Transportation Support: Company vehicle issues, commute problems, travel arrangements
            
            Ticket: {text}
            
            Respond with only the category name (HR Support, IT Support, or Transportation Support).
            """
        )
        
        chain = prompt | llm
        response = chain.invoke({"text": text})
        
        # Clean the response
        category = response.strip()
        
        # Map to consistent format
        category_mapping = {
            'HR Support': 'HR Support',
            'IT Support': 'IT Support', 
            'Transportation Support': 'Transportation Support'
        }
        
        category = category_mapping.get(category, 'IT Support')  # Default to IT Support
        
        return {
            'category': category,
            'confidence': None,
            'method': 'llm'
        }
    except Exception as e:
        # Ultimate fallback
        return {
            'category': 'IT Support',
            'confidence': None,
            'method': 'fallback'
        }
