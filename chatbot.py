# chatbot.py
from transformers import pipeline
import spacy

sentiment_model = pipeline("sentiment-analysis")
ner = spacy.load("en_core_web_sm")

def chatbot_response(user_input):
    sentiment = sentiment_model(user_input)[0]
    doc = ner(user_input)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    response = f"You sound {sentiment['label'].lower()} about {entities if entities else 'this topic'}!"
    return response

if __name__ == '__main__':
    while True:
        msg = input("You: ")
        print("Bot:", chatbot_response(msg))
