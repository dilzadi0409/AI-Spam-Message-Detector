import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

st.set_page_config(page_title="Spam Message Detector", page_icon="🛡️")
st.title("🛡️ AI Spam Message Detector")
st.write("Enter a message and the machine-learning model will predict whether it looks like spam.")

# Small built-in educational dataset. For real deployment, train with a larger, balanced dataset.
texts = [
    "Congratulations you won a free prize click now", "Claim your cash reward today",
    "You have won a lottery send your bank details", "Urgent claim your free gift now",
    "Get a free phone by clicking this link", "Limited offer buy now and win cash",
    "You are selected for a free vacation", "Click here to receive your reward",
    "Earn money fast from home register now", "Your account won a bonus claim it",
    "Exclusive deal free entry claim today", "Send your details to get a prize",
    "Win big money in our online lottery", "Congratulations claim your gift card",
    "Free entry in a prize draw click link", "Act now to get a free voucher",
    "You have been chosen for a cash award", "Get rich quickly with this offer",
    "Claim your reward before it expires", "Free shopping voucher waiting for you",
    "Hey are we still meeting for lunch", "Please call me when you are free",
    "Your appointment is confirmed for tomorrow", "Can you send me the project file",
    "I will reach home in ten minutes", "Thanks for helping me yesterday",
    "The class starts at nine in the morning", "Please review the attached document",
    "Happy birthday have a wonderful day", "I am waiting outside the office",
    "Let's discuss the website design today", "Your order has been delivered",
    "Can we reschedule our meeting", "I will share the notes after class",
    "Please remember to bring your laptop", "Good morning how are you",
    "The team meeting is at three pm", "I have completed the assignment",
    "See you at the training session", "Thank you for your quick response"
]
labels = [1]*20 + [0]*20

model = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), lowercase=True)),
    ("classifier", LogisticRegression(max_iter=1000, random_state=42))
])
model.fit(texts, labels)

message = st.text_area("Message", placeholder="Example: Congratulations! Claim your free prize now.")
if st.button("Check message", type="primary"):
    if not message.strip():
        st.warning("Please enter a message first.")
    else:
        prediction = model.predict([message])[0]
        probabilities = model.predict_proba([message])[0]
        confidence = max(probabilities) * 100
        if prediction == 1:
            st.error(f"Prediction: Likely SPAM (demo confidence: {confidence:.1f}%)")
        else:
            st.success(f"Prediction: Likely NOT SPAM (demo confidence: {confidence:.1f}%)")
        st.caption("Educational demo only. The model uses a small sample dataset and can make mistakes. Do not use it to make security decisions.")

with st.expander("How does this project work?"):
    st.write("TF-IDF converts message text into numerical features. Logistic Regression learns patterns from labeled examples and predicts a class.")
