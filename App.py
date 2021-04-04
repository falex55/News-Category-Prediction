import pickle

import streamlit as st


# load the model
clf = pickle.load(open("Nave_Bayes_Classifier.pkl", 'rb'))
tf = pickle.load(open("TFIDF.pkl", 'rb'))
le = pickle.load(open("LabelEncoder.pkl", 'rb'))


def predict(description):
    prediction = clf.predict(tf.transform(description))
    return le.inverse_transform(prediction)


def main():
    st.title("News Category Prediction")
    st.markdown("#### Enter News Below")
    description = st.text_area("Description")
    if st.button("Predict"):
        text =[description]
        news_group = predict(text)
        st.write("### News group: ")
        st.success(news_group)
    if st.button("clear"):
        st.empty()

if __name__ == '__main__':
    main()
