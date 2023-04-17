import pickle
import streamlit as st
import string
import nltk


from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer

ps=PorterStemmer()

load_model_mnb=pickle.load(open('spam_model.pkl','rb'))

load_tfidf=pickle.load(open('vectorizer.pkl','rb'))

def transform_sms(message):
    
    message=message.lower()
    message=nltk.word_tokenize(message)
    
    temp=[]
    for i in message:
        if i.isalnum():
            temp.append(i)

    message=temp[:] 
    temp.clear()
    
    for i in message:
        if i not in stopwords.words('english') and i not in string.punctuation:
            temp.append(i)
    
    message=temp[:]
    temp.clear()
    
    for i in message:
        temp.append(ps.stem(i))
    
    return " ".join(temp)

def main():
    st.set_page_config(layout="wide")


    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://images.unsplash.com/photo-1679189789181-06448aa7c382?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=1856&q=80");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
         
     )
    html_temp="""
    <div style="background-color:DarkBlue;padding:10xp">
    <h2 style="color:white;text-align:center;">SMS Spam Detection Model </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)

    input_sms=st.text_area("**Enter the message for testing**")
    input_sms=transform_sms(input_sms)
    input_sms=load_tfidf.transform([input_sms])
    pred=load_model_mnb.predict(input_sms)[0]
    if st.button("Predict"):
        if pred == 1:
            st.success("**Spam sms **💬 ")
        else:
            st.success("**Not Spam sms **💬")


if __name__ == '__main__':
    main()