import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, silhouette_score

def read_data(file):
    if file.type == "text/csv":
        data = pd.read_csv(file)
    elif file.type == "text/plain":
        data = pd.read_csv(file, sep="\s+", header=None)
    else:
        raise ValueError("Invalid file type")
    return data

def supervised_learning(df):
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    return accuracy_score(y_test, y_pred)

def unsupervised_learning(df):
    X = df.iloc[:, :-1]
    kmeans = KMeans(n_clusters=len(df.iloc[:, -1].unique()), random_state=42)
    kmeans.fit(X)

    return silhouette_score(X, kmeans.labels_)

st.title("SW Machine Learning Application")
st.write("Upload a CSV or TXT file containing data in the NxK format, where N are the samples, K-1 are the features, and the last column is the label.")

uploaded_file = st.file_uploader("Choose a file", type=["csv", "txt"])

if uploaded_file is not None:
    try:
        data = read_data(uploaded_file)
        st.write("Data Preview:")
        st.write(data.head())

        supervised_result = supervised_learning(data)
        unsupervised_result = unsupervised_learning(data)

        st.write("Results:")
        results_df = pd.DataFrame({"Method": ["Supervised (Accuracy)", "Unsupervised (Silhouette Score)"], "Score": [supervised_result, unsupervised_result]})
        st.write(results_df)

    except ValueError as e:
        st.error(e)

else:
    st.warning("Please upload a file.")
