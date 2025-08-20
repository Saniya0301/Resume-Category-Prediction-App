import streamlit as st
import pickle
import docx  # Extract text from Word file
import PyPDF2  # Extract text from PDF
import re
import os

# ==========================
# Load pre-trained model files safely
# ==========================
def load_pickle(file_name):
    if os.path.exists(file_name):
        return pickle.load(open(file_name, "rb"))
    else:
        st.error(f"❌ Missing required file: {file_name}. Please train your model and save it first.")
        st.stop()

svc_model = load_pickle("clf.pkl")       # Classifier
tfidf = load_pickle("tfidf.pkl")         # TF-IDF Vectorizer
le = load_pickle("encoder.pkl")          # Label Encoder


# ==========================
# Resume text cleaning
# ==========================
def cleanResume(txt):
    cleanText = re.sub('http\S+\s', ' ', txt)
    cleanText = re.sub('RT|cc', ' ', cleanText)
    cleanText = re.sub('#\S+\s', ' ', cleanText)
    cleanText = re.sub('@\S+', '  ', cleanText)
    cleanText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText)
    cleanText = re.sub('\s+', ' ', cleanText)
    return cleanText.strip()


# ==========================
# File extractors
# ==========================
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ''
    for page in pdf_reader.pages:
        text += page.extract_text() or ''
    return text


def extract_text_from_docx(file):
    doc = docx.Document(file)
    return "\n".join([p.text for p in doc.paragraphs])


def extract_text_from_txt(file):
    try:
        text = file.read().decode("utf-8")
    except UnicodeDecodeError:
        text = file.read().decode("latin-1")
    return text


def handle_file_upload(uploaded_file):
    file_extension = uploaded_file.name.split(".")[-1].lower()
    if file_extension == "pdf":
        return extract_text_from_pdf(uploaded_file)
    elif file_extension == "docx":
        return extract_text_from_docx(uploaded_file)
    elif file_extension == "txt":
        return extract_text_from_txt(uploaded_file)
    else:
        raise ValueError("Unsupported file type. Please upload a PDF, DOCX, or TXT file.")


# ==========================
# Prediction function
# ==========================
def pred(input_resume):
    cleaned_text = cleanResume(input_resume)
    vectorized_text = tfidf.transform([cleaned_text]).toarray()
    predicted_category = svc_model.predict(vectorized_text)
    predicted_category_name = le.inverse_transform(predicted_category)
    return predicted_category_name[0]


# ==========================
# Streamlit App
# ==========================
def main():
    st.set_page_config(page_title="Resume Category Prediction", page_icon="📄", layout="wide")

    st.title("📄 Resume Category Prediction App")
    st.markdown("Upload a resume in **PDF, TXT, or DOCX** format to get the predicted job category.")

    uploaded_file = st.file_uploader("Upload a Resume", type=["pdf", "docx", "txt"])

    if uploaded_file is not None:
        try:
            resume_text = handle_file_upload(uploaded_file)
            st.success("✅ Successfully extracted text from the uploaded resume.")

            if st.checkbox("Show Extracted Text"):
                st.text_area("Extracted Resume Text", resume_text, height=300)

            st.subheader("📌 Predicted Category")
            category = pred(resume_text)
            st.success(f"The predicted category is: **{category}**")

        except Exception as e:
            st.error(f"⚠️ Error processing the file: {str(e)}")


if __name__ == "__main__":
    main()
