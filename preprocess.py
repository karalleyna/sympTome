import re
import nltk
import spacy
import pandas as pd
from securespacy.tokenizer import custom_tokenizer
from securespacy.patterns import add_entity_ruler_pipeline
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
import emoji

# -----------------------------------
# Download NLTK resources
# -----------------------------------
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

# -----------------------------------
# Initialize stemmer, lemmatizer, and stopwords
# -----------------------------------
ps = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

# -----------------------------------
# 1) Hybrid Rule + Securespacy Pipeline
# -----------------------------------
# Load securespacy, which includes its own custom tokenizer and EntityRuler for security-related entities
nlp = spacy.load("en_core_web_sm")
nlp.tokenizer = custom_tokenizer(nlp)
add_entity_ruler_pipeline(nlp)


def remove_html_tag(text):
    return re.sub(r"<.*?>", "", text)


def replace_emojis(text):
    return emoji.replace_emoji(text, replace="")


def detect_and_preserve_entities(text):
    doc = nlp(text)
    preserved = {}
    new_text = text
    ents = sorted(doc.ents, key=lambda e: e.start_char)
    keywords, labels = set([]), set([])
    offset = 0
    for i, ent in enumerate(ents):
        label = ent.label_
        val = ent.text
        placeholder = f"__{label}_{i}__"
        start = ent.start_char + offset
        end = ent.end_char + offset
        new_text = new_text[:start] + placeholder + new_text[end:]
        offset += len(placeholder) - (end - start)
        preserved[placeholder] = (label, val)
        keywords.add(val.lower())
        labels.add(label.lower())

    return new_text, preserved, keywords, labels


def restore_entities(text, preserved):
    for placeholder, (label, val) in preserved.items():
        text = text.replace(placeholder.lower(), val.lower())
    return text


def remove_urls(text):
    return re.sub(r"https?://\S+|www\.\S+", "", text)


def remove_punctuation_preserving_entities(text):
    exclude = "!\"#$%&'’‘()*+,/:;<=>?@[\\]^`‘'{|}~"
    return text.translate(str.maketrans("", "", exclude))


def correct_spelling(text):
    return str(TextBlob(text).correct())


def remove_stopwords(text):
    return " ".join([w for w in text.split() if w.lower() not in stop_words])


def stem_words(text):
    return " ".join([ps.stem(w) for w in text.split()])


def lemmatize_words(text):
    words = nltk.word_tokenize(text)
    punctuations = "?:!.,;"
    words = [w for w in words if w not in punctuations]
    return " ".join([lemmatizer.lemmatize(w, pos="v") for w in words])


def preprocess_text(
    data,
    columns,
    spelling_correction=False,
    stopword_removal=True,
    do_stemming=False,
    do_lemmatizing=False,
    suffix="orig",
):
    if isinstance(columns, str):
        columns = [columns]
    for col in columns:
        if suffix is not None:
            data[f"{col}_{suffix}"] = data[col].astype(str)

        def _process(text):
            txt = remove_html_tag(text)
            txt = replace_emojis(txt)
            txt = txt.lower()
            txt_preserved, preserved, keywords, labels = detect_and_preserve_entities(
                txt
            )
            txt_preserved = remove_urls(txt_preserved)
            txt_preserved = txt.replace("-", "")
            txt_preserved = remove_punctuation_preserving_entities(txt_preserved)
            txt_preserved = txt_preserved.lower()

            if spelling_correction:
                txt_preserved = correct_spelling(txt_preserved)
            if stopword_removal:
                txt_preserved = remove_stopwords(txt_preserved)
            if do_stemming:
                txt_preserved = stem_words(txt_preserved)
            if do_lemmatizing:
                txt_preserved = lemmatize_words(txt_preserved)

            txt_final = restore_entities(txt_preserved, preserved)
            return txt_final, keywords, labels

        results = data[col].fillna("").apply(lambda text: _process(text))
        # unzip into new columns
        data[col], data[f"keywords"], data[f"labels"] = zip(*results)
    return data


def split_into_sentences(data, column):
    """
    Splits text in the specified column into sentences and creates a new row for each sentence.
    Returns a new DataFrame with sentences in place of the original text.
    """
    # Ensure column is string
    data = data.copy()
    data[column] = data[column].astype(str)
    # Use NLTK's sentence tokenizer
    data["sentence_list"] = data[column].apply(lambda x: nltk.sent_tokenize(x))
    # Explode into separate rows
    data = data.explode("sentence_list").reset_index(drop=True)
    # Rename the sentence_list column back to original column name
    data = data.drop(columns=[column])
    data = data.rename(columns={"sentence_list": column})
    return data


# # -----------------------------------
# # Example Usage
# # -----------------------------------
# if __name__ == "__main__":
#     # Example DataFrame
#     df = pd.DataFrame({
#         'text': [
#             "Hello world! This is a test. Let's see how it works.",
#             "Another example: split me into sentences. And preprocess each one."
#         ]
#     })

#     # 1) Split into sentences (new rows)
#     df_sentences = split_into_sentences(df, 'text')

#     # 2) Preprocess text (keywords and labels will be added)
#     df_processed = preprocess_text(df_sentences, 'text', correct_spelling=False, remove_stopwords=True, do_stemming=False, do_lemmatizing=True)

#     print(df_processed)
