import streamlit as st
import cohere
import numpy as np
import pandas as pd
import pdfplumber
import translators as ts
from annoy import AnnoyIndex
from concurrent.futures import ThreadPoolExecutor

# Configure page title
st.set_page_config(page_title="Document Cofinder")

# Access the API key value
api_key = st.secrets['API_KEY']

# initialize Cohere client
co = cohere.Client(api_key)

CHUNK_WIDTH = 1500
OVERLAP = 500
INITIAL_RETRIEVAL_COUNT = 10
RERANK_RETRIEVAL_COUNT = 3

# UTIL FUNCTIONS
def chunk_text(df, width=CHUNK_WIDTH, overlap=OVERLAP):
    # create an empty dataframe to store the chunked text
    new_df = pd.DataFrame(
        columns=['text_chunk', 'start_index', 'title', 'page'])

    # iterate over each row in the original dataframe
    for index, row in df.iterrows():
        # split the text into chunks of size 'width', with overlap of 'overlap'
        chunks = []
        for i in range(0, len(row['text']), width - overlap):
            chunk = row['text'][i:i+width]
            chunks.append(chunk)

        # iterate over each chunk and add it to the new dataframe
        chunk_rows = []
        for i, chunk in enumerate(chunks):
            # calculate the start index based on the chunk index and overlap
            start_index = i * (width - overlap)

            # create a new row with the chunked text and the original row's file title and page number
            new_row = {'text_chunk': chunk, 'start_index': start_index,
                       'title': row['title'], 'page': row['page']}
            chunk_rows.append(new_row)
        chunk_df = pd.DataFrame(chunk_rows)
        new_df = pd.concat([new_df, chunk_df], ignore_index=True)

    return new_df


def chunk_and_index(uploaded_files):
    df = pd.DataFrame(columns=['text', 'title', 'page'])

    for uploaded_file in uploaded_files:
        # print file name
        title = uploaded_file.name
        with st.spinner(f"Reading {title}..."):
            # extract text and pass to df
            with pdfplumber.open(uploaded_file) as pdf:
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    df = pd.concat([df, pd.DataFrame({
                        "text": [text],
                        "title": [title],
                        "page": [i+1]
                    })])

    # chunk text in df
    with st.spinner("Chunking text..."):
        df = chunk_text(df)

    with st.spinner("Building index..."):
        # Get the embeddings
        embeds = co.embed(texts=list(df['text_chunk']),
                          model="embed-multilingual-v2.0",
                          truncate="RIGHT").embeddings
        embeds = np.array(embeds)

        # Create the search index, pass the size of embedding
        search_index = AnnoyIndex(embeds.shape[1], 'dot')
        # Add all the vectors to the search index
        for i in range(len(embeds)):
            search_index.add_item(i, embeds[i])

        search_index.build(10)  # 10 trees

    return search_index, df


def search(query, n_results, df, search_index, co):
    # Get the query's embedding
    query_embed = co.embed(texts=[query],
                           model="embed-multilingual-v2.0",
                           truncate="LEFT").embeddings

    # Get the nearest neighbors and similarity score for the query and the embeddings,
    # append it to the dataframe
    nearest_neighbors = search_index.get_nns_by_vector(
        query_embed[0],
        n_results,
        include_distances=True)

    # filter the dataframe to include the nearest neighbors using the index
    result_df = df[df.index.isin(nearest_neighbors[0])]
    index_similarity_df = pd.DataFrame(
        {'similarity': nearest_neighbors[1]}, index=nearest_neighbors[0])
    # Match similarities based on indexes
    result_df = result_df.join(index_similarity_df)
    result_df = result_df.sort_values(by='similarity', ascending=False)
    return result_df


def gen_answer(q, para):
    response = co.generate(
        model='command-xlarge-20221108',
        prompt=f'''Paragraph:{para}\n\n
                Answer the question using this paragraph.\n\n
                Question: {q}\nAnswer:''',
        max_tokens=100,
        temperature=0)
    return response.generations[0].text


def gen_better_answer(ques, ans):
    response = co.generate(
        model='command-xlarge-20221108',
        prompt=f'''Input statements:{ans}\n\n
                Question: {ques}\n\n
                Using the input statements, generate one answer to the question. ''',
        max_tokens=100,
        temperature=0.3)
    return response.generations[0].text


def display(query, results):
    # 1. Run co.generate functions to generate answers

    # for each row in the dataframe, generate an answer concurrently
    with ThreadPoolExecutor(max_workers=1) as executor:
        results['answer'] = list(executor.map(gen_answer,
                                              [query]*len(results),
                                              results['chunk_translation']))
    answers = results['answer'].tolist()
    # run the function to generate a better answer
    answ = gen_better_answer(query, answers)

    # 2. Code to display the resuls in a user-friendly format
    # add a spacer
    st.write('')
    st.write('')
    st.subheader(query)
    st.write(answ)
    # add a spacer
    st.write('')
    st.write('')
    st.subheader("Relevant documents")
    # display the results
    for i, row in results.iterrows():
        # display the 'Category' outlined
        st.markdown(
            f'Answer from **page {row["page"]}** of **{row["title"]}**')
        st.write(row['answer'])
        # collapse the text
        with st.expander('Read more'):
            st.write('**Original:**')
            st.write(row['text_chunk'])
            st.write('**English:**')
            st.write(row['chunk_translation'])
        st.write('')


def translate_chunk(chunk):
    translation = ""
    try:
        translation = ts.translate_text(chunk, to_language='en', translator='google')
    except Exception as e:
        print("#TranslationFailed:")
        print(e)
    return translation


def translation_failed(df):
    translations = set(df['chunk_translation'])
    # print("############Translations:############\n", translations)
    if (len(translations) == 1) and (list(translations)[0] == ""):
        return True
    return False


# Title
st.title("Document Cofinder")
# Subtitle
st.subheader("A language-agnostic semantic search tool built for PDF's")
# Warning about rate-limiting
st.write("---")
with st.expander("⚠️ **Please note...**"):
    st.info("This is a proof of concept that uses Cohere's rate-limited trial API key.  \n"\
            "It allows for a max of one search per minute. _It has not been set up to accomodate multiple users yet._")
st.write("---")

# File uploader
uploaded_files = st.file_uploader(
    "Add your reference PDF files:", accept_multiple_files=True)

st.write("")
st.write("")

query = st.text_input('Interrogate your documents')

if st.button('Search') or query:
    search_index, df = chunk_and_index(uploaded_files)

    with st.spinner("Running Search..."):
        results = search(query, INITIAL_RETRIEVAL_COUNT, df, search_index, co).reindex(range(INITIAL_RETRIEVAL_COUNT))

    with st.spinner("Translating..."):
        results['chunk_translation'] = results.apply(lambda x: translate_chunk(x['text_chunk']), axis=1)

    if not translation_failed(results):
        with st.spinner("Reranking..."):
            rerank_hits = co.rerank(query=query, documents=results['chunk_translation'].to_list(), 
                                    top_n=RERANK_RETRIEVAL_COUNT, model='rerank-multilingual-v2.0')
            top_index_list = [hit.index for hit in rerank_hits]
            results = results.iloc[top_index_list]
    else:
        results = results.head(3)

    with st.spinner("Generating Output..."):
        display(query, results)
