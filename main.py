import streamlit as st
import cohere
import numpy as np
import pandas as pd
import pdfplumber
from annoy import AnnoyIndex
from concurrent.futures import ThreadPoolExecutor

# Access the API key value
api_key = st.secrets['API_KEY']


# Chunking function
def chunk_text(df, width=1500, overlap=500):
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


co = cohere.Client(api_key)

# add title
st.title("Document Cofinder")
# add a subtitle
st.subheader("A semantic search tool built for PDF's")

# Add file uploader
uploaded_files = st.file_uploader(
    "Add your reference PDF files:", accept_multiple_files=True)

def chunk_and_index(uploaded_files=uploaded_files):
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
                        model="large",
                        truncate="RIGHT").embeddings
        embeds = np.array(embeds)

        # Create the search index, pass the size of embedding
        search_index = AnnoyIndex(embeds.shape[1], 'angular')
        # Add all the vectors to the search index
        for i in range(len(embeds)):
            search_index.add_item(i, embeds[i])

        search_index.build(10) # 10 trees

    return search_index, df

st.write("")
st.write("")


# # Load the search index
# search_index = AnnoyIndex(f=4096, metric='angular')
# search_index.load('search_index.ann')

# # load the csv file called cohere_final.csv
# df = pd.read_csv('cohere_text_final.csv')


def search(query, n_results, df, search_index, co):
    # Get the query's embedding
    query_embed = co.embed(texts=[query],
                           model="large",
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
    result_df = result_df.join(index_similarity_df)  # Match similarities based on indexes
    result_df = result_df.sort_values(by='similarity', ascending=False)
    return result_df


# define a function to generate an answer
def gen_answer(q, para):
    response = co.generate(
        model='command-xlarge-20221108',
        prompt=f'''Paragraph:{para}\n\n
                Answer the question using this paragraph.\n\n
                Question: {q}\nAnswer:''',
        max_tokens=100,
        temperature=0.4)
    return response.generations[0].text


def gen_better_answer(ques, ans):
    response = co.generate(
        model='command-xlarge-20221108',
        prompt=f'''Input statements:{ans}\n\n
                Question: {ques}\n\n
                Using the input statements, generate one answer to the question. ''',
        max_tokens=100,
        temperature=0.4)
    return response.generations[0].text


def display(query, results):
    # 1. Run co.generate functions to generate answers

    # for each row in the dataframe, generate an answer concurrently
    with ThreadPoolExecutor(max_workers=1) as executor:
        results['answer'] = list(executor.map(gen_answer,
                                              [query]*len(results),
                                              results['text_chunk']))
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
        st.markdown(f'Answer from **page {row["page"]}** of **{row["title"]}**')
        st.write(row['answer'])
        # collapse the text
        with st.expander('Read more'):
            st.write(row['text_chunk'])
        st.write('')

# add the if statements to run the search function when the user clicks the buttons


query = st.text_input('Interrogate your documents')

if st.button('Search'):
    search_index, df = chunk_and_index()
    with st.spinner("Running Search..."):
        results = search(query, 3, df, search_index, co)
    with st.spinner("Generating Output..."):
        display(query, results)
