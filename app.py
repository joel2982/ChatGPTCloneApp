import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import os
import pickle
import openai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import faiss
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import mysql.connector
import MySQLdb
from langchain.chat_models import ChatOpenAI
from chatui import css,user_template,bot_template
from streamlit_extras.add_vertical_space import add_vertical_space

#OPENAI_API_KEY=st.secrets['OPENAI_API_KEY']
host = os.getenv("host")
user = os.getenv("user")
password = os.getenv("password")
database = os.getenv("database")

config = {
    'host':host,
    'user':user,
    'password':password,
    'database':database    
}
chatdb = MySQLdb.Connect(**config)

def new_session_state():
    st.session_state.conversation = None
    st.session_state.chat_history = None
    st.session_state.chain = None

def get_pdf_text(pdf_docs):
    title = []
    text_docs = []
    for pdf in pdf_docs:
        title.append(pdf.name[:-4])
        pdf_reader = PdfReader(pdf)
        text = ''
        for page in pdf_reader.pages:
            text += page.extract_text()
        text_docs.append(text)
    return title,text_docs

def get_text_chunks(texts):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = []
    for text in texts:
        chunks.append(text_splitter.split_text(text))
    return chunks

def get_vectorstore(titles,text_chunks):
    it_titles = iter(titles)
    title = next(it_titles)
    # if os.path.exists(f'embeddings\{title}.pkl'):
    #     with open(f'embeddings\{title}.pkl','rb') as f:
    #         vectorstore = pickle.load(f)
    # else:
    index = titles.index(title)
    embeddings = OpenAIEmbeddings()
    vectorstore = faiss.FAISS.from_texts(texts=text_chunks[index],embedding=embeddings)
        # with open(f'embeddings\{title}.pkl','wb') as f:
        #     pickle.dump(vectorstore,f)
    for title in it_titles:
        #dv = pd.Dataframe()
        if os.path.exists(f'\embeddings\{title}.pkl'):
            with open(f'\embeddings\{title}.pkl','rb') as f:
                dv = pickle.load(f)
        else:
            index = titles.index(title)
            embeddings = OpenAIEmbeddings()
            dv = faiss.FAISS.from_texts(texts=text_chunks[index],embedding=embeddings)
            with open(f'embeddings\{title}.pkl','wb') as f:
                pickle.dump(dv,f)
        vectorstore.merge_from(dv)
    return vectorstore

def get_conversation_chain():
    llm = ChatOpenAI(temperature=0.0, model_name = "gpt-3.5-turbo")
    retriever = st.session_state.embeddings.as_retriever()
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation = ConversationalRetrievalChain.from_llm(
        llm = llm,
        chain_type = "stuff",
        memory = memory,
        retriever = retriever
    )
    return conversation

def create_chat_history():
    st.write("creating")
    create_cursor = chatdb.cursor()
    sql_cmd = "INSERT INTO chat_history (title, vs_name, chat_history) VALUES (%s, %s, %s)"
    chat_history = str(st.session_state.chat_history)
    values = ("joel",st.session_state.title,chat_history)
    create_cursor.execute(sql_cmd,values)
    chatdb.commit()
    st.write(create_cursor.rowcount, "record(s) affected")


def update_chat_history():
    st.write("updating")
    update_cursor = chatdb.cursor()
    sql_cmd = "UPDATE chat_history SET chat_history = %s WHERE id = %s"
    chat_history = str(st.session_state.chat_history)
    values = (chat_history,1)
    update_cursor.execute(sql_cmd,values)
    chatdb.commit()
    st.write(update_cursor.rowcount, "record(s) affected")


def handle_user_input(user_question):
    response = st.session_state.conversation(user_question)
    st.session_state.chat_history = response['chat_history']
    st.write(str(st.session_state.chat_history))
    for i,chat in enumerate(st.session_state.chat_history):
        if i%2 == 0:
            st.write(user_template.replace("{{MSG}}",chat.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}",chat.content), unsafe_allow_html=True)
    if len(st.session_state.chat_history) == 2:
        create_chat_history()
    else:
        update_chat_history()

def retrieve_names(cmd):
    string = list(str(cmd.fetchall()))
    names=[]
    val = ''
    for i in string:
        if i in ',':
            if len(val)==0:
                continue
            names.append(val)
            val = ''
        elif i not in " ''[]()":
            val+=i
    names.sort()
    return names

def main():
    load_dotenv()
    st.set_page_config(page_title='Chat with PDFs',page_icon='books:')
    st.write(css, unsafe_allow_html=True)
    if 'conversation' not in st.session_state:
        st.session_state.conversation = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = None
    if 'embeddings' not in st.session_state:
        st.session_state.embeddings = None
    if 'title' not in st.session_state:
        st.session_state.title = None
    st.subheader('Chat with your PDFs :books:')
    st.write(password,host,user,database)
    select_cursor = chatdb.cursor()
    select_cursor.execute("SELECT title FROM chat_history")
    chat_titles = retrieve_names(select_cursor)
    with st.sidebar:
        if st.button('New Chat'):
            st.session_state.clear()
            new_session_state()
        add_vertical_space(2)
        chat = st.selectbox(label="Previous Chats",options=chat_titles, index=None, placeholder="Select the Chat")
        add_vertical_space(2)
        with st.form("my-form", clear_on_submit=True):
            pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True)
            submitted = st.form_submit_button("UPLOAD")
        if submitted and len(pdf_docs) != 0:
            with st.spinner('Processing'):
                #get the pdf text
                title,text_docs = get_pdf_text(pdf_docs)
                st.session_state.title = title 
                #divide the pdf text into text chunks
                text_chunks = get_text_chunks(text_docs)
                #create vectorstores
                st.session_state.embeddings = get_vectorstore(title,text_chunks)
                #initializing conversation
                st.session_state.conversation = get_conversation_chain()
        elif chat:
            sql_cmd = f"SELECT id FROM chat_history WHERE title = '{chat}'"
            select_cursor.execute(sql_cmd)
            st.session_state.chat_id = select_cursor.fetchone()[0]
        add_vertical_space(4)
        st.write('Made with ❤️ by [Joel John](https://www.linkedin.com/in/joeljohn29082002/)')
    if st.session_state.conversation:
        user_question = st.text_input('Ask your Questions!')
        st.write(bot_template.replace("{{MSG}}","How can I help you!"), unsafe_allow_html=True)
        if user_question:
            handle_user_input(user_question)

if __name__ == '__main__':
    main()
