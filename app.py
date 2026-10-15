import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
from langchain_core.messages import HumanMessage, AIMessage

st.title("🎬 Asistente Salas de Edición UDP")

# Configurar la API Key desde los secretos de Streamlit
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

# Configurar el sistema (se guarda en caché para no recargar el PDF en cada mensaje)
@st.cache_resource
def iniciar_sistema():
    loader = PyPDFLoader("manual-salas-edicion.pdf")
    paginas = loader.load_and_split()
    embeddings = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")
    base_de_datos = Chroma.from_documents(paginas, embeddings)
    buscador = base_de_datos.as_retriever()
    
    def formatear_documentos(docs):
        return "\n\n".join(doc.page_content for doc in docs)
        
    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.2)
    
    instrucciones = ChatPromptTemplate.from_messages([
        ("system", "Eres el asistente de las Salas de Edición de Comunicación y Letras UDP. "
                   "Utiliza SOLO la información proporcionada en el contexto. "
                   "Si te saludan, debes decir: '¡Hola! Soy el asistente de las Salas de Edición de Comunicación y Letras UDP. Te puedo ayudar con reservas, horarios, reglas y el servidor NEXIS. ¿En qué te ayudo?' "
                   "Si preguntan temas fuera del manual, excepciones, fallas de equipos o del servidor NEXIS, deriva la consulta a Javier Guerra al correo edicion.comyletras@mail.udp.cl. "
                   "Contexto:\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])
    
    chatbot = (
        {"context": itemgetter("input") | buscador | formatear_documentos, "input": itemgetter("input"), "chat_history": itemgetter("chat_history")}
        | instrucciones | llm | StrOutputParser()
    )
    return chatbot

chatbot = iniciar_sistema()

# Memoria visual del chat
if "mensajes" not in st.session_state:
    st.session_state.mensajes = []

for mensaje in st.session_state.mensajes:
    with st.chat_message(mensaje["rol"]):
        st.markdown(mensaje["contenido"])

# Caja de texto para el alumno
if pregunta := st.chat_input("Escribe tu duda aquí..."):
    st.chat_message("user").markdown(pregunta)
    
    # Preparar historial para Gemini
    historial_langchain = []
    for msg in st.session_state.mensajes:
        if msg["rol"] == "user":
            historial_langchain.append(HumanMessage(content=msg["contenido"]))
        else:
            historial_langchain.append(AIMessage(content=msg["contenido"]))
            
    # Obtener respuesta
    respuesta = chatbot.invoke({"input": pregunta, "chat_history": historial_langchain})
    
    st.chat_message("assistant").markdown(respuesta)
    
    # Guardar en memoria
    st.session_state.mensajes.append({"rol": "user", "contenido": pregunta})
    st.session_state.mensajes.append({"rol": "assistant", "contenido": respuesta})
