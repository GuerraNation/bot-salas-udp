import streamlit as st
import os
import json
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
from langchain_core.messages import HumanMessage, AIMessage

st.set_page_config(page_title="Asistente Salas UDP", page_icon="🎬")
st.title("🎬 Asistente Salas de Edición UDP")

# Configurar la API Key desde los secretos de Streamlit
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

# ----------------- CONEXIÓN A GOOGLE SHEETS -----------------
def guardar_registro_nube(pregunta, respuesta):
    estado = "Derivada a Javier" if "edicion.comyletras@mail.udp.cl" in respuesta else "Respondida por PDF"
    fecha = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    try:
        # Leer el secreto de Streamlit de forma segura
        credenciales_dict = json.loads(st.secrets["CREDENCIALES_JSON"])
        
        # Configurar permisos para leer y escribir
        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive"
        ]
        creds = Credentials.from_service_account_info(credenciales_dict, scopes=scopes)
        cliente = gspread.authorize(creds)
        
        # ABRIR EL GOOGLE SHEET POR SU URL 
        url_sheet = "https://docs.google.com/spreadsheets/d/133S4N9ePAKy8ZH86xTJaNBAR-6VBvCU4N3ortTlalw0/edit?gid=0#gid=0"
        
        hoja = cliente.open_by_url(url_sheet).sheet1
        hoja.append_row([fecha, estado, pregunta, respuesta])
    except Exception as e:
        print(f"Error al guardar en la nube: {e}")
# ------------------------------------------------------------

# Configurar el sistema
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

# ----------------- PANTALLA DE INICIO (ESTILO GEMINI) -----------------
pregunta_sugerida = None
if len(st.session_state.mensajes) == 0:
    st.markdown("<h3 style='text-align: center; color: #888;'>¿Por dónde empezamos?</h3>", unsafe_allow_html=True)
    st.write("") # Espaciador
    col1, col2, col3 = st.columns(3)
    if col1.button("⏱️ Horarios"): 
        pregunta_sugerida = "¿Cuál es el horario de las salas de edición?"
    if col2.button("📅 Reservas"): 
        pregunta_sugerida = "¿Cómo puedo reservar una sala?"
    if col3.button("🖥️ Uso de NEXIS"): 
        pregunta_sugerida = "¿Cuáles son las reglas del servidor NEXIS?"
# ----------------------------------------------------------------------

# Caja de texto para el alumno
pregunta_usuario = st.chat_input("Escribe tu duda aquí...")

# Lógica principal
pregunta_final = pregunta_sugerida or pregunta_usuario

if pregunta_final:
    st.chat_message("user").markdown(pregunta_final)
    
    # Preparar historial
    historial_langchain = []
    for msg in st.session_state.mensajes:
        if msg["rol"] == "user":
            historial_langchain.append(HumanMessage(content=msg["contenido"]))
        else:
            historial_langchain.append(AIMessage(content=msg["contenido"]))
            
    # Obtener respuesta
    respuesta = chatbot.invoke({"input": pregunta_final, "chat_history": historial_langchain})
    
    st.chat_message("assistant").markdown(respuesta)
    
    # Guardar en memoria y en GOOGLE SHEETS
    st.session_state.mensajes.append({"rol": "user", "contenido": pregunta_final})
    st.session_state.mensajes.append({"rol": "assistant", "contenido": respuesta})
    guardar_registro_nube(pregunta_final, respuesta)
    
    if pregunta_sugerida:
        st.rerun()
