import streamlit as st
import os
import tempfile
import glob
import sqlite3
import datetime
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from streamlit_mic_recorder import speech_to_text # Nueva librería para voz (RF-01)

# ==========================================
# 1. BASE DE DATOS SQLITE (RF-07 y RNF-01)
# ==========================================
# Conectamos a SQLite (check_same_thread=False es necesario para Streamlit)
conn = sqlite3.connect('agrobot_cache.db', check_same_thread=False)
c = conn.cursor()

# Tabla para Caché (Offline)
c.execute('''CREATE TABLE IF NOT EXISTS cache_offline
             (pregunta TEXT PRIMARY KEY, respuesta TEXT)''')
# Tabla para Historial/Auditoría
c.execute('''CREATE TABLE IF NOT EXISTS logs_auditoria
             (id INTEGER PRIMARY KEY AUTOINCREMENT, fecha TEXT, pregunta TEXT, respuesta TEXT)''')
conn.commit()

def buscar_en_cache(pregunta):
    """Busca si la pregunta ya se hizo antes para responder sin internet."""
    c.execute("SELECT respuesta FROM cache_offline WHERE pregunta=?", (pregunta.lower().strip(),))
    resultado = c.fetchone()
    return resultado[0] if resultado else None

def guardar_interaccion(pregunta, respuesta):
    """Guarda en la caché y en el registro de auditoría."""
    fecha_actual = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Guardar en logs
    c.execute("INSERT INTO logs_auditoria (fecha, pregunta, respuesta) VALUES (?, ?, ?)", 
              (fecha_actual, pregunta, respuesta))
    # Guardar en cache (Ignora si ya existe)
    c.execute("INSERT OR IGNORE INTO cache_offline (pregunta, respuesta) VALUES (?, ?)", 
              (pregunta.lower().strip(), respuesta))
    conn.commit()

# ==========================================
# 2. CONFIGURACIÓN E INTERFAZ
# ==========================================
st.set_page_config(page_title="Agrobot Plátano", page_icon="🍌", layout="centered")

@st.cache_resource
def cargar_modelo_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

embeddings = cargar_modelo_embeddings()

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
    st.session_state.documentos_cargados = False

# Cargamos los documentos en backend silenciosamente (si existe la carpeta)
if not st.session_state.documentos_cargados and os.path.exists("documentos"):
    archivos = glob.glob("documentos/*.pdf")
    if archivos:
        docs = []
        for ruta in archivos:
            loader = PyPDFLoader(ruta)
            docs.extend(loader.load())
        if docs:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(docs)
            st.session_state.vectorstore = FAISS.from_documents(splits, embeddings)
            st.session_state.documentos_cargados = True

# Intentamos obtener la API Key oculta
try:
    api_key = st.secrets["GROQ_API_KEY"]
except KeyError:
    api_key = None # Si no hay internet o falla, permitimos modo offline

# ==========================================
# 3. LÓGICA PRINCIPAL DEL CHATBOT
# ==========================================
st.title("🍌 Agrobot - Experto en Cultivo de Plátano")
st.markdown("Consultas mediante texto o voz. Sistema con memoria caché integrada.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- RF-01: Entrada Multimodal (Texto o Voz) ---
col1, col2 = st.columns([0.85, 0.15])
with col1:
    prompt_texto = st.chat_input("Escribe tu pregunta...")
with col2:
    # Micrófono integrado usando la API del navegador
    prompt_voz = speech_to_text(language='es-ES', use_container_width=True, just_once=True, key='STT')

# Determinamos si el usuario usó voz o texto
prompt = prompt_texto or prompt_voz

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # 1. VERIFICAR CACHÉ (RNF-01: Operación Offline)
        respuesta_cache = buscar_en_cache(prompt)
        
        if respuesta_cache:
            st.success("⚡ Respuesta recuperada desde caché local (Modo Offline)")
            st.markdown(respuesta_cache)
            st.session_state.messages.append({"role": "assistant", "content": respuesta_cache})
            
        else:
            # Si no está en caché, comprobamos internet/API
            if not api_key:
                st.error("❌ Sin conexión y la respuesta no está en caché. Revisa tu conexión.")
                st.stop()
                
            with st.spinner("Analizando..."):
                try:
                    contexto = ""
                    if st.session_state.documentos_cargados and st.session_state.vectorstore is not None:
                        retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})
                        docs_relevantes = retriever.invoke(prompt)
                        contexto = "\n\n".join(doc.page_content for doc in docs_relevantes)

                    llm = ChatGroq(
                        groq_api_key=api_key, 
                        model_name="llama-3.1-8b-instant", 
                        temperature=0.2 
                    )

                    # PROMPT ACTUALIZADO (RF-06 y RNF-06)
                    prompt_template = ChatPromptTemplate.from_messages([
                        ("system", """Eres un ingeniero agrónomo experto en el cultivo de plátano.
                        
                        Contexto técnico extraído de manuales:
                        {context}

                        REGLAS ESTRICTAS PARA RESPONDER:
                        1. RESPONDE DIRECTAMENTE A LA PREGUNTA. Tienes PROHIBIDO saludar.
                        2. Prioriza SIEMPRE la información del Contexto.
                        3. (RF-06) Si la pregunta es muy compleja o tu certeza es baja, SÚGIERE al final consultar físicamente a un técnico agrícola local.
                        4. (RNF-06) Confiabilidad: Si tu respuesta menciona el uso de pesticidas, fungicidas o cualquier agroquímico, DEBES incluir una advertencia de seguridad sobre el uso de equipo de protección personal.
                        """),
                        ("user", "{question}")
                    ])
                    
                    mensaje = prompt_template.format_messages(context=contexto, question=prompt)
                    respuesta_ia = llm.invoke(mensaje).content

                    st.markdown(respuesta_ia)
                    
                    # RF-07: Guardamos en SQLite
                    guardar_interaccion(prompt, respuesta_ia)

                    st.session_state.messages.append({"role": "assistant", "content": respuesta_ia})
                    
                except Exception as e:
                    st.error(f"Error procesando la solicitud: {str(e)}")