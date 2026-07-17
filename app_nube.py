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
from streamlit_mic_recorder import speech_to_text # Librería para voz (RF-01)

# ==========================================
# 1. BASE DE DATOS SQLITE (RF-07 y RNF-01)
# ==========================================
conn = sqlite3.connect('agrobot_cache.db', check_same_thread=False)
c = conn.cursor()

c.execute('''CREATE TABLE IF NOT EXISTS cache_offline
             (pregunta TEXT PRIMARY KEY, respuesta TEXT)''')
c.execute('''CREATE TABLE IF NOT EXISTS logs_auditoria
             (id INTEGER PRIMARY KEY AUTOINCREMENT, fecha TEXT, pregunta TEXT, respuesta TEXT)''')
conn.commit()

def buscar_en_cache(pregunta):
    c.execute("SELECT respuesta FROM cache_offline WHERE pregunta=?", (pregunta.lower().strip(),))
    resultado = c.fetchone()
    return resultado[0] if resultado else None

def guardar_interaccion(pregunta, respuesta):
    fecha_actual = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT INTO logs_auditoria (fecha, pregunta, respuesta) VALUES (?, ?, ?)", 
              (fecha_actual, pregunta, respuesta))
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

# Carga de documentos en backend
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

try:
    api_key = st.secrets["GROQ_API_KEY"]
except KeyError:
    api_key = None 

# ==========================================
# 3. INTERFAZ GRÁFICA PRINCIPAL Y CSS
# ==========================================
st.title("🍌 Agrobot - Plátano")

# --- BARRA LATERAL: HISTORIAL DE CHAT ---
with st.sidebar:
    st.header("🕒 Historial de Consultas")
    st.caption("Haz clic en una conversación para verla completa en la pantalla principal.")
    
    if st.button("➕ Nueva Consulta", type="primary", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
        
    st.divider()
    
    c.execute("SELECT pregunta, respuesta FROM logs_auditoria ORDER BY id DESC")
    historial_db = c.fetchall()
    
    for i, (preg, resp) in enumerate(historial_db):
        titulo = preg[:30] + "..." if len(preg) > 30 else preg
        
        if st.button(f"🗣️ {titulo}", key=f"hist_{i}", use_container_width=True):
            st.session_state.messages = [
                {"role": "user", "content": preg},
                {"role": "assistant", "content": resp}
            ]

# --- CSS HACK EXTREMO PARA EL MICRÓFONO Y LA BARRA ---
st.markdown(
    """
    <style>
    /* 1. Transformar la caja de texto en una píldora */
    div[data-testid="stChatInput"] {
        padding-bottom: 20px !important;
    }
    div[data-testid="stChatInput"] textarea {
        border-radius: 30px !important; 
        padding-right: 90px !important; /* Hueco reservado para iconos */
        padding-left: 20px !important;
    }
    
    /* 2. Botón de enviar (Círculo azul) */
    div[data-testid="stChatInput"] button {
        background-color: #1a73e8 !important; 
        border-radius: 50% !important;
        height: 38px !important;
        width: 38px !important;
        padding: 0 !important;
        margin-right: 8px !important;
        margin-bottom: 6px !important;
        transition: transform 0.2s;
    }
    div[data-testid="stChatInput"] button:hover {
        transform: scale(1.05); 
    }
    div[data-testid="stChatInput"] button svg {
        fill: white !important;
        color: white !important;
    }

    /* 3. EL HACK DEFINITIVO PARA EL MICRÓFONO */
    div[data-testid="stElementContainer"]:has(iframe[title*="streamlit_mic_recorder"]) {
        position: fixed !important;
        bottom: 50px !important; /* <--- ELEVACIÓN FORZADA AL CENTRO DE LA BARRA */
        z-index: 99999 !important;
        width: 35px !important; 
        height: 35px !important;
        background-color: transparent !important; /* <--- ELIMINA EL FONDO FEO */
        border: none !important;
        box-shadow: none !important;
        border-radius: 50% !important;
        overflow: hidden !important;
    }
    
    /* Forzar transparencia interna del componente */
    div[data-testid="stElementContainer"]:has(iframe[title*="streamlit_mic_recorder"]) iframe {
        background-color: transparent !important;
    }
    
    /* Ajuste en Celulares */
    @media (max-width: 767px) {
        div[data-testid="stElementContainer"]:has(iframe[title*="streamlit_mic_recorder"]) {
            right: 65px !important; /* Justo al lado de la flecha azul */
        }
    }
    
    /* Ajuste en Computadoras */
    @media (min-width: 768px) {
        div[data-testid="stElementContainer"]:has(iframe[title*="streamlit_mic_recorder"]) {
            right: calc(50vw - 295px) !important; /* Anclado dentro de la caja de 730px */
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Renderizamos el micrófono (ahora sin fondo)
prompt_voz = speech_to_text(
    language='es-ES', 
    use_container_width=False, 
    just_once=True, 
    key='STT',
    start_prompt="🎤", 
    stop_prompt="🛑",
)

# ==========================================
# CARGA DE HISTORIAL PERSISTENTE
# ==========================================
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- ZONA DE ENTRADA NATIVA ---
prompt_texto = st.chat_input("Escribe tu duda sobre el cultivo...")

prompt = prompt_texto or prompt_voz

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        respuesta_cache = buscar_en_cache(prompt)
        
        if respuesta_cache:
            st.success("⚡ Respuesta recuperada desde caché local (Modo Offline)")
            st.markdown(respuesta_cache)
            st.session_state.messages.append({"role": "assistant", "content": respuesta_cache})
            
        else:
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
                    
                    guardar_interaccion(prompt, respuesta_ia)
                    st.session_state.messages.append({"role": "assistant", "content": respuesta_ia})
                    
                except Exception as e:
                    st.error(f"Error procesando la solicitud: {str(e)}")