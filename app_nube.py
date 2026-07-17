import streamlit as st
import os
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
# 1. BASE DE DATOS SQLITE (Historial y Caché)
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

def cargar_historial():
    """Recupera todas las interacciones previas de la base de datos"""
    c.execute("SELECT pregunta, respuesta FROM logs_auditoria ORDER BY id DESC")
    return c.fetchall()

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
# 3. INTERFAZ GRÁFICA PRINCIPAL
# ==========================================
st.title("🍌 Agrobot - Plátano")

# --- DISEÑO: Botón Flotante (FAB) para el Micrófono ---
st.markdown(
    """
    <style>
    /* Contenedor flotante para el micrófono - Solo posición, sin fondos extraños */
    div[data-testid="stElementContainer"]:has(iframe[title*="streamlit_mic_recorder"]) {
        position: fixed;
        bottom: 90px; /* En móviles: más arriba para NO TOCAR la barra */
        right: 20px;
        z-index: 9999;
        transition: transform 0.2s;
    }
    
    div[data-testid="stElementContainer"]:has(iframe[title*="streamlit_mic_recorder"]):hover {
        transform: scale(1.05); /* Pequeño efecto zoom al pasar el mouse */
    }

    /* Ajuste para Laptops/Monitores: Al lado derecho de la barra, totalmente separado */
    @media (min-width: 850px) {
        div[data-testid="stElementContainer"]:has(iframe[title*="streamlit_mic_recorder"]) {
            bottom: 27px; /* Alineado verticalmente con la barra de texto */
            right: calc(50vw - 425px); /* Posicionado por fuera de la barra, a la derecha */
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Inicializar mensajes de la sesión actual
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- BARRA LATERAL: HISTORIAL DE CHAT ---
with st.sidebar:
    st.header("🕒 Historial de Consultas")
    
    # Botón para limpiar la pantalla principal y hacer una nueva pregunta
    if st.button("➕ Nueva Consulta", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
        
    st.markdown("---")
    st.caption("Tus conversaciones guardadas:")
    
    historial = cargar_historial()
    if historial:
        for preg, resp in historial:
            # Crea un desplegable por cada pregunta pasada
            with st.expander(f"👤 {preg[:35]}..."):
                st.markdown(f"**Tú:** {preg}")
                st.markdown(f"**Agrobot:** {resp}")
    else:
        st.info("Aún no hay consultas guardadas en la base de datos.")

# Imprimimos los mensajes activos en la pantalla principal
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Renderizamos el micrófono (El CSS de arriba lo atrapará y lo hará flotar permanentemente)
prompt_voz = speech_to_text(
    language='es-ES', 
    use_container_width=False, 
    just_once=True, 
    key='STT',
    start_prompt="🎤", 
    stop_prompt="🛑",
)

# --- ZONA DE ENTRADA DE TEXTO ---
prompt_texto = st.chat_input("Escribe tu duda sobre el cultivo...")

# Determinamos si el usuario usó voz o texto
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