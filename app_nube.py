import streamlit as st
import os
import tempfile
import glob
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS # <-- CAMBIAMOS A FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# ==========================================
# 1. CONFIGURACIÓN DE LA INTERFAZ
# ==========================================
st.set_page_config(page_title="Agrobot Plátano - Tesis", page_icon="🍌", layout="wide")

# ==========================================
# 2. CARGA DE MODELOS (Caché para velocidad)
# ==========================================
@st.cache_resource
def cargar_modelo_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

embeddings = cargar_modelo_embeddings()

# Inicializamos la base de datos (Vacía al principio)
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
    st.session_state.documentos_cargados = False

# ==========================================
# 3. BARRA LATERAL (Configuración y Entrenamiento)
# ==========================================
st.sidebar.title("⚙️ Panel de Control")

api_key = st.sidebar.text_input("Ingresa tu Groq API Key:", type="password")
st.sidebar.markdown("[👉 Consigue tu API Key Gratis aquí](https://console.groq.com/keys)")
st.sidebar.markdown("---")

st.sidebar.subheader("📚 Entrenar al Chatbot")
st.sidebar.caption("Sube manuales sobre nuevas enfermedades, riegos, etc.")
archivos_pdf = st.sidebar.file_uploader("Sube documentos PDF", type="pdf", accept_multiple_files=True)

if st.sidebar.button("🧠 Procesar y Aprender"):
    if archivos_pdf:
        with st.spinner("Estudiando los documentos..."):
            for archivo in archivos_pdf:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(archivo.read())
                    tmp_path = tmp_file.name

                loader = PyPDFLoader(tmp_path)
                docs = loader.load()
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                splits = text_splitter.split_documents(docs)

                # Inyectar a FAISS (Manejo sin errores en la nube)
                if st.session_state.vectorstore is None:
                    st.session_state.vectorstore = FAISS.from_documents(splits, embeddings)
                else:
                    st.session_state.vectorstore.add_documents(splits)
                    
                os.remove(tmp_path) 
                
            st.session_state.documentos_cargados = True
            st.sidebar.success("¡Información procesada y memorizada!")
    else:
        st.sidebar.warning("Selecciona al menos un PDF primero.")

# ==========================================
# 4. LÓGICA PRINCIPAL DEL CHATBOT
# ==========================================
st.title("🍌 Agrobot - Experto en Cultivo de Plátano")
st.markdown("Soy tu ingeniero agrónomo de bolsillo. Baso mis respuestas en manuales técnicos, pero si no encuentro el dato exacto, te daré mis mejores recomendaciones profesionales.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ej: ¿Qué recomiendas si las hojas se ponen amarillas?"):
    
    if not api_key:
        st.error("⚠️ Para que el bot funcione en la nube, ingresa tu Groq API Key en la barra lateral.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Consultando manuales y analizando..."):
            try:
                contexto = ""
                docs_relevantes = []
                
                # Buscar en la memoria FAISS si ya cargamos documentos
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
                    1. RESPONDE DIRECTAMENTE A LA PREGUNTA DEL USUARIO. Tienes PROHIBIDO saludar, decir "¡Bienvenido!" o hacer preguntas de cierre como "¿En qué puedo ayudarte?".
                    2. Prioriza SIEMPRE la información del Contexto para dar tu respuesta técnica.
                    3. Si la respuesta exacta no está en el Contexto, NO digas "no sé". Utiliza tu conocimiento experto general para dar la mejor recomendación posible.
                    4. Actúa 100% como un profesional, ve directo al grano.
                    """),
                    ("user", "{question}")
                ])
                
                mensaje = prompt_template.format_messages(context=contexto, question=prompt)

                respuesta_ia = llm.invoke(mensaje)
                texto_respuesta = respuesta_ia.content

                st.markdown(texto_respuesta)
                
                if docs_relevantes:
                    with st.expander("📚 Fuentes técnicas consultadas (Documentos)"):
                        for i, doc in enumerate(docs_relevantes):
                            st.write(f"**Fragmento {i+1}:** {doc.page_content[:250]}...")

                st.session_state.messages.append({"role": "assistant", "content": texto_respuesta})
                
            except Exception as e:
                st.error(f"Error de conexión con la Nube: {str(e)}")