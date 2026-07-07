import streamlit as st
import os
import glob
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# ==========================================
# 1. CONFIGURACIÓN DE LA PÁGINA
# ==========================================
st.set_page_config(page_title="Agrobot Plátano", page_icon="🍌", layout="centered")

# ==========================================
# 2. CARGA AUTOMÁTICA DEL CONOCIMIENTO (BACKEND)
# ==========================================
# @st.cache_resource hace que esto solo se ejecute una vez cuando el servidor arranca.
@st.cache_resource(show_spinner="Iniciando el cerebro del Agrobot...")
def preparar_base_de_conocimiento():
    carpeta_docs = "documentos"
    
    # Si la carpeta no existe, no hay problema, devolvemos None
    if not os.path.exists(carpeta_docs):
        return None

    # Buscamos todos los PDFs en la carpeta
    archivos_pdf = glob.glob(f"{carpeta_docs}/*.pdf")
    if not archivos_pdf:
        return None

    docs = []
    for ruta in archivos_pdf:
        try:
            loader = PyPDFLoader(ruta)
            docs.extend(loader.load())
        except Exception as e:
            print(f"Error cargando {ruta}: {e}")

    if not docs:
        return None

    # Dividimos y procesamos el texto
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # Creamos la base de datos vectorial FAISS en memoria
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(splits, embeddings)
    
    return vectorstore

# Ejecutamos la carga en el backend
vectorstore = preparar_base_de_conocimiento()

# ==========================================
# 3. INTERFAZ DEL CHAT (FRONTEND)
# ==========================================
st.title("🍌 Agrobot - Experto en Cultivo de Plátano")
st.markdown("¡Hola! Soy tu ingeniero agrónomo virtual. Estoy entrenado con los manuales oficiales de cultivo. ¿En qué te puedo ayudar hoy?")

# Intentamos obtener la API Key oculta de los secretos de Streamlit
try:
    api_key = st.secrets["GROQ_API_KEY"]
except KeyError:
    st.error("⚠️ Error de configuración: No se encontró la API Key en los secretos del servidor.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ej: ¿Cómo controlo el picudo negro?"):
    
    # 1. Mostrar mensaje del usuario
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Generar respuesta
    with st.chat_message("assistant"):
        with st.spinner("Analizando manuales técnicos..."):
            try:
                contexto = ""
                docs_relevantes = []
                
                # Si hay documentos cargados en el backend, buscamos en ellos
                if vectorstore is not None:
                    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
                    docs_relevantes = retriever.invoke(prompt)
                    contexto = "\n\n".join(doc.page_content for doc in docs_relevantes)

                llm = ChatGroq(
                    groq_api_key=api_key, 
                    model_name="llama-3.1-8b-instant", 
                    temperature=0.2 
                )

                prompt_template = ChatPromptTemplate.from_messages([
                    ("system", """Eres un ingeniero agrónomo experto en el cultivo de plátano.
                    
                    Contexto técnico extraído de manuales oficiales:
                    {context}

                    REGLAS ESTRICTAS PARA RESPONDER:
                    1. RESPONDE DIRECTAMENTE A LA PREGUNTA DEL USUARIO. Tienes PROHIBIDO saludar, presentarte o hacer preguntas de cierre.
                    2. Prioriza SIEMPRE la información del Contexto.
                    3. Si la respuesta exacta no está en el Contexto, utiliza tu conocimiento experto general para dar la mejor recomendación posible (NUNCA digas "no sé" o "no tengo información").
                    4. Actúa 100% como un profesional, ve directo al grano.
                    """),
                    ("user", "{question}")
                ])
                
                mensaje = prompt_template.format_messages(context=contexto, question=prompt)
                respuesta_ia = llm.invoke(mensaje)
                texto_respuesta = respuesta_ia.content

                st.markdown(texto_respuesta)
                
                # Fuentes ocultas en un desplegable (muy profesional para la tesis)
                if docs_relevantes:
                    with st.expander("📚 Fuentes técnicas consultadas (Documentos internos)"):
                        for i, doc in enumerate(docs_relevantes):
                            nombre_archivo = os.path.basename(doc.metadata.get('source', 'Desconocido'))
                            st.caption(f"**De: {nombre_archivo}**\n{doc.page_content[:150]}...")

                st.session_state.messages.append({"role": "assistant", "content": texto_respuesta})
                
            except Exception as e:
                st.error(f"Error interno del servidor: {str(e)}")