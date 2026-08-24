import streamlit as st
import os
import glob
import sqlite3
import datetime
import re
import unicodedata
import difflib
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from streamlit_mic_recorder import speech_to_text # Librería para voz (RF-01)

# ==========================================
# 1. BASE DE DATOS SQLITE (Historial y Caché Offline)
# ==========================================
conn = sqlite3.connect('agrobot_cache.db', check_same_thread=False)
c = conn.cursor()

c.execute('''CREATE TABLE IF NOT EXISTS cache_offline
             (pregunta TEXT PRIMARY KEY, respuesta TEXT)''')
c.execute('''CREATE TABLE IF NOT EXISTS logs_auditoria
             (id INTEGER PRIMARY KEY AUTOINCREMENT, fecha TEXT, pregunta TEXT, respuesta TEXT)''')
conn.commit()

def inicializar_base_offline():
    c.execute("SELECT COUNT(*) FROM cache_offline")
    if c.fetchone()[0] < 5:
        base_conocimiento = [
            (
                "mi planta está amarilla",
                """El amarilleo de las hojas en el cultivo de plátano suele originarse por:

1. **Deficiencia de Nitrógeno (N)**: Amarilleo uniforme y generalizado que empieza en las hojas viejas y avanza a las jóvenes. Se corrige aplicando Urea o fertilizantes nitrogenados.
2. **Deficiencia de Potasio (K)**: Amarillamiento y quemazón en los bordes de las hojas con secado rápido. El plátano es altamente demandante de Potasio; aplique Cloruro de Potasio (KCl) o Sulfato de Potasio.
3. **Deficiencia de Magnesio (Mg) o Hierro (Fe)**: Clorosis interveinal (venas verdes y tejido amarillo entre ellas).
4. **Encharcamiento / Asfixia radical**: Suelos con mal drenaje pudren las raíces e impiden absorber nutrientes. Es prioritario construir y limpiar canales de drenaje.
5. **Problemas fitosanitarios (Sigatoka negra o Moko)**: Lesiones y manchas que inician amarillas y luego se necrosan.

**Recomendaciones:** Realice análisis de suelo, mantenga drenajes a 40 cm de profundidad y aplique fertilización balanceada. Si aplica correctores o agroquímicos, **utilice siempre Equipo de Protección Personal (guantes, mascarilla y gafas)**. Si el síntoma persiste, consulte con un técnico agrícola local."""
            ),
            (
                "como prevenir el moko",
                """Para prevenir y controlar el Moko bacteriano (*Ralstonia solanacearum*) en el plátano:

1. **Desinfección obligatoria de herramientas**: Desinfectar machetes, palas y deshijadores con solución de Yodo agrícola al 20% o Hipoclorito de sodio al 10% entre cada planta.
2. **Semilla certificada**: Utilizar únicamente material de siembra libre de bacterias (vitroplantas o colinos de lotes certificados).
3. **Embolse y desflore temprano**: Retirar la bellota o flor masculina (deschiverar) a tiempo para evitar que insectos polinizadores transmitan la bacteria.
4. **Erradicación de focos**: Si detecta una planta enferma, no la corte ni la traslade. Aísle el sitio con cerca, inyéctela con herbicida (Glifosato) y aplique cal viva en el suelo en un radio de 5 metros (cuarentena de 6 meses).

*Advertencia de seguridad: Utilice equipo de protección personal (guantes, mascarilla y botas) al manipular desinfectantes o químicos. Ante sospecha de brotes, reporte de inmediato a las autoridades fitosanitarias locales.*"""
            ),
            (
                "que es la sigatoka negra y como controlarla",
                """La Sigatoka negra (*Pseudocercospora fijiensis*) es la enfermedad foliar fúngica más destructiva del plátano:

**Manejo Integrado:**
1. **Deshoje fitosanitario semanal**: Cortar las hojas o secciones de hojas con manchas necróticas maduras para reducir la fuente de inóculo.
2. **Mejora del drenaje y ventilación**: Evitar encharcamientos y mantener densidades adecuadas de siembra para disminuir la humedad relativa del dosel.
3. **Nutrición balanceada**: Niveles óptimos de Potasio y Silicio aumentan la resistencia de las hojas al ataque fúngico.
4. **Control químico**: Rotación de fungicidas sistémicos y protectantes en mezcla con aceite mineral bajo monitoreo biológico.

*Advertencia de seguridad: Al aplicar fungicidas, es obligatorio el uso de Equipo de Protección Personal completo (overol, gafas protectoras, mascarilla con filtro para vapores y guantes de nitrilo).*"""
            ),
            (
                "como controlar el picudo negro del platano",
                """El picudo negro (*Cosmopolites sordidus*) es una plaga cuyas larvas perforan túneles en el cormo o cepa del plátano, provocando volcamiento y muerte:

**Estrategias de Control:**
1. **Limpieza de semilla**: Pelar y desinfectar los cormos antes de la siembra sumergiéndolos en agua caliente (54°C por 20 min) o solución insecticida/nematicida.
2. **Trampas de pseudotallo**: Instalar trampas de tipo sándwich o cuña con tallos recién cosechados (con o sin feromona) para captura y monitoreo de adultos (10 a 20 trampas/ha).
3. **Control biológico**: Aplicación del hongo *Beauveria bassiana* en las trampas o al cuello de la planta.
4. **Manejo del cultivo**: Repicar los pseudotallos después de la cosecha para acelerar su descomposición y no dejar refugio a la plaga.

*Advertencia: En caso de requerir insecticidas químicos, use siempre guantes, gafas y mascarilla protectora.*"""
            ),
            (
                "como fertilizar el cultivo de platano",
                """El cultivo de plátano es altamente exigente en nutrientes, principalmente en **Potasio (K)** y **Nitrógeno (N)**:

**Pautas de Fertilización:**
1. **Siembra**: Aplicar 2 a 3 kg de materia orgánica bien compostada por hoyo, más 100 g de fuente fosfórica (DAP o Roca Fosfórica).
2. **Crecimiento vegetativo (1 a 6 meses)**: Fraccionar aplicaciones de Nitrógeno (Urea) y Potasio cada 30 a 45 días en círculo a 30-50 cm de la planta.
3. **Diferenciación floral y llenado**: Incrementar la dosis de Cloruro de Potasio (KCl) para garantizar racimos pesados y frutos largos.
4. **Micronutrientes**: Aplicar Magnesio, Boro y Zinc según los resultados del análisis de suelo.

*Recomendación: Consulte a un ingeniero agrónomo local para formular un plan nutricional exacto según su análisis de suelo.*"""
            ),
            (
                "distancia de siembra y siembra de platano",
                """Para la siembra y establecimiento del cultivo de plátano:

**Distancias recomendadas:**
- **Sistema tradicional (perenne)**: 3.0 m x 3.0 m (1,111 plantas/ha) o 2.5 m x 2.5 m (1,600 plantas/ha).
- **Alta densidad (un solo ciclo)**: 2.0 m x 2.0 m (2,500 plantas/ha) o doble surco (3.0 m x 1.5 m x 1.5 m).

**Preparación y Siembra:**
- Hoyos de 40 cm x 40 cm x 40 cm.
- Colocar en el fondo tierra superficial rica en materia orgánica.
- Sembrar cormos o colinos desinfectados de 1.5 a 2.5 kg o vitroplantas enraizadas.
- Apisonar bien la tierra alrededor para evitar bolsas de aire."""
            ),
            (
                "como hacer el deshije en platano",
                """El deshije consiste en seleccionar el mejor hijo para continuar la producción y eliminar los brotes sobrantes:

**Procedimiento:**
1. **Seleccionar el hijo de espada**: Escoger el retoño con hojas angostas y base cónica profunda que esté orientado en la misma dirección de la hilera.
2. **Eliminar hijos de agua**: Quitar los hijuelos de hojas anchas precoces, pues no desarrollan racimos comerciales.
3. **Frecuencia**: Realizar el deshije cada 6 a 8 semanas utilizando una barretilla o deshijador afilado.
4. **Desinfección**: Desinfectar la herramienta con yodo agrícola al 20% al pasar de una mata a otra para no propagar bacterias."""
            ),
            (
                "buenas practicas agricolas y uso de epp en platano",
                """Las Buenas Prácticas Agrícolas (BPA) garantizan fruta sana y seguridad para los trabajadores:

1. **Equipo de Protección Personal (EPP)**: Es obligatorio usar overol impermeable, botas de hule, guantes de nitrilo, gafas protectoras y mascarilla con filtro para vapores durante la preparación y aplicación de cualquier agroquímico.
2. **Triple lavado**: Todo envase vacío de plaguicida debe lavarse 3 veces, verter el agua en la bomba de aspersión, perforar el envase y llevarlo al centro de acopio.
3. **Periodo de carencia**: Respetar el tiempo mínimo entre la última aplicación de agroquímicos y la cosecha de los racimos.
4. **Higiene en cosecha**: Lavar los frutos con agua limpia para eliminar látex y restos de suciedad."""
            )
        ]
        for preg, resp in base_conocimiento:
            c.execute("INSERT OR REPLACE INTO cache_offline (pregunta, respuesta) VALUES (?, ?)", (preg.lower().strip(), resp))
        conn.commit()

inicializar_base_offline()

def normalizar_texto(texto):
    if not texto: return ''
    texto = texto.lower().strip()
    texto = unicodedata.normalize('NFKD', texto).encode('ASCII', 'ignore').decode('utf-8')
    texto = re.sub(r'[^a-z0-9\s]', ' ', texto)
    palabras = [w for w in texto.split() if len(w) > 1]
    stopwords = {'para', 'como', 'con', 'que', 'por', 'porque', 'del', 'los', 'las', 'una', 'uno', 'unas', 'unos', 'mis', 'tus', 'sus', 'esta', 'estan', 'este', 'estos', 'tengo', 'tiene', 'hacer', 'hago', 'sobre', 'todo', 'cual', 'cuales', 'hay', 'debo', 'debe', 'puedo', 'puede', 'pongo', 'poner', 'dan', 'da'}
    
    sinonimos = {
        'abono': 'fertiliz',
        'abonar': 'fertiliz',
        'fertilizante': 'fertiliz',
        'fertilizacion': 'fertiliz',
        'fertilizar': 'fertiliz',
        'sembrar': 'siembr',
        'siembra': 'siembr',
        'fumigar': 'epp',
        'fumigacion': 'epp',
        'plaguicida': 'epp',
        'pesticida': 'epp',
        'agroquimico': 'epp',
        'proteccion': 'epp',
        'amarillas': 'amarill',
        'amarillo': 'amarill',
        'amarillamiento': 'amarill',
        'clorosis': 'amarill',
        'hojas': 'hoja',
        'plantas': 'planta'
    }
    
    filtradas = [sinonimos.get(p, p) for p in palabras if p not in stopwords]
    return ' '.join(filtradas)

def similitud(q1, q2):
    n1 = normalizar_texto(q1)
    n2 = normalizar_texto(q2)
    if not n1 or not n2: return 0.0
    if n1 == n2: return 1.0
    
    t1 = n1.split()
    t2 = n2.split()
    
    def raiz(w):
        if w.endswith('es') and len(w) > 4: return w[:-2]
        if w.endswith('s') and len(w) > 3: return w[:-1]
        return w
    r1 = {raiz(w) for w in t1}
    r2 = {raiz(w) for w in t2}
    
    inter = r1.intersection(r2)
    union = r1.union(r2)
    jaccard = len(inter) / len(union) if union else 0
    seq = difflib.SequenceMatcher(None, n1, n2).ratio()
    
    keywords_clave = {'moko', 'sigatoka', 'picudo', 'amarill', 'fertiliz', 'siembr', 'deshij', 'deshoj', 'riego', 'drenaj', 'cosech', 'embols', 'epp', 'nematod', 'bacteria', 'hongo', 'nutricion', 'hoja', 'cepa', 'cormo'}
    match_clave = any(k in w for k in keywords_clave for w in inter)
    bonus = 0.30 if match_clave and len(inter) > 0 else 0.0
    
    score = max(jaccard, seq) + bonus
    return min(score, 1.0)

def buscar_en_cache(pregunta_usuario):
    # 1. Coincidencia exacta directa
    c.execute("SELECT respuesta FROM cache_offline WHERE LOWER(TRIM(pregunta))=?", (pregunta_usuario.lower().strip(),))
    res = c.fetchone()
    if res:
        return res[0], 1.0
    
    # 2. Búsqueda por similitud semántica y coincidencia de palabras clave
    c.execute("SELECT pregunta, respuesta FROM cache_offline")
    filas = c.fetchall()
    
    mejor_resp = None
    mejor_score = 0.0
    
    for preg_db, resp_db in filas:
        score = similitud(pregunta_usuario, preg_db)
        if score > mejor_score:
            mejor_score = score
            mejor_resp = resp_db
            
    if mejor_score >= 0.35:
        return mejor_resp, mejor_score
    return None, mejor_score

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
st.set_page_config(page_title="Agrobot Plátano · DevFlow", page_icon="🍌", layout="centered")

# ==========================================
# CSS PERSONALIZADO - TEMA DEVFLOW
# ==========================================
st.markdown("""
<style>
/* ===== GOOGLE FONTS ===== */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ===== VARIABLES GLOBALES ===== */
:root {
    --devflow-primary: #2E7D32;
    --devflow-primary-light: #4CAF50;
    --devflow-primary-dark: #1B5E20;
    --devflow-accent: #FFC107;
    --devflow-accent-dark: #F9A825;
    --devflow-bg-dark: #0F1923;
    --devflow-bg-card: #1A2733;
    --devflow-bg-sidebar: #0D1520;
    --devflow-text-primary: #E8EAED;
    --devflow-text-secondary: #9AA0A6;
    --devflow-border: #2D3748;
    --devflow-user-bubble: #1A3A2A;
    --devflow-assistant-bubble: #1E2A3A;
    --devflow-shadow: 0 4px 24px rgba(0, 0, 0, 0.3);
    --devflow-radius: 12px;
    --devflow-transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

/* ===== FONDO GENERAL ===== */
.stApp {
    background: linear-gradient(160deg, var(--devflow-bg-dark) 0%, #0A1628 50%, #0D1F1A 100%) !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* ===== ENCABEZADO PRINCIPAL ===== */
.stApp header {
    background: transparent !important;
}

h1 {
    font-family: 'Inter', sans-serif !important;
    font-weight: 700 !important;
    font-size: 2rem !important;
    background: linear-gradient(135deg, var(--devflow-accent) 0%, var(--devflow-primary-light) 50%, var(--devflow-accent-dark) 100%);
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
    text-align: center !important;
    padding: 0.5rem 0 !important;
    letter-spacing: -0.5px !important;
    animation: fadeSlideDown 0.8s ease-out;
}

@keyframes fadeSlideDown {
    from { opacity: 0; transform: translateY(-20px); }
    to { opacity: 1; transform: translateY(0); }
}

/* ===== SIDEBAR ===== */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, var(--devflow-bg-sidebar) 0%, #0B1218 100%) !important;
    border-right: 1px solid var(--devflow-border) !important;
    box-shadow: 4px 0 24px rgba(0, 0, 0, 0.4) !important;
}

section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: var(--devflow-accent) !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}

section[data-testid="stSidebar"] .stCaption {
    color: var(--devflow-text-secondary) !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.3px;
}

section[data-testid="stSidebar"] hr {
    border-color: var(--devflow-border) !important;
    opacity: 0.5;
}

/* Botón "Nueva Consulta" */
section[data-testid="stSidebar"] button[kind="primary"] {
    background: linear-gradient(135deg, var(--devflow-primary) 0%, var(--devflow-primary-light) 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    padding: 0.6rem 1rem !important;
    transition: var(--devflow-transition) !important;
    box-shadow: 0 2px 12px rgba(46, 125, 50, 0.3) !important;
    text-transform: none !important;
}

section[data-testid="stSidebar"] button[kind="primary"]:hover {
    background: linear-gradient(135deg, var(--devflow-primary-light) 0%, #66BB6A 100%) !important;
    box-shadow: 0 4px 20px rgba(76, 175, 80, 0.5) !important;
    transform: translateY(-1px) !important;
}

/* Botones del historial */
section[data-testid="stSidebar"] button[kind="secondary"],
section[data-testid="stSidebar"] button:not([kind="primary"]) {
    background: var(--devflow-bg-card) !important;
    color: var(--devflow-text-primary) !important;
    border: 1px solid var(--devflow-border) !important;
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 400 !important;
    font-size: 0.8rem !important;
    padding: 0.5rem 0.75rem !important;
    text-align: left !important;
    transition: var(--devflow-transition) !important;
    margin-bottom: 4px !important;
}

section[data-testid="stSidebar"] button[kind="secondary"]:hover,
section[data-testid="stSidebar"] button:not([kind="primary"]):hover {
    background: rgba(46, 125, 50, 0.15) !important;
    border-color: var(--devflow-primary) !important;
    color: var(--devflow-primary-light) !important;
    transform: translateX(4px) !important;
}

/* ===== BURBUJAS DE CHAT ===== */
div[data-testid="stChatMessage"] {
    border-radius: var(--devflow-radius) !important;
    padding: 1rem 1.25rem !important;
    margin-bottom: 1rem !important;
    border: 1px solid transparent !important;
    animation: fadeIn 0.4s ease-out;
    font-family: 'Inter', sans-serif !important;
    line-height: 1.65 !important;
    max-width: 90% !important;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(8px); }
    to { opacity: 1; transform: translateY(0); }
}

/* Burbuja del usuario */
div[data-testid="stChatMessage"]:has(img[alt="user"]),
div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background: linear-gradient(135deg, var(--devflow-user-bubble) 0%, #1F4030 100%) !important;
    border-color: rgba(46, 125, 50, 0.3) !important;
    margin-left: auto !important;
    box-shadow: 0 2px 12px rgba(46, 125, 50, 0.1) !important;
}

/* Burbuja del asistente */
div[data-testid="stChatMessage"]:has(img[alt="assistant"]),
div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    background: linear-gradient(135deg, var(--devflow-assistant-bubble) 0%, #1A2D42 100%) !important;
    border-color: rgba(255, 193, 7, 0.15) !important;
    margin-right: auto !important;
    box-shadow: 0 2px 12px rgba(0, 0, 0, 0.2) !important;
}

/* Texto dentro de las burbujas */
div[data-testid="stChatMessage"] p,
div[data-testid="stChatMessage"] li,
div[data-testid="stChatMessage"] span {
    color: var(--devflow-text-primary) !important;
    font-size: 0.92rem !important;
}

div[data-testid="stChatMessage"] strong {
    color: var(--devflow-accent) !important;
}

div[data-testid="stChatMessage"] code {
    background: rgba(0, 0, 0, 0.3) !important;
    color: var(--devflow-primary-light) !important;
    padding: 2px 6px !important;
    border-radius: 4px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.85rem !important;
}

/* ===== CAMPO DE ENTRADA DE CHAT ===== */
div[data-testid="stChatInput"] {
    border-radius: 16px !important;
    overflow: hidden;
}

div[data-testid="stChatInput"] textarea {
    background: var(--devflow-bg-card) !important;
    color: var(--devflow-text-primary) !important;
    border: 1px solid var(--devflow-border) !important;
    border-radius: 16px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.9rem !important;
    padding: 0.85rem 1.2rem !important;
    transition: var(--devflow-transition) !important;
}

div[data-testid="stChatInput"] textarea:focus {
    border-color: var(--devflow-primary-light) !important;
    box-shadow: 0 0 0 3px rgba(76, 175, 80, 0.2) !important;
    outline: none !important;
}

div[data-testid="stChatInput"] textarea::placeholder {
    color: var(--devflow-text-secondary) !important;
    font-style: italic;
}

/* Botón de enviar */
div[data-testid="stChatInput"] button {
    background: var(--devflow-primary) !important;
    color: white !important;
    border-radius: 12px !important;
    transition: var(--devflow-transition) !important;
}

div[data-testid="stChatInput"] button:hover {
    background: var(--devflow-primary-light) !important;
    transform: scale(1.05) !important;
}

/* ===== BOTÓN DE MICRÓFONO ===== */
.stAudio button,
button:has(~ [data-testid="stAudio"]),
[class*="mic"] button,
div[data-testid="stVerticalBlock"] > div:has(iframe) button {
    background: linear-gradient(135deg, #E65100, #FF6D00) !important;
    color: white !important;
    border: none !important;
    border-radius: 50px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    padding: 0.6rem 1.5rem !important;
    transition: var(--devflow-transition) !important;
    animation: pulseGlow 2s infinite;
}

@keyframes pulseGlow {
    0%, 100% { box-shadow: 0 0 8px rgba(255, 109, 0, 0.3); }
    50% { box-shadow: 0 0 20px rgba(255, 109, 0, 0.6); }
}

/* ===== SPINNER ===== */
div[data-testid="stSpinner"] {
    color: var(--devflow-primary-light) !important;
}

div[data-testid="stSpinner"] > div {
    border-color: var(--devflow-primary-light) transparent transparent transparent !important;
}

/* ===== ALERTAS Y MENSAJES ===== */
div[data-testid="stAlert"] {
    border-radius: 10px !important;
    font-family: 'Inter', sans-serif !important;
    border: none !important;
}

/* Success */
.stSuccess, div[data-baseweb="notification"][kind="positive"] {
    background: rgba(46, 125, 50, 0.15) !important;
    color: var(--devflow-primary-light) !important;
    border-left: 4px solid var(--devflow-primary-light) !important;
}

/* Error */
.stError, div[data-baseweb="notification"][kind="negative"] {
    background: rgba(211, 47, 47, 0.15) !important;
    color: #EF5350 !important;
    border-left: 4px solid #EF5350 !important;
}

/* Info */
.stInfo, div[data-baseweb="notification"][kind="info"] {
    background: rgba(33, 150, 243, 0.12) !important;
    color: #42A5F5 !important;
    border-left: 4px solid #42A5F5 !important;
}

/* ===== SCROLLBAR PERSONALIZADO ===== */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}

::-webkit-scrollbar-track {
    background: var(--devflow-bg-dark);
}

::-webkit-scrollbar-thumb {
    background: var(--devflow-border);
    border-radius: 3px;
}

::-webkit-scrollbar-thumb:hover {
    background: var(--devflow-primary);
}

/* ===== FOOTER BRANDING ===== */
footer {
    visibility: hidden !important;
}

.stApp::after {
    content: "DevFlow · Agrobot Plátano v2.0";
    position: fixed;
    bottom: 8px;
    left: 50%;
    transform: translateX(-50%);
    font-family: 'Inter', sans-serif;
    font-size: 0.65rem;
    color: var(--devflow-text-secondary);
    opacity: 0.5;
    letter-spacing: 1px;
    text-transform: uppercase;
    z-index: 9999;
    pointer-events: none;
}

/* ===== TEXTOS GENERALES ===== */
.stMarkdown p, .stMarkdown li {
    color: var(--devflow-text-primary) !important;
    font-family: 'Inter', sans-serif !important;
}

.stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
}

/* ===== SEPARADOR HR ===== */
hr {
    border-color: var(--devflow-border) !important;
    opacity: 0.4 !important;
}

/* ===== TOOLTIP / POPOVER ===== */
div[data-baseweb="popover"] {
    background: var(--devflow-bg-card) !important;
    border: 1px solid var(--devflow-border) !important;
    border-radius: var(--devflow-radius) !important;
}

/* ===== RESPONSIVE ===== */
@media (max-width: 768px) {
    h1 {
        font-size: 1.5rem !important;
    }
    
    div[data-testid="stChatMessage"] {
        max-width: 98% !important;
        padding: 0.75rem 1rem !important;
    }
    
    div[data-testid="stChatInput"] textarea {
        font-size: 0.85rem !important;
    }
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def cargar_modelo_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

embeddings = cargar_modelo_embeddings()

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
    st.session_state.documentos_cargados = False

# Carga de documentos en backend
carpeta_pdfs = "datos" if os.path.exists("datos") else "documentos"
if not st.session_state.documentos_cargados and os.path.exists(carpeta_pdfs):
    archivos = glob.glob(f"{carpeta_pdfs}/*.pdf")
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

# Inicializar mensajes de la sesión actual
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- BARRA LATERAL: HISTORIAL DE CHAT ESTILO CHATGPT ---
with st.sidebar:
    st.header("🕒 Historial de Consultas")
    
    if st.button("➕ Nueva Consulta", use_container_width=True, type="primary"):
        st.session_state.messages = []
        st.rerun()
        
    st.markdown("---")
    st.caption("Tus conversaciones guardadas:")
    
    historial = cargar_historial()
    if historial:
        for idx, (preg, resp) in enumerate(historial):
            titulo_boton = f"💬 {preg[:28]}..." if len(preg) > 28 else f"💬 {preg}"
            
            if st.button(titulo_boton, key=f"hist_{idx}", use_container_width=True):
                st.session_state.messages = [
                    {"role": "user", "content": preg},
                    {"role": "assistant", "content": resp}
                ]
                st.rerun() 
    else:
        st.info("Aún no hay consultas guardadas.")

# Imprimimos los mensajes en la pantalla principal
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Renderizamos el botón de voz (RF-01)
prompt_voz = speech_to_text(
    language='es-ES', 
    use_container_width=False, 
    just_once=True, 
    key='STT',
    start_prompt="🎤 Hablar por micrófono", 
    stop_prompt="🛑 Detener",
)

# --- ZONA DE ENTRADA DE TEXTO ---
prompt_texto = st.chat_input("Escribe tu duda sobre el cultivo...")

# Determinamos si el usuario usó voz o texto
prompt = prompt_texto or prompt_voz

if prompt:
    st.session_state.messages = [{"role": "user", "content": prompt}]
    st.rerun()

# Lógica de respuesta de la IA / Modo Offline
if len(st.session_state.messages) > 0 and st.session_state.messages[-1]["role"] == "user":
    ultimo_prompt = st.session_state.messages[-1]["content"]
    
    with st.chat_message("assistant"):
        # 1. Búsqueda en la base de datos offline (SQLite)
        respuesta_cache, score = buscar_en_cache(ultimo_prompt)
        
        if respuesta_cache:
            st.success("⚡ Respuesta recuperada desde la base técnica local (Modo Sin Conexión)")
            st.markdown(respuesta_cache)
            st.session_state.messages.append({"role": "assistant", "content": respuesta_cache})
            
        else:
            respuesta_generada = None
            modo_offline = False
            
            # 2. Si hay clave API, intentar consultar el modelo en la nube (Groq)
            if api_key:
                try:
                    with st.spinner("Analizando consulta con manuales técnicos..."):
                        contexto = ""
                        if st.session_state.documentos_cargados and st.session_state.vectorstore is not None:
                            retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})
                            docs_relevantes = retriever.invoke(ultimo_prompt)
                            contexto = "\n\n".join(doc.page_content for doc in docs_relevantes)

                        llm = ChatGroq(
                            groq_api_key=api_key, 
                            model_name="openai/gpt-oss-20b", 
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
                        
                        mensaje = prompt_template.format_messages(context=contexto, question=ultimo_prompt)
                        respuesta_generada = llm.invoke(mensaje).content
                        
                except Exception as e:
                    # En caso de fallo de red o API, activar modo de rescate offline
                    modo_offline = True
            else:
                modo_offline = True

            # 3. MODO DE RESCATE OFFLINE (Extracción de información técnica desde los manuales PDF locales)
            if respuesta_generada is None or modo_offline:
                with st.spinner("Buscando en manuales técnicos locales (Modo Sin Conexión)..."):
                    if st.session_state.documentos_cargados and st.session_state.vectorstore is not None:
                        try:
                            retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})
                            docs_relevantes = retriever.invoke(ultimo_prompt)
                            if docs_relevantes:
                                fragmentos = []
                                for doc in docs_relevantes:
                                    texto_limpio = re.sub(r'\s+', ' ', doc.page_content.strip())
                                    if len(texto_limpio) > 40:
                                        fragmentos.append(f"• {texto_limpio[:380]}...")
                                
                                if fragmentos:
                                    respuesta_generada = (
                                        "📖 **Información técnica extraída de los manuales del cultivo de plátano:**\n\n"
                                        + "\n\n".join(fragmentos[:3])
                                        + "\n\n---\n"
                                        + "⚠️ **Recomendación (RF-06):** Información recuperada en modo sin conexión. Ante problemas agronómicos severos, consulte presencialmente a un técnico agrícola local.\n"
                                        + "🛡️ **Seguridad (RNF-06):** Al aplicar fungicidas, insecticidas o fertilizantes, utilice siempre Equipo de Protección Personal (EPP: mascarilla, guantes y botas)."
                                    )
                                    st.info("📶 Modo Sin Conexión: Información recuperada desde los manuales PDF del proyecto.")
                        except Exception:
                            pass
                    
                    if not respuesta_generada:
                        respuesta_generada = (
                            "ℹ️ **Modo Sin Conexión:** No se encontró información específica para esta consulta en la base local.\n\n"
                            "🌿 **Temas disponibles sin conexión en tu base de datos:**\n"
                            "- *¿Por qué mis plantas están amarillas?* (Deficiencias de N-P-K, encharcamiento, Sigatoka)\n"
                            "- *¿Cómo prevenir y controlar el Moko?* (Desinfección de herramientas y erradicación)\n"
                            "- *¿Qué es la Sigatoka negra y cómo controlarla?* (Deshoje y fungicidas)\n"
                            "- *¿Cómo combatir el picudo negro del plátano?* (Trampas y desinfección de cormos)\n"
                            "- *¿Cómo fertilizar el cultivo de plátano?* (Nutrición y abonado)\n"
                            "- *Distancia y densidad de siembra*\n"
                            "- *Deshije y selección del hijo de espada*\n"
                            "- *Buenas Prácticas Agrícolas (BPA) y uso de EPP*"
                        )
                        st.warning("⚠️ Sin conexión a internet. Mostrando base de conocimiento local.")

            st.markdown(respuesta_generada)
            guardar_interaccion(ultimo_prompt, respuesta_generada)
            st.session_state.messages.append({"role": "assistant", "content": respuesta_generada})
