import os
import re
import json
import io

import fitz  # PyMuPDF
import numpy as np
import pandas as pd
import streamlit as st
from openai import OpenAI
from PIL import Image
import easyocr


# =========================
# 💾 Utilidades generales
# =========================
def safe_json_loads(raw: str) -> dict:
    raw = (raw or "").strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    return json.loads(raw)


def only_digits(x: str | None) -> str | None:
    if x is None:
        return None
    d = re.sub(r"\D", "", str(x))
    return d if d else None


def normalize_text(x: str | None) -> str | None:
    if x is None:
        return None
    x = str(x).strip()
    return x if x else None


def normalize_date(x: str | None) -> str | None:
    """
    Deja fechas como texto, pero intenta normalizar un poco.
    Acepta ejemplos tipo 16-OCT-1986 / 12-NOV-2004 / 2004-11-12
    """
    if not x:
        return None
    x = str(x).strip().upper()

    # Si ya viene ISO-ish
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", x):
        return x

    # dd-MMM-yyyy
    m = re.search(r"(\d{1,2})[-/ ]([A-Z]{3})[-/ ](\d{4})", x)
    if m:
        dd = int(m.group(1))
        mon = m.group(2)
        yyyy = int(m.group(3))
        months = {
            "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
            "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12
        }
        if mon in months:
            return f"{yyyy:04d}-{months[mon]:02d}-{dd:02d}"

    # dd-mm-yyyy
    m = re.search(r"(\d{1,2})[-/](\d{1,2})[-/](\d{4})", x)
    if m:
        dd = int(m.group(1))
        mm = int(m.group(2))
        yyyy = int(m.group(3))
        return f"{yyyy:04d}-{mm:02d}-{dd:02d}"

    return x  # fallback

import unicodedata

def limpiar_texto_para_llm(text: str) -> str:
    """
    Limpia texto extraído de PDF para evitar UnicodeEncodeError:
    - Normaliza Unicode
    - Elimina caracteres de control e invisibles
    - Reemplaza espacios raros por espacios normales
    """
    if not text:
        return ""

    # Normaliza (quita rarezas tipo ligaduras)
    t = unicodedata.normalize("NFKC", text)

    # Reemplazar espacios raros
    t = t.replace("\u00A0", " ")  # non-breaking space
    t = t.replace("\u200B", "")   # zero-width space
    t = t.replace("\u200E", "")   # LRM
    t = t.replace("\u200F", "")   # RLM

    # Eliminar caracteres de control (excepto saltos de línea y tab)
    cleaned = []
    for ch in t:
        cat = unicodedata.category(ch)
        if cat.startswith("C") and ch not in ["\n", "\t"]:
            continue
        cleaned.append(ch)

    t = "".join(cleaned)

    # Compactar espacios
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)

    return t.strip()
# =========================
# ✅ Mejora RUT: anti-código-de-barras (numero_identificacion)
# =========================
def extraer_numero_identificacion_rut_desde_texto(text: str) -> str | None:
    """
    Versión robusta:
    - Prioriza 26. Número de Identificación
    - SOLO acepta 8-10 dígitos (evita falsos positivos típicos)
    """
    if not text:
        return None

    t = " ".join(text.split())

    # Anclaje fuerte al campo 26
    m = re.search(
        r"26\.\s*N[úu]mero\s+de\s+Identificaci[óo]n\s*[:\-]?\s*([0-9][0-9\.\s]{6,20})",
        t,
        flags=re.IGNORECASE,
    )
    if m:
        cand = only_digits(m.group(1))
        if cand and 8 <= len(cand) <= 10:
            return cand

    # Fallback: elegir mejor candidato 8-10 (10 gana)
    nums = [only_digits(x) for x in re.findall(r"\d[\d\.\s]{6,20}", t)]
    nums = [n for n in nums if n and 8 <= len(n) <= 10]
    if not nums:
        return None

    def score(n: str) -> int:
        if len(n) == 10:
            return 100
        if len(n) == 9:
            return 80
        if len(n) == 8:
            return 60
        return 0

    nums = sorted(set(nums), key=score, reverse=True)
    return nums[0]
    
def validar_numero_identificacion_rut(rut_texto: str, candidate: str | None) -> str | None:
    """
    - Descarta 11+ dígitos (códigos / consecutivos)
    - SOLO permite 8-10
    - Si candidate no aparece en texto, intenta anclaje al campo 26
    """
    cand = only_digits(candidate) if candidate else None

    if not cand:
        return extraer_numero_identificacion_rut_desde_texto(rut_texto)

    if len(cand) >= 11:
        return extraer_numero_identificacion_rut_desde_texto(rut_texto)

    if 8 <= len(cand) <= 10:
        if rut_texto:
            t_digits = only_digits(" ".join(rut_texto.split())) or ""
            if cand not in t_digits:
                anchored = extraer_numero_identificacion_rut_desde_texto(rut_texto)
                return anchored or cand
        return cand

    return extraer_numero_identificacion_rut_desde_texto(rut_texto)
    
# =========================
# 📄 Extracción RUT (texto embebido)
# =========================
def extract_text_pymupdf(pdf_bytes: bytes) -> str:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    parts = []
    for page in doc:
        parts.append(page.get_text("text"))
    return "\n".join(parts).strip()


def extract_rut_fields_raw(client: OpenAI, text: str) -> str:
    prompt = f"""
Extrae del siguiente texto (RUT DIAN) ÚNICAMENTE estos campos y devuelve SOLO JSON válido:
- tipo_documento
- numero_identificacion
- primer_apellido
- segundo_apellido
- primer_nombre
- otros_nombres

Reglas:
- Si un campo no aparece, pon null.
- No inventes datos.
- numero_identificacion debe quedar solo con dígitos (sin espacios ni puntos).
- Devuelve únicamente el JSON, sin explicación, sin markdown.

TEXTO:
{text}
"""
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Devuelve SOLO JSON válido. Sin markdown."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )
    return resp.choices[0].message.content

def normalizar_campos_rut(data: dict, rut_texto: str = "") -> dict:
    """
    Normaliza salida del LLM + aplica validación anti-código-de-barras para numero_identificacion.
    """
    data = data or {}

    # ✅ FIX clave: no confiar solo en only_digits. Validar con el texto del RUT.
    data["numero_identificacion"] = validar_numero_identificacion_rut(
        rut_texto=rut_texto,
        candidate=data.get("numero_identificacion"),
    )

    for k in ["primer_apellido", "segundo_apellido", "primer_nombre", "otros_nombres", "tipo_documento"]:
        data[k] = normalize_text(data.get(k))

    return data


# =========================
# 🪪 Extracción Cédula (PDF imagen -> OCR)
# =========================
@st.cache_resource
def get_ocr_reader():
    return easyocr.Reader(["es"], gpu=False)

import math

def extraer_numero_rut_por_layout(pdf_bytes: bytes) -> str | None:
    """
    Busca el label '26. Número de Identificación' por BLOQUES (coordenadas)
    y extrae el número cercano (derecha/abajo). Ideal para RUT en tablas.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    label_pat = re.compile(r"\b26\.\s*N[úu]mero\s+de\s+Identificaci[óo]n\b", re.IGNORECASE)
    num_pat = re.compile(r"\b(\d{8,10})\b")  # ✅ 8-10 (NO 11)

    def dist(ax, ay, bx, by):
        return math.hypot(ax - bx, ay - by)

    for page in doc:
        blocks = page.get_text("blocks")  # (x0,y0,x1,y1,text, block_no, block_type)
        norm = []
        for b in blocks:
            x0, y0, x1, y1, txt, *_ = b
            if not txt or not str(txt).strip():
                continue
            t = " ".join(str(txt).split())
            norm.append((x0, y0, x1, y1, t))

        label_blocks = [bl for bl in norm if label_pat.search(bl[4])]
        if not label_blocks:
            continue

        lx0, ly0, lx1, ly1, _ = label_blocks[0]
        lcx, lcy = (lx0 + lx1) / 2, (ly0 + ly1) / 2

        candidates = []
        for x0, y0, x1, y1, t in norm:
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2

            # cerca en vertical o justo debajo
            vertical_ok = abs(cy - lcy) < 25 or (y0 > ly1 and (y0 - ly1) < 60)
            right_ok = x0 > (lx1 - 10)       # a la derecha del label
            below_ok = y0 >= (ly1 - 5)       # debajo del label (misma columna aprox)

            if (vertical_ok and right_ok) or (below_ok and x0 >= lx0 - 5):
                for m in num_pat.finditer(t):
                    n = m.group(1)
                    candidates.append((n, dist(cx, cy, lcx, lcy)))

        if candidates:
            candidates.sort(key=lambda z: z[1])  # más cercano al label
            return candidates[0][0]

    return None


def extraer_numero_rut_por_ocr(pdf_bytes: bytes, zoom: float = 2.5) -> str | None:
    """
    OCR SOLO de la primera página del RUT y extrae el número del campo 26.
    (fallback cuando el PDF no trae texto usable o el layout no encuentra bien)
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    if doc.page_count == 0:
        return None

    page = doc[0]
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")

    reader = get_ocr_reader()
    lines = reader.readtext(np.array(img), detail=0)
    text = " ".join([str(l) for l in lines if l])
    text = " ".join(text.split())

    # ventana cerca del campo 26
    m = re.search(r"26\.\s*N[úu]mero\s+de\s+Identificaci[óo]n(.{0,80})", text, flags=re.IGNORECASE)
    if m:
        window = m.group(1)
        m2 = re.search(r"\b(\d{8,10})\b", window)
        if m2:
            return m2.group(1)

    # fallback: mejor candidato 10 dígitos
    nums = re.findall(r"\b\d{8,10}\b", text)
    if not nums:
        return None

    def score(n: str) -> int:
        return 100 if len(n) == 10 else (80 if len(n) == 9 else 60)

    nums = sorted(set(nums), key=score, reverse=True)
    return nums[0]
    
def pdf_to_images_pymupdf(pdf_bytes: bytes, zoom: float = 2.5) -> list[Image.Image]:
    """
    Renderiza páginas PDF a imágenes (sin poppler), ideal para Streamlit Cloud.
    zoom 2.5 = más nitidez para OCR.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    images = []
    mat = fitz.Matrix(zoom, zoom)
    for page in doc:
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
        images.append(img)
    return images


def ocr_images_easyocr(images: list[Image.Image]) -> str:
    reader = get_ocr_reader()
    all_lines = []
    for img in images:
        img_np = np.array(img)
        lines = reader.readtext(img_np, detail=0)
        all_lines.extend(lines)
    return "\n".join([l for l in all_lines if l and str(l).strip()]).strip()


def extract_cc_fields_raw(client: OpenAI, ocr_text: str) -> str:
    prompt = f"""
A partir del texto OCR de una CÉDULA DE CIUDADANÍA de Colombia, extrae SOLO estos campos y devuelve SOLO JSON válido:
- doc_pais_emisor
- doc_tipo_documento
- doc_numero
- doc_apellidos
- doc_nombres
- doc_fecha_nacimiento
- doc_lugar_nacimiento
- doc_sexo
- doc_estatura
- doc_grupo_sanguineo_rh
- doc_fecha_expedicion
- doc_lugar_expedicion
- doc_registrador
- doc_codigo_barras
- doc_huella_indice
- doc_firma_titular

Reglas:
- Si un campo no aparece, pon null.
- No inventes datos.
- doc_numero debe quedar solo con dígitos (sin puntos ni espacios).
- doc_estatura en metros (ej: 1.57) si aparece.
- doc_huella_indice y doc_firma_titular deben ser "Sí" o "No" si puedes inferirlo por palabras como INDICE/HUELLA/FIRMA.
- Devuelve únicamente JSON, sin explicación, sin markdown.

TEXTO_OCR:
{ocr_text}
"""
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Devuelve SOLO JSON válido. Sin markdown."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )
    return resp.choices[0].message.content


def normalizar_campos_cc(data: dict) -> dict:
    data = data or {}
    data["doc_pais_emisor"] = normalize_text(data.get("doc_pais_emisor"))
    data["doc_tipo_documento"] = normalize_text(data.get("doc_tipo_documento"))
    data["doc_numero"] = only_digits(data.get("doc_numero"))
    data["doc_apellidos"] = normalize_text(data.get("doc_apellidos"))
    data["doc_nombres"] = normalize_text(data.get("doc_nombres"))
    data["doc_fecha_nacimiento"] = normalize_date(data.get("doc_fecha_nacimiento"))
    data["doc_lugar_nacimiento"] = normalize_text(data.get("doc_lugar_nacimiento"))
    data["doc_sexo"] = normalize_text(data.get("doc_sexo"))
    data["doc_estatura"] = normalize_text(data.get("doc_estatura"))
    data["doc_grupo_sanguineo_rh"] = normalize_text(data.get("doc_grupo_sanguineo_rh"))
    data["doc_fecha_expedicion"] = normalize_date(data.get("doc_fecha_expedicion"))
    data["doc_lugar_expedicion"] = normalize_text(data.get("doc_lugar_expedicion"))
    data["doc_registrador"] = normalize_text(data.get("doc_registrador"))
    data["doc_codigo_barras"] = normalize_text(data.get("doc_codigo_barras"))

    def norm_si_no(v):
        if v is None:
            return None
        v = str(v).strip().lower()
        if v in ["si", "sí", "s", "yes", "true", "1"]:
            return "Sí"
        if v in ["no", "n", "false", "0"]:
            return "No"
        return None

    data["doc_huella_indice"] = norm_si_no(data.get("doc_huella_indice"))
    data["doc_firma_titular"] = norm_si_no(data.get("doc_firma_titular"))
    return data


# =========================
# 📦 Diccionario maestro + Excel consolidado
# =========================
MASTER_ROWS = [
    # ---------- DOC14 (RUT) ----------
    {"doc_id": "DOC14", "Fuente": "DOC14_RUT_DIAN", "Caracterización variable": "Identificación personal",
     "Nombre de la Variable": "tipo_documento", "Tipo_Variable": "texto", "Caracterización": "Tipo documento"},
    {"doc_id": "DOC14", "Fuente": "DOC14_RUT_DIAN", "Caracterización variable": "Identificación personal",
     "Nombre de la Variable": "numero_identificacion", "Tipo_Variable": "texto", "Caracterización": "Número de identificación"},
    {"doc_id": "DOC14", "Fuente": "DOC14_RUT_DIAN", "Caracterización variable": "Identificación personal",
     "Nombre de la Variable": "primer_apellido", "Tipo_Variable": "texto", "Caracterización": "Primer apellido"},
    {"doc_id": "DOC14", "Fuente": "DOC14_RUT_DIAN", "Caracterización variable": "Identificación personal",
     "Nombre de la Variable": "segundo_apellido", "Tipo_Variable": "texto", "Caracterización": "Segundo apellido"},
    {"doc_id": "DOC14", "Fuente": "DOC14_RUT_DIAN", "Caracterización variable": "Identificación personal",
     "Nombre de la Variable": "primer_nombre", "Tipo_Variable": "texto", "Caracterización": "Primer nombre"},
    {"doc_id": "DOC14", "Fuente": "DOC14_RUT_DIAN", "Caracterización variable": "Identificación personal",
     "Nombre de la Variable": "otros_nombres", "Tipo_Variable": "texto", "Caracterización": "Otros nombres"},

    # ---------- DOC12 (Cédula) ----------
    {"doc_id": "DOC12", "Fuente": "DOC12_DocumentoIdentificacion", "Caracterización variable": "Identificación del documento",
     "Nombre de la Variable": "doc_tipo", "Tipo_Variable": "texto", "Caracterización": "Tipo (diccionario)"},
    {"doc_id": "DOC12", "Fuente": "DOC12_DocumentoIdentificacion", "Caracterización variable": "Datos del documento",
     "Nombre de la Variable": "doc_pais_emisor", "Tipo_Variable": "texto", "Caracterización": "País emisor"},
    {"doc_id": "DOC12", "Fuente": "DOC12_DocumentoIdentificacion", "Caracterización variable": "Datos del documento",
     "Nombre de la Variable": "doc_tipo_documento", "Tipo_Variable": "texto", "Caracterización": "Tipo documento"},
    {"doc_id": "DOC12", "Fuente": "DOC12_DocumentoIdentificacion", "Caracterización variable": "Datos del documento",
     "Nombre de la Variable": "doc_numero", "Tipo_Variable": "texto", "Caracterización": "Número"},
    {"doc_id": "DOC12", "Fuente": "DOC12_DocumentoIdentificacion", "Caracterización variable": "Datos del documento",
     "Nombre de la Variable": "doc_apellidos", "Tipo_Variable": "texto", "Caracterización": "Apellidos"},
    {"doc_id": "DOC12", "Fuente": "DOC12_DocumentoIdentificacion", "Caracterización variable": "Datos del documento",
     "Nombre de la Variable": "doc_nombres", "Tipo_Variable": "texto", "Caracterización": "Nombres"},
    {"doc_id": "DOC12", "Fuente": "DOC12_DocumentoIdentificacion", "Caracterización variable": "Datos del documento",
     "Nombre de la Variable": "doc_fecha_nacimiento", "Tipo_Variable": "fecha", "Caracterización": "Fecha de nacimiento"},
    {"doc_id": "DOC12", "Fuente": "DOC12_DocumentoIdentificacion", "Caracterización variable": "Datos del documento",
     "Nombre de la Variable": "doc_lugar_nacimiento", "Tipo_Variable": "texto", "Caracterización": "Lugar de nacimiento"},
    {"doc_id": "DOC12", "Fuente": "DOC12_DocumentoIdentificacion", "Caracterización variable": "Datos del documento",
     "Nombre de la Variable": "doc_sexo", "Tipo_Variable": "texto", "Caracterización": "Sexo"},
    {"doc_id": "DOC12", "Fuente": "DOC12_DocumentoIdentificacion", "Caracterización variable": "Datos del documento",
     "Nombre de la Variable": "doc_estatura", "Tipo_Variable": "texto", "Caracterización": "Estatura"},
    {"doc_id": "DOC12", "Fuente": "DOC12_DocumentoIdentificacion", "Caracterización variable": "Datos del documento",
     "Nombre de la Variable": "doc_grupo_sanguineo_rh", "Tipo_Variable": "texto", "Caracterización": "Grupo sanguíneo y RH"},
    {"doc_id": "DOC12", "Fuente": "DOC12_DocumentoIdentificacion", "Caracterización variable": "Datos del documento",
     "Nombre de la Variable": "doc_fecha_expedicion", "Tipo_Variable": "fecha", "Caracterización": "Fecha de expedición"},
    {"doc_id": "DOC12", "Fuente": "DOC12_DocumentoIdentificacion", "Caracterización variable": "Datos del documento",
     "Nombre de la Variable": "doc_lugar_expedicion", "Tipo_Variable": "texto", "Caracterización": "Lugar de expedición"},
    {"doc_id": "DOC12", "Fuente": "DOC12_DocumentoIdentificacion", "Caracterización variable": "Datos del documento",
     "Nombre de la Variable": "doc_registrador", "Tipo_Variable": "texto", "Caracterización": "Registrador"},
    {"doc_id": "DOC12", "Fuente": "DOC12_DocumentoIdentificacion", "Caracterización variable": "Datos del documento",
     "Nombre de la Variable": "doc_codigo_barras", "Tipo_Variable": "texto", "Caracterización": "Código de barras"},
    {"doc_id": "DOC12", "Fuente": "DOC12_DocumentoIdentificacion", "Caracterización variable": "Biométricos (opcional)",
     "Nombre de la Variable": "doc_huella_indice", "Tipo_Variable": "texto", "Caracterización": "Huella índice"},
    {"doc_id": "DOC12", "Fuente": "DOC12_DocumentoIdentificacion", "Caracterización variable": "Firma (opcional)",
     "Nombre de la Variable": "doc_firma_titular", "Tipo_Variable": "texto", "Caracterización": "Firma titular"},
]


def fill_master_values(rut_data: dict | None, cc_data: dict | None) -> pd.DataFrame:
    rows = [r.copy() for r in MASTER_ROWS]

    # RUT
    if rut_data:
        for r in rows:
            if r["doc_id"] == "DOC14":
                key = r["Nombre de la Variable"]
                r["Valor"] = rut_data.get(key)

    # Cédula
    if cc_data:
        cc_data = cc_data.copy()
        cc_data.setdefault("doc_pais_emisor", "República de Colombia")
        cc_data.setdefault("doc_tipo_documento", "Cédula de ciudadanía")

        for r in rows:
            if r["doc_id"] == "DOC12":
                key = r["Nombre de la Variable"]
                if key == "doc_tipo":
                    r["Valor"] = "Documento de identidad (Cédula de ciudadanía) – imagen anverso/reverso"
                else:
                    r["Valor"] = cc_data.get(key)

    return pd.DataFrame(rows)


def dataframe_to_excel_bytes(df: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="extraccion")
    return output.getvalue()


# =========================
# 🖥️ UI Streamlit
# =========================
st.set_page_config(page_title="OCR Atenea - Piloto", layout="centered")
st.title("📄 OCR Atenea → Excel (Piloto)")
st.caption("Carga RUT (PDF texto) y Cédula (PDF imagen). Descarga un Excel consolidado con el diccionario maestro.")

api_key = st.secrets.get("OPENAI_API_KEY", None) if hasattr(st, "secrets") else None
if not api_key:
    api_key = os.environ.get("OPENAI_API_KEY")

with st.expander("🔑 Configuración (si no está en Secrets)"):
    api_key_input = st.text_input("OpenAI API Key", type="password")
    if api_key_input:
        api_key = api_key_input

col1, col2 = st.columns(2)
with col1:
    rut_pdf = st.file_uploader("📤 Cargar RUT (PDF)", type=["pdf"], key="rut_pdf")
with col2:
    cc_pdf = st.file_uploader("🪪 Cargar Cédula (PDF imagen)", type=["pdf"], key="cc_pdf")

if st.button("🚀 Procesar todo"):
    if not api_key:
        st.error("Falta la OPENAI_API_KEY. Ponla en Secrets (Cloud) o pégala en configuración.")
        st.stop()

    client = OpenAI(api_key=api_key)
    rut_data = None
    cc_data = None

    # ---- RUT ----
if rut_pdf:
    rut_bytes = rut_pdf.read()  # ✅ guardar bytes una sola vez

    with st.spinner("📄 RUT: extrayendo texto del PDF..."):
        rut_texto = extract_text_pymupdf(rut_bytes)
        rut_texto = limpiar_texto_para_llm(rut_texto)

    if len(rut_texto) < 100:
        st.warning("RUT: detecté muy poco texto. Intentaré extracción por OCR/layout.")
        rut_texto = ""  # para que no rompa validadores

    with st.spinner("🤖 RUT: extrayendo campos con IA..."):
        raw = extract_rut_fields_raw(client, rut_texto)
        rut_data = normalizar_campos_rut(safe_json_loads(raw), rut_texto=rut_texto)

    # ✅ Capa 1: Layout (más confiable)
    numero_layout = extraer_numero_rut_por_layout(rut_bytes)

    # ✅ Capa 2: ya está (regex campo 26 dentro de validar_numero_identificacion_rut)
    # rut_data["numero_identificacion"] ya viene validado

    # ✅ Capa 3: OCR fallback si sigue sospechoso o vacío
    rut_num = only_digits(rut_data.get("numero_identificacion"))
    sospechoso = (rut_num is None) or (len(rut_num) < 8) or (len(rut_num) > 10)

    if numero_layout:
        rut_data["numero_identificacion"] = numero_layout
        rut_data["_fuente_numero_identificacion"] = "layout"
    elif sospechoso:
        numero_ocr = extraer_numero_rut_por_ocr(rut_bytes)
        if numero_ocr:
            rut_data["numero_identificacion"] = numero_ocr
            rut_data["_fuente_numero_identificacion"] = "ocr"
        else:
            rut_data["_fuente_numero_identificacion"] = "ia/regex"

    st.success("✅ RUT listo")
    st.dataframe(pd.DataFrame([rut_data]), use_container_width=True)

else:
    st.info("ℹ️ No cargaste RUT. El Excel saldrá con DOC14 en blanco.")
    
    # ---- CÉDULA ----
    if cc_pdf:
        cc_bytes = cc_pdf.read()

        with st.spinner("🪪 Cédula: renderizando PDF a imágenes..."):
            images = pdf_to_images_pymupdf(cc_bytes, zoom=2.5)

        with st.spinner("🔍 Cédula: haciendo OCR (EasyOCR)..."):
            cc_ocr_text = ocr_images_easyocr(images)
            cc_ocr_text = limpiar_texto_para_llm(cc_ocr_text)

        with st.spinner("🤖 Cédula: extrayendo campos con IA..."):
            raw_cc = extract_cc_fields_raw(client, cc_ocr_text)
            cc_data = normalizar_campos_cc(safe_json_loads(raw_cc))

        st.success("✅ Cédula lista")
        st.dataframe(pd.DataFrame([cc_data]), use_container_width=True)
    else:
        st.info("ℹ️ No cargaste Cédula. El Excel saldrá con DOC12 en blanco.")

    
# ✅ Verificación de coincidencia (NO forzar)
if rut_data and cc_data:
    rut_num = only_digits(rut_data.get("numero_identificacion"))
    cc_num = only_digits(cc_data.get("doc_numero"))

    if rut_num and cc_num:
        if rut_num == cc_num:
            st.success(f"✅ Coinciden: {rut_num}")
        else:
            st.error(f"❌ NO coinciden → RUT: {rut_num} vs Cédula: {cc_num}")
            st.info(f"Fuente RUT numero_identificacion: {rut_data.get('_fuente_numero_identificacion')}")
    
    # ---- Consolidado diccionario maestro ----
    df_master = fill_master_values(rut_data, cc_data)
    st.subheader("📌 Consolidado (Diccionario maestro)")
    st.dataframe(df_master, use_container_width=True)

    excel_bytes = dataframe_to_excel_bytes(df_master)
    st.download_button(
        "⬇️ Descargar Excel consolidado (diccionario_maestro.xlsx)",
        data=excel_bytes,
        file_name="diccionario_maestro.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
