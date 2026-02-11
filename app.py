# Descripción avance del proceso: Dos documentos (Rut y cedula) con extracción y una regla muy simple de validación

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
def ocr_numero_identificacion_desde_campo26(pdf_bytes: bytes) -> str | None:
    """
    Busca en el PDF el texto 'Número de Identificación' y hace OCR SOLO en un recorte
    cerca de ese campo para capturar la cédula correcta (8-10 dígitos).
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    reader = get_easyocr_reader()

    targets = ["Número de Identificación", "Numero de Identificacion"]

    for page in doc:
        rects = []
        for t in targets:
            rects += page.search_for(t)

        if not rects:
            continue

        r = rects[0]

        clip = fitz.Rect(
            r.x0,
            max(r.y0 - 20, 0),
            min(r.x1 + 350, page.rect.x1),
            min(r.y1 + 80, page.rect.y1)
        )

        pix = page.get_pixmap(clip=clip, dpi=300)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        results = reader.readtext(np.array(img), detail=0)

        candidatos = []
        for s in results:
            dig = re.sub(r"\D", "", s)
            if 8 <= len(dig) <= 10:
                candidatos.append(dig)

        if candidatos:
            candidatos.sort(key=len, reverse=True)
            return candidatos[0]

    return None


def numero_id_es_sospechoso(num: str | None) -> bool:
    if not num:
        return True
    num = re.sub(r"\D", "", str(num))
    if not (8 <= len(num) <= 10):
        return True
    return False


def extraer_numero_identificacion_regla(texto: str) -> str | None:
    """
    Intenta extraer el campo 26. Número de Identificación (CC) del RUT.
    Evita confundirlo con:
      - 4. Número de formulario
      - 5. NIT
    """
    t = " ".join(texto.split())

    m = re.search(r"26\.\s*Número de Identificación\s*([0-9\s]{6,20})", t, re.IGNORECASE)
    if m:
        cand = re.sub(r"\D", "", m.group(1))
        if 6 <= len(cand) <= 11:
            return cand

    m = re.search(r"Cédula de Ciudadanía\s*([0-9\s]{6,20})", t, re.IGNORECASE)
    if m:
        cand = re.sub(r"\D", "", m.group(1))
        if 6 <= len(cand) <= 11:
            return cand

    return None


def corregir_numero_identificacion(data: dict, texto_pdf: str) -> dict:
    """
    Si la IA se equivoca, reemplaza numero_identificacion por el detectado en el PDF.
    """
    regla = extraer_numero_identificacion_regla(texto_pdf)
    if regla:
        data["numero_identificacion"] = regla
    return data


def validar_numero_identificacion(texto: str, candidato: str) -> str | None:
    """
    Valida que el número venga del campo 26 del RUT
    """
    if not candidato:
        return None

    candidato = re.sub(r"\D", "", candidato)

    if not (8 <= len(candidato) <= 10):
        return None

    patron = re.compile(r"26\.\s*Número de Identificación\s*[\n: ]+\s*(\d{8,10})")
    match = patron.search(texto)

    if match:
        return match.group(1)

    return None
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
    Normaliza salida del LLM + aplica validación anti-error para numero_identificacion (campo 26).
    """
    data = data or {}

    # 1) Primero intenta corregir con regla (campo 26)
    data = corregir_numero_identificacion(data, rut_texto)

    # 2) Luego valida que realmente venga del campo 26
    data["numero_identificacion"] = validar_numero_identificacion(
        rut_texto,
        data.get("numero_identificacion")
    )

    # Normalización de strings
    for k in ["primer_apellido", "segundo_apellido", "primer_nombre", "otros_nombres", "tipo_documento"]:
        data[k] = normalize_text(data.get(k))

    return data

# =========================
# 🪪 Extracción Cédula (PDF imagen -> OCR)
# =========================
@st.cache_resource
def get_easyocr_reader():
    return easyocr.Reader(["es"], gpu=False)

# (opcional) alias por compatibilidad si ya usas get_ocr_reader en otros lados
get_ocr_reader = get_easyocr_reader

# (opcional) alias por compatibilidad si ya usas get_ocr_reader en otros lados
get_ocr_reader = get_easyocr_reader

import math

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
    # Inicializaciones (evita NameError)
    rut_data = None
    cc_data = None
    rut_texto = ""

    # ✅ Crear cliente OpenAI
    if not api_key:
        st.error("Falta la OPENAI_API_KEY. Ponla en Secrets (Cloud) o pégala en configuración.")
        st.stop()
    client = OpenAI(api_key=api_key)

    # -------------------------
    # ---- RUT ----
    # -------------------------
    if rut_pdf:
        rut_bytes = rut_pdf.read()  # ✅ guardar bytes una sola vez

        with st.spinner("📄 RUT: extrayendo texto del PDF..."):
            rut_texto = extract_text_pymupdf(rut_bytes)
            rut_texto = limpiar_texto_para_llm(rut_texto)

        if len(rut_texto) < 100:
            st.warning("RUT: detecté muy poco texto. Intentaré extracción por OCR/layout.")
            rut_texto = ""

        with st.spinner("🤖 RUT: extrayendo campos con IA..."):
            raw = extract_rut_fields_raw(client, rut_texto)
            rut_data = normalizar_campos_rut(safe_json_loads(raw), rut_texto=rut_texto)

        # ✅ Fallback OCR SOLO para numero_identificacion (versión campo 26)
        id_ocr = None
        
        # 1) Tomamos lo que quedó tras IA + normalización básica
        rut_num = rut_data.get("numero_identificacion")
        
        # 2) Si es sospechoso → OCR recortado del campo 26
        if numero_id_es_sospechoso(rut_num):
            id_ocr = ocr_numero_identificacion_desde_campo26(rut_bytes)
            if id_ocr:
                rut_data["numero_identificacion"] = id_ocr
                rut_data["_fuente_numero_identificacion"] = "ocr_campo26"
        
        # 3) Si NO hubo OCR exitoso → validar contra texto (campo 26)
        if not id_ocr:
            numero_validado = validar_numero_identificacion(rut_texto, rut_data.get("numero_identificacion"))
            if numero_validado:
                rut_data["numero_identificacion"] = numero_validado
                rut_data["_fuente_numero_identificacion"] = "validado_campo26"
            else:
                rut_data["_fuente_numero_identificacion"] = "ia_no_validado"

    # -------------------------
    # ---- CÉDULA ----
    # -------------------------
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

    # -------------------------
    # ✅ Verificación (NO forzar)
    # -------------------------
    if rut_data and cc_data:
        rut_num = only_digits(rut_data.get("numero_identificacion"))
        cc_num = only_digits(cc_data.get("doc_numero"))

        if rut_num and cc_num:
            if rut_num == cc_num:
                st.success(f"✅ Coinciden: {rut_num}")
            else:
                st.error(f"❌ NO coinciden → RUT: {rut_num} vs Cédula: {cc_num}")
                st.info(f"Fuente RUT numero_identificacion: {rut_data.get('_fuente_numero_identificacion')}")

    # -------------------------
    # ---- Consolidado ----
    # -------------------------
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
