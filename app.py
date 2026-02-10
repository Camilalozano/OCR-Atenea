import os
import re
import json
import io

import fitz  # PyMuPDF
import pandas as pd
import streamlit as st
from openai import OpenAI
import numpy as np
from PIL import Image
import easyocr

# -------------------------
# 💾 Utilidades
# -------------------------
def safe_json_loads(raw: str) -> dict:
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    return json.loads(raw)

def normalizar_campos(data: dict) -> dict:
    if data.get("numero_identificacion"):
        data["numero_identificacion"] = re.sub(r"\D", "", str(data["numero_identificacion"]))
    for k in ["primer_apellido","segundo_apellido","primer_nombre","otros_nombres","tipo_documento"]:
        if data.get(k) and isinstance(data[k], str):
            data[k] = data[k].strip()
    return data

def extract_text_pymupdf(pdf_bytes: bytes) -> str:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    parts = []
    for page in doc:
        parts.append(page.get_text("text"))
    return "\n".join(parts).strip()

@st.cache_resource
def get_easyocr_reader():
    # cache para no recargar modelo cada vez
    return easyocr.Reader(["es"], gpu=False)

def ocr_numero_identificacion_desde_campo26(pdf_bytes: bytes) -> str | None:
    """
    Busca en el PDF el texto 'Número de Identificación' y hace OCR SOLO en un recorte
    cerca de ese campo para capturar la cédula correcta (8-10 dígitos).
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    reader = get_easyocr_reader()

    # Frases a buscar (por si cambia el texto)
    targets = ["Número de Identificación", "Numero de Identificacion"]

    for page in doc:
        rects = []
        for t in targets:
            rects += page.search_for(t)

        if not rects:
            continue

        # Tomamos el primer match
        r = rects[0]

        # 👇 Recorte: normalmente el número está a la derecha o un poco abajo del label
        # Ajusta estos márgenes si luego ves otro formato de RUT.
        clip = fitz.Rect(
            r.x0,                # inicio en el label
            max(r.y0 - 20, 0),    # un poquito arriba
            min(r.x1 + 350, page.rect.x1),  # bastante a la derecha
            min(r.y1 + 80, page.rect.y1)    # y algo abajo
        )

        # Render del recorte como imagen (sin poppler)
        pix = page.get_pixmap(clip=clip, dpi=300)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        # OCR
        results = reader.readtext(np.array(img), detail=0)

        # Extraer candidatos numéricos 8-10 dígitos
        candidatos = []
        for s in results:
            dig = re.sub(r"\D", "", s)
            if 8 <= len(dig) <= 10:
                candidatos.append(dig)

        if candidatos:
            # Elegimos el más largo (normalmente 10)
            candidatos.sort(key=len, reverse=True)
            return candidatos[0]

    return None

def numero_id_es_sospechoso(num: str | None) -> bool:
    if not num:
        return True
    num = re.sub(r"\D", "", str(num))
    # cédula típica 8-10 dígitos, y en Colombia usualmente 10
    if not (8 <= len(num) <= 10):
        return True
    # descarta números demasiado “largos” o raros (ya cubierto por len)
    return False
    
def extraer_numero_identificacion_regla(texto: str) -> str | None:
    """
    Intenta extraer el campo 26. Número de Identificación (CC) del RUT.
    Evita confundirlo con:
      - 4. Número de formulario
      - 5. NIT
    """
    t = " ".join(texto.split())  # normaliza espacios

    # Caso ideal: aparece el rótulo exacto del campo 26
    m = re.search(r"26\.\s*Número de Identificación\s*([0-9\s]{6,20})", t, re.IGNORECASE)
    if m:
        cand = re.sub(r"\D", "", m.group(1))
        if 6 <= len(cand) <= 11:
            return cand

    # Fallback: a veces aparece cerca de “Cédula de Ciudadanía”
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

    # Solo dígitos
    candidato = re.sub(r"\D", "", candidato)

    # Regla básica: cédula = 8 a 10 dígitos (normalmente 10)
    if not (8 <= len(candidato) <= 10):
        return None

    # Buscar patrón explícito en el texto
    patron = re.compile(
        r"26\.\s*Número de Identificación\s*[\n: ]+\s*(\d{8,10})"
    )
    match = patron.search(texto)

    if match:
        return match.group(1)

    # Si no se encuentra el patrón, no confiamos
    return None
    
def extract_rut_fields_raw(client: OpenAI, text: str) -> str:
    prompt = f"""
Extrae del siguiente texto (RUT DIAN) ÚNICAMENTE estos campos y devuelve SOLO JSON válido:
- tipo_documento
- numero_identificacion
- primer_apellido
- segundo_apellido
- primer_nombre
- otros_nombres

REGLAS ESTRICTAS:
- No inventes datos.
- El campo numero_identificacion DEBE corresponder al valor que aparece
  JUNTO a "26. Número de Identificación".
- NO usar:
  - Número de formulario
  - NIT
  - Códigos de barras
  - Números largos sin rótulo
- Si hay duda, devuelve null.
- Devuelve SOLO JSON válido, sin texto adicional.


TEXTO:
{text}
"""
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role":"system","content":"Devuelve SOLO JSON válido. Sin markdown."},
            {"role":"user","content": prompt}
        ],
        temperature=0
    )
    return resp.choices[0].message.content

def dataframe_to_excel_bytes(df: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="rut")
    return output.getvalue()


# -------------------------
# 🖥️ UI Streamlit
# -------------------------
st.set_page_config(page_title="OCR Atenea - RUT", layout="centered")
st.title("📄 RUT → Excel (Atenea)")
st.caption("Sube el PDF del RUT y descarga el Excel con los campos extraídos.")

# ✅ API Key desde Secrets (Cloud) o entorno local
api_key = st.secrets.get("OPENAI_API_KEY", None) if hasattr(st, "secrets") else None
if not api_key:
    api_key = os.environ.get("OPENAI_API_KEY")

# Para demo local: permitir pegarla si no está configurada
with st.expander("🔑 Configuración (si no está en Secrets)"):
    api_key_input = st.text_input("OpenAI API Key", type="password")
    if api_key_input:
        api_key = api_key_input

uploaded = st.file_uploader("📤 Cargar RUT (PDF)", type=["pdf"])
min_chars = st.number_input("Umbral mínimo de texto (para detectar escaneo)", 50, 3000, 200)

if st.button("🚀 Procesar"):
    if not api_key:
        st.error("Falta la OPENAI_API_KEY. Ponla en Secrets (Cloud) o pégala arriba.")
        st.stop()
    if not uploaded:
        st.error("Sube un PDF.")
        st.stop()

    client = OpenAI(api_key=api_key)
    pdf_bytes = uploaded.read()

    with st.spinner("📄 Extrayendo texto del PDF..."):
        texto = extract_text_pymupdf(pdf_bytes)

    if len(texto) < int(min_chars):
        st.warning(
            "Detecté muy poco texto (parece escaneado). "
            "Esta versión 1 extrae texto directo del PDF.\n\n"
            "✅ Siguiente paso: le añadimos OCR (EasyOCR) para PDFs escaneados."
        )
        st.stop()

    with st.spinner("🤖 Extrayendo campos con IA..."):
        raw = extract_rut_fields_raw(client, texto)
        data = normalizar_campos(safe_json_loads(raw))
        
        # ✅ Fallback OCR SOLO para numero_identificacion
        id_ocr = None
        
        if numero_id_es_sospechoso(data.get("numero_identificacion")):
            id_ocr = ocr_numero_identificacion_desde_campo26(pdf_bytes)
            if id_ocr:
                data["numero_identificacion"] = id_ocr
        

        # 🔎 Validar SOLO si NO hubo OCR exitoso (porque el PDF puede traer texto “malo”)
        if not id_ocr:
            numero_validado = validar_numero_identificacion(texto, data.get("numero_identificacion"))
            if numero_validado:
                data["numero_identificacion"] = numero_validado
            else:
                st.warning("⚠️ No pude validar el número en el texto del PDF. Dejo el valor extraído (IA).")
                # 👈 NO lo borres; lo dejamos tal cual


    st.success("✅ Extracción lista")
    df = pd.DataFrame([data])
    st.dataframe(df, use_container_width=True)

    excel_bytes = dataframe_to_excel_bytes(df)
    st.download_button(
        "⬇️ Descargar rut_extraido.xlsx",
        data=excel_bytes,
        file_name="rut_extraido.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
