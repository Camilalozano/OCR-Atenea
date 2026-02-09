import os
import re
import json
import io

import fitz  # PyMuPDF
import pandas as pd
import streamlit as st
from openai import OpenAI


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
- numero_identificacion corresponde al campo 26 "Número de Identificación" (NO es "Número de formulario" campo 4, NI el NIT campo 5).
- Si un campo no aparece, pon null.
- No inventes datos.
- numero_identificacion debe quedar solo con dígitos (sin espacios).
- Devuelve únicamente el JSON, sin explicación, sin markdown, sin ```.

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
