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
# 🏦 DOC16 - Certificación bancaria (texto/OCR + IA + reglas)
# =========================
def normalize_date_es(x: str | None) -> str | None:
    """Intenta normalizar fechas en español a YYYY-MM-DD cuando sea posible."""
    if not x:
        return None
    s = str(x).strip()
    s_norm = unicodedata.normalize("NFKC", s).upper()

    # ISO ya
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s_norm):
        return s_norm

    # dd/mm/yyyy o dd-mm-yyyy
    m = re.search(r"(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})", s_norm)
    if m:
        dd = int(m.group(1)); mm = int(m.group(2)); yyyy = int(m.group(3))
        if yyyy < 100:
            yyyy += 2000
        return f"{yyyy:04d}-{mm:02d}-{dd:02d}"

    months = {
        "ENERO":1, "FEBRERO":2, "MARZO":3, "ABRIL":4, "MAYO":5, "JUNIO":6,
        "JULIO":7, "AGOSTO":8, "SEPTIEMBRE":9, "SETIEMBRE":9, "OCTUBRE":10,
        "NOVIEMBRE":11, "DICIEMBRE":12
    }
    # "5 de febrero de 2026" / "05 FEBRERO 2026"
    m = re.search(r"(\d{1,2})\s*(?:DE\s*)?([A-ZÁÉÍÓÚÑ]+)\s*(?:DE\s*)?(\d{4})", s_norm)
    if m and m.group(2) in months:
        dd = int(m.group(1)); mm = months[m.group(2)]; yyyy = int(m.group(3))
        return f"{yyyy:04d}-{mm:02d}-{dd:02d}"

    return s.strip()

def extraer_banco_nit_regla(texto: str) -> str | None:
    # NIT 800.244.627-7 / N.I.T. 800.244.627
    m = re.search(r"\bN\.?I\.?T\.?\s*[:\- ]*([0-9\.]{5,15}(?:\-[0-9])?)", texto, re.IGNORECASE)
    if not m:
        return None
    return m.group(1).strip()

def extraer_numero_cuenta_regla(texto: str) -> str | None:
    # Cuenta / No. Cuenta / Cuenta de Inversión
    patterns = [
        r"\bN[°o]?\.?\s*CUENTA\s*[:\- ]*([0-9\- ]{6,30})",
        r"\bCUENTA\s*[:\- ]*([0-9\- ]{6,30})",
        r"\bCUENTA\s+DE\s+INVERSI[ÓO]N\s*[:\- ]*([0-9\- ]{6,30})",
    ]
    for pat in patterns:
        m = re.search(pat, texto, re.IGNORECASE)
        if m:
            cand = re.sub(r"\D", "", m.group(1))
            if 6 <= len(cand) <= 30:
                return cand
    return None

def extraer_estado_cuenta_regla(texto: str) -> str | None:
    t = texto.upper()
    if "INACT" in t:
        return "INACTIVA"
    if "ACTIV" in t:
        return "ACTIVA"
    return None

def extract_doc16_fields_raw(client: OpenAI, text: str) -> str:
    prompt = f"""
A partir del texto de una CERTIFICACIÓN BANCARIA (Colombia), extrae SOLO estos campos y devuelve SOLO JSON válido:
- doc_tipo
- banco_nombre
- banco_nit
- producto_tipo
- producto_nombre
- numero_cuenta
- fecha_apertura
- titular_nombre
- titular_tipo_documento
- titular_num_documento
- estado_cuenta
- fecha_expedicion
- ciudad_expedicion

REGLAS:
- Si un campo no aparece, pon null.
- No inventes datos.
- numero_cuenta y titular_num_documento deben quedar SOLO con dígitos (sin puntos ni espacios) si aplica.
- doc_tipo: si el documento es certificación bancaria escribe "Certificación bancaria"; si no, pon null.
- Devuelve SOLO JSON válido, sin texto adicional.

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

def normalizar_campos_doc16(data: dict, texto: str = "") -> dict:
    data = data or {}

    # Normalizar texto base
    for k in ["doc_tipo","banco_nombre","producto_tipo","producto_nombre",
              "titular_nombre","titular_tipo_documento","ciudad_expedicion"]:
        data[k] = normalize_text(data.get(k))

    # IDs / números
    data["numero_cuenta"] = only_digits(data.get("numero_cuenta"))
    data["titular_num_documento"] = only_digits(data.get("titular_num_documento"))

    # Fechas
    data["fecha_apertura"] = normalize_date_es(data.get("fecha_apertura"))
    data["fecha_expedicion"] = normalize_date_es(data.get("fecha_expedicion"))

    # Reglas complementarias (si IA dejó vacío o raro)
    if not data.get("banco_nit"):
        nit = extraer_banco_nit_regla(texto)
        if nit:
            data["banco_nit"] = nit

    if not data.get("numero_cuenta"):
        nc = extraer_numero_cuenta_regla(texto)
        if nc:
            data["numero_cuenta"] = nc

    if not data.get("estado_cuenta"):
        est = extraer_estado_cuenta_regla(texto)
        if est:
            data["estado_cuenta"] = est

    # Banco nombre: fallback simple por marcas conocidas (evita inventar)
    if not data.get("banco_nombre"):
        t = texto.upper()
        if "BANCOLOMBIA" in t:
            data["banco_nombre"] = "Bancolombia"
        elif "DAVIVIENDA" in t:
            data["banco_nombre"] = "Davivienda"
        elif "SCOTIABANK" in t or "COLPATRIA" in t:
            data["banco_nombre"] = "Scotiabank Colpatria"

    return data

def extract_doc16_text(pdf_bytes: bytes) -> str:
    """Primero intenta texto embebido; si no, OCR a imagen con EasyOCR."""
    text = extract_text_pymupdf(pdf_bytes)
    text = limpiar_texto_para_llm(text)
    if len(text) >= 120:
        return text

    # OCR fallback (1-2 páginas típicamente)
    images = pdf_to_images_pymupdf(pdf_bytes, zoom=2.5)
    ocr_text = ocr_images_easyocr(images)
    return limpiar_texto_para_llm(ocr_text)


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

    # ---------- DOC16 (Certificación bancaria) ----------
    {"doc_id": "DOC16", "Fuente": "DOC16_CertificacionBancaria", "Caracterización variable": "Identificación del documento",
     "Nombre de la Variable": "doc_tipo", "Tipo_Variable": "texto", "Caracterización": "Certificación bancaria / Certifica a quien interese que…"},
    {"doc_id": "DOC16", "Fuente": "DOC16_CertificacionBancaria", "Caracterización variable": "Entidad financiera",
     "Nombre de la Variable": "banco_nombre", "Tipo_Variable": "texto", "Caracterización": "Nombre banco"},
    {"doc_id": "DOC16", "Fuente": "DOC16_CertificacionBancaria", "Caracterización variable": "Entidad financiera",
     "Nombre de la Variable": "banco_nit", "Tipo_Variable": "texto", "Caracterización": "NIT banco"},
    {"doc_id": "DOC16", "Fuente": "DOC16_CertificacionBancaria", "Caracterización variable": "Producto",
     "Nombre de la Variable": "producto_tipo", "Tipo_Variable": "texto", "Caracterización": "Tipo producto (Cuenta de ahorro / corriente)"},
    {"doc_id": "DOC16", "Fuente": "DOC16_CertificacionBancaria", "Caracterización variable": "Producto",
     "Nombre de la Variable": "producto_nombre", "Tipo_Variable": "texto", "Caracterización": "Nombre del producto"},
    {"doc_id": "DOC16", "Fuente": "DOC16_CertificacionBancaria", "Caracterización variable": "Producto",
     "Nombre de la Variable": "numero_cuenta", "Tipo_Variable": "texto", "Caracterización": "Número de cuenta"},
    {"doc_id": "DOC16", "Fuente": "DOC16_CertificacionBancaria", "Caracterización variable": "Producto",
     "Nombre de la Variable": "fecha_apertura", "Tipo_Variable": "fecha", "Caracterización": "Fecha de apertura"},
    {"doc_id": "DOC16", "Fuente": "DOC16_CertificacionBancaria", "Caracterización variable": "Titular",
     "Nombre de la Variable": "titular_nombre", "Tipo_Variable": "texto", "Caracterización": "Nombre del titular"},
    {"doc_id": "DOC16", "Fuente": "DOC16_CertificacionBancaria", "Caracterización variable": "Titular",
     "Nombre de la Variable": "titular_tipo_documento", "Tipo_Variable": "texto", "Caracterización": "Tipo documento titular"},
    {"doc_id": "DOC16", "Fuente": "DOC16_CertificacionBancaria", "Caracterización variable": "Titular",
     "Nombre de la Variable": "titular_num_documento", "Tipo_Variable": "texto", "Caracterización": "Número documento titular"},
    {"doc_id": "DOC16", "Fuente": "DOC16_CertificacionBancaria", "Caracterización variable": "Estado",
     "Nombre de la Variable": "estado_cuenta", "Tipo_Variable": "texto", "Caracterización": "Estado (ACTIVA/INACTIVA)"},
    {"doc_id": "DOC16", "Fuente": "DOC16_CertificacionBancaria", "Caracterización variable": "Expedición",
     "Nombre de la Variable": "fecha_expedicion", "Tipo_Variable": "texto", "Caracterización": "Fecha de expedición (día/mes/año en texto)"},
    {"doc_id": "DOC16", "Fuente": "DOC16_CertificacionBancaria", "Caracterización variable": "Expedición",
     "Nombre de la Variable": "ciudad_expedicion", "Tipo_Variable": "texto", "Caracterización": "Ciudad (si está indicada)"},

]


def fill_master_values(rut_data: dict | None, cc_data: dict | None, doc16_data: dict | None) -> pd.DataFrame:
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

    # DOC16 - Certificación bancaria
    if doc16_data:
        for r in rows:
            if r["doc_id"] == "DOC16":
                key = r["Nombre de la Variable"]
                if key == "doc_tipo":
                    # Valor fijo del diccionario (evita que el LLM invente)
                    r["Valor"] = "Certificación bancaria"
                else:
                    r["Valor"] = doc16_data.get(key)

    return pd.DataFrame(rows)


def dataframe_to_excel_bytes(df: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="extraccion")
    return output.getvalue()


# =========================
# 🖥️ UI Streamlit
# =========================

# =========================
# 🖥️ UI Streamlit (REDISEÑO BONITO)
# =========================
st.set_page_config(page_title="OCR Atenea - Piloto", page_icon="📄", layout="wide")

# ---- 🎨 CSS (look & feel) ----
st.markdown("""
<style>
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
.small-muted { color: rgba(49, 51, 63, 0.65); font-size: 0.95rem; }
.badge {
  display:inline-block; padding: 3px 10px; border-radius: 999px;
  border: 1px solid rgba(49, 51, 63, 0.15);
  background: rgba(49, 51, 63, 0.03);
  font-size: 0.8rem; margin-left: 8px; vertical-align: middle;
}
.card {
  border: 1px solid rgba(49, 51, 63, 0.12);
  border-radius: 16px;
  padding: 16px 16px 10px 16px;
  background: white;
  box-shadow: 0 6px 18px rgba(0,0,0,0.04);
}
.card-title { font-weight: 750; font-size: 1.02rem; margin-bottom: 0.25rem; }
.card-meta { color: rgba(49, 51, 63, 0.65); font-size: 0.85rem; margin-bottom: 0.75rem; }
.stButton > button { border-radius: 12px !important; padding: 0.6rem 1rem !important; }
[data-testid="stDataFrame"] { border-radius: 14px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

# ---- 🧠 Session state para resultados ----
for k in ["rut_data","cc_data","doc16_data","df_master","excel_bytes","logs"]:
    if k not in st.session_state:
        st.session_state[k] = None

# ---- 🧷 Header ----
st.markdown("## 📄 OCR Atenea → Excel <span class='badge'>Piloto</span>", unsafe_allow_html=True)
st.markdown("<div class='small-muted'>Carga RUT, Cédula y Certificación bancaria. Procesa y descarga un Excel consolidado según el diccionario maestro.</div>", unsafe_allow_html=True)
st.write("")

# =========================
# 🧩 Sidebar (configuración)
# =========================
api_key = st.secrets.get("OPENAI_API_KEY", None) if hasattr(st, "secrets") else None
if not api_key:
    api_key = os.environ.get("OPENAI_API_KEY")

with st.sidebar:
    st.markdown("### ⚙️ Configuración")
    st.caption("Si estás en Streamlit Cloud, lo ideal es usar Secrets. Si no, pega la API Key aquí.")
    api_key_input = st.text_input("🔑 OpenAI API Key", type="password", placeholder="sk-...")
    if api_key_input:
        api_key = api_key_input

    st.divider()
    st.markdown("### 🧪 Opciones")
    mostrar_logs = st.toggle("🪵 Mostrar logs", value=False)
    validaciones_extra = st.toggle("✅ Validaciones extra (RUT vs Cédula)", value=True)

    st.divider()
    st.markdown("### 🧭 Ayuda rápida")
    st.caption("• RUT: PDF con texto seleccionable (ideal)\n\n• Cédula: PDF imagen (escaneada)\n\n• DOC16: PDF texto o escaneado")

# =========================
# ✅ Paso 1: Carga (tarjetas)
# =========================
st.markdown("### 1) Carga de documentos")
c1, c2, c3 = st.columns(3, gap="large")

def uploader_card(col, icon, title, meta, key, help_text):
    with col:
        st.markdown(f"<div class='card'><div class='card-title'>{icon} {title}</div><div class='card-meta'>{meta}</div>", unsafe_allow_html=True)
        f = st.file_uploader("", type=["pdf"], accept_multiple_files=False, key=key, help=help_text, label_visibility="collapsed")
        if f is None:
            st.info("📌 Aún no cargado")
        else:
            st.success(f"✅ Cargado: {f.name}")
        st.markdown("</div>", unsafe_allow_html=True)
    return f

rut_pdf = uploader_card(
    c1, "🧾", "RUT (PDF texto)",
    "Recomendado: PDF con texto seleccionable",
    "rut_pdf",
    "Si el RUT es escaneado, puede venir con poco texto y requerir OCR parcial."
)

cc_pdf = uploader_card(
    c2, "🪪", "Cédula (PDF imagen)",
    "Escaneada / foto en PDF",
    "cc_pdf",
    "PDF imagen: el texto no se puede seleccionar; se procesa con OCR."
)

doc16_pdf = uploader_card(
    c3, "🏦", "Certificación bancaria (DOC16)",
    "Formato variable según banco",
    "doc16_pdf",
    "PDF texto o escaneado. Se intenta texto embebido y luego OCR."
)

# =========================
# 🚀 Paso 2: Procesar
# =========================
st.write("")
st.markdown("### 2) Procesar")
files_loaded = any([rut_pdf, cc_pdf, doc16_pdf])

col_btn, col_note = st.columns([1, 2], gap="large")
with col_btn:
    run = st.button("🚀 Procesar todo", type="primary", disabled=not files_loaded)

with col_note:
    if not files_loaded:
        st.warning("Carga al menos un documento para habilitar el procesamiento.")
    else:
        st.info("Listo para procesar ✅")

# =========================
# 🔄 Procesamiento
# =========================
if run:
    if not api_key:
        st.error("Falta la OPENAI_API_KEY. Ponla en Secrets (Cloud) o pégala en configuración.")
        st.stop()

    client = OpenAI(api_key=api_key)

    # Reset resultados anteriores
    st.session_state.rut_data = None
    st.session_state.cc_data = None
    st.session_state.doc16_data = None
    st.session_state.df_master = None
    st.session_state.excel_bytes = None
    st.session_state.logs = []

    progress = st.progress(0, text="Iniciando…")
    step = 0
    total_steps = sum([rut_pdf is not None, cc_pdf is not None, doc16_pdf is not None]) * 3 + 1
    total_steps = max(total_steps, 2)

    def tick(msg):
        nonlocal step
        step += 1
        progress.progress(min(step / total_steps, 1.0), text=msg)
        st.session_state.logs.append(msg)

    # ---- RUT ----
    if rut_pdf:
        rut_bytes = rut_pdf.read()

        tick("📄 RUT: extrayendo texto del PDF…")
        rut_texto = extract_text_pymupdf(rut_bytes)
        rut_texto = limpiar_texto_para_llm(rut_texto)

        if len(rut_texto) < 100:
            tick("⚠️ RUT: poco texto detectado (PDF escaneado). Continuaré con IA igualmente.")
            rut_texto = ""

        tick("🤖 RUT: extrayendo campos con IA…")
        raw = extract_rut_fields_raw(client, rut_texto)
        rut_data = normalizar_campos_rut(safe_json_loads(raw), rut_texto=rut_texto)

        tick("🧠 RUT: validando número de identificación (campo 26)…")
        id_ocr = None
        rut_num = rut_data.get("numero_identificacion")

        if numero_id_es_sospechoso(rut_num):
            id_ocr = ocr_numero_identificacion_desde_campo26(rut_bytes)
            if id_ocr:
                rut_data["numero_identificacion"] = id_ocr
                rut_data["_fuente_numero_identificacion"] = "ocr_campo26"

        if not id_ocr:
            numero_validado = validar_numero_identificacion(rut_texto, rut_data.get("numero_identificacion"))
            if numero_validado:
                rut_data["numero_identificacion"] = numero_validado
                rut_data["_fuente_numero_identificacion"] = "validado_campo26"
            else:
                rut_data["_fuente_numero_identificacion"] = "ia_no_validado"

        st.session_state.rut_data = rut_data

    # ---- CÉDULA ----
    if cc_pdf:
        cc_bytes = cc_pdf.read()

        tick("🪪 Cédula: renderizando PDF a imágenes…")
        images = pdf_to_images_pymupdf(cc_bytes, zoom=2.5)

        tick("🔍 Cédula: OCR (EasyOCR)…")
        cc_ocr_text = ocr_images_easyocr(images)
        cc_ocr_text = limpiar_texto_para_llm(cc_ocr_text)

        tick("🤖 Cédula: extrayendo campos con IA…")
        raw_cc = extract_cc_fields_raw(client, cc_ocr_text)
        cc_data = normalizar_campos_cc(safe_json_loads(raw_cc))
        st.session_state.cc_data = cc_data

    # ---- DOC16 ----
    if doc16_pdf:
        doc16_bytes = doc16_pdf.read()

        tick("🏦 DOC16: extrayendo texto (PDF/OCR)…")
        doc16_texto = extract_doc16_text(doc16_bytes)

        if len(doc16_texto) < 80:
            tick("⚠️ DOC16: poco texto detectado incluso con OCR (calidad baja).")

        tick("🤖 DOC16: extrayendo campos con IA…")
        raw_16 = extract_doc16_fields_raw(client, doc16_texto)
        doc16_data = normalizar_campos_doc16(safe_json_loads(raw_16), texto=doc16_texto)
        st.session_state.doc16_data = doc16_data

    # ---- Validación cruzada (opcional) ----
    if validaciones_extra and st.session_state.rut_data and st.session_state.cc_data:
        rut_num = only_digits(st.session_state.rut_data.get("numero_identificacion"))
        cc_num = only_digits(st.session_state.cc_data.get("doc_numero"))
        if rut_num and cc_num:
            if rut_num == cc_num:
                tick(f"✅ Validación: coinciden (RUT y Cédula) → {rut_num}")
            else:
                tick(f"❌ Validación: NO coinciden → RUT {rut_num} vs Cédula {cc_num}")

    # ---- Consolidado ----
    tick("📌 Generando consolidado (diccionario maestro)…")
    df_master = fill_master_values(st.session_state.rut_data, st.session_state.cc_data, st.session_state.doc16_data)
    st.session_state.df_master = df_master
    st.session_state.excel_bytes = dataframe_to_excel_bytes(df_master)

    progress.progress(1.0, text="Listo ✅")

# =========================
# 📊 Paso 3: Resultados (tabs)
# =========================
st.write("")
st.markdown("### 3) Resultados")

k1, k2, k3 = st.columns(3)
docs_count = sum([rut_pdf is not None, cc_pdf is not None, doc16_pdf is not None])
processed_count = sum([
    st.session_state.rut_data is not None,
    st.session_state.cc_data is not None,
    st.session_state.doc16_data is not None
])

k1.metric("📦 Documentos cargados", docs_count)
k2.metric("✅ Documentos procesados", processed_count)
k3.metric("⬇️ Excel", "Listo" if st.session_state.excel_bytes else "Pendiente")

tab_res, tab_cc, tab_rut, tab_16, tab_master, tab_xls = st.tabs(
    ["🧩 Resumen", "🪪 Cédula", "🧾 RUT", "🏦 DOC16", "📌 Consolidado", "⬇️ Descargar"]
)

with tab_res:
    st.caption("Resumen de ejecución y alertas.")
    if st.session_state.logs:
        if mostrar_logs:
            with st.expander("🪵 Logs"):
                for line in st.session_state.logs:
                    st.write("•", line)
        else:
            st.info("Activa “🪵 Mostrar logs” en el sidebar si quieres ver el detalle.")
    else:
        st.info("Aún no has procesado nada.")

    # Mini resumen de validación
    if validaciones_extra and st.session_state.rut_data and st.session_state.cc_data:
        rut_num = only_digits(st.session_state.rut_data.get("numero_identificacion"))
        cc_num = only_digits(st.session_state.cc_data.get("doc_numero"))
        if rut_num and cc_num:
            if rut_num == cc_num:
                st.success(f"✅ Coinciden: {rut_num}")
            else:
                st.error(f"❌ NO coinciden → RUT: {rut_num} vs Cédula: {cc_num}")
                st.info(f"Fuente RUT numero_identificacion: {st.session_state.rut_data.get('_fuente_numero_identificacion')}")

with tab_cc:
    if st.session_state.cc_data is None:
        st.info("Aún no hay resultado de Cédula.")
    else:
        st.success("✅ Cédula lista")
        st.dataframe(pd.DataFrame([st.session_state.cc_data]), use_container_width=True)

with tab_rut:
    if st.session_state.rut_data is None:
        st.info("Aún no hay resultado de RUT.")
    else:
        st.success("✅ RUT listo")
        st.dataframe(pd.DataFrame([st.session_state.rut_data]), use_container_width=True)

with tab_16:
    if st.session_state.doc16_data is None:
        st.info("Aún no hay resultado de DOC16.")
    else:
        st.success("✅ DOC16 listo")
        st.dataframe(pd.DataFrame([st.session_state.doc16_data]), use_container_width=True)

with tab_master:
    if st.session_state.df_master is None:
        st.info("Aún no hay consolidado.")
    else:
        st.subheader("📌 Consolidado (Diccionario maestro)")
        st.dataframe(st.session_state.df_master, use_container_width=True)

with tab_xls:
    if st.session_state.excel_bytes:
        st.success("✅ Excel consolidado listo para descarga")
        st.download_button(
            "💾 Descargar Excel consolidado (diccionario_maestro.xlsx)",
            data=st.session_state.excel_bytes,
            file_name="diccionario_maestro.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    else:
        st.info("Cuando proceses, aquí aparecerá el botón de descarga.")
