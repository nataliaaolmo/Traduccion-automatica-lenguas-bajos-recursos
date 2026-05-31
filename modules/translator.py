"""
Componente C4 — Traducción multilingüe con mBART-50.

Envuelve el modelo facebook/mbart-large-50-many-to-many-mmt en una única función
`translate(text, src_lang, tgt_lang, mode)` con dos backends:

    mode="local"  -> pesos descargados, ejecución en CPU/GPU local (fiable, lento).
    mode="api"    -> Hugging Face Inference API.

OJO (heredado de mBART.py): el router serverless de Hugging Face NO siempre respeta
src_lang/tgt_lang para mBART, por lo que el modo "api" puede devolver traducciones
incorrectas. El modo "local" es el camino fiable; "api" se ofrece para entornos sin
los pesos descargados pero conviene validarlo.

Setup (una vez, modo local):
    pip install transformers torch sentencepiece
    La primera ejecución descarga el modelo (~2.4 GB); luego funciona offline.

Modo api: requiere un token de Hugging Face en la variable de entorno
    HF_TOKEN (o HUGGING_FACE_HUB_TOKEN).

Códigos de idioma: en_XX (inglés), es_XX (español), sw_KE (suajili), xh_ZA (xhosa)...
"""

import os

# Solo PyTorch: evita que transformers importe TensorFlow/Flax (banners y avisos).
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import env  # noqa: F401  carga .env (HF_TOKEN) en os.environ

MODEL = "facebook/mbart-large-50-many-to-many-mmt"

# Lenguas de bajos recursos soportadas por mBART-50 que ofrecemos en la interfaz.
# nombre visible -> código mBART. Verificables con tokenizer.lang_code_to_id.
LOW_RESOURCE_LANGS = {
    "Pashto": "ps_AF",
    "Xhosa": "xh_ZA",
    "Nepali": "ne_NP",
    "Sinhala": "si_LK",
    "Burmese": "my_MM",
    "Mongolian": "mn_MN",
}

# El doctor siempre habla inglés (lengua de altos recursos).
HIGH_RESOURCE_LANG = "en_XX"

# Singletons cargados de forma perezosa: cargar el modelo es la parte lenta, así que
# se reutiliza entre llamadas y NO se carga al importar (la app de Gradio importa
# este módulo antes de traducir nada).
_tokenizer = None
_model = None


def _load():
    global _tokenizer, _model
    if _model is None:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        # Tokenizer SentencePiece "lento"; la ruta rápida está rota para mBART-50.
        _tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=False)
        _model = AutoModelForSeq2SeqLM.from_pretrained(MODEL)
    return _tokenizer, _model


def _translate_local(text, src_lang, tgt_lang):
    tokenizer, model = _load()
    tokenizer.src_lang = src_lang
    inputs = tokenizer(text, return_tensors="pt")
    generated = model.generate(
        **inputs,
        # Forzar el primer token al idioma destino es lo que hace traducir a mBART.
        forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang],
        max_length=200,
    )
    return tokenizer.batch_decode(generated, skip_special_tokens=True)[0]


def _translate_api(text, src_lang, tgt_lang):
    from huggingface_hub import InferenceClient

    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise RuntimeError(
            "Modo 'api' para la traducción requiere un token de Hugging Face en "
            "HF_TOKEN o HUGGING_FACE_HUB_TOKEN. Usa mode='local' si no dispones de él."
        )
    client = InferenceClient(model=MODEL, token=token)
    # NOTA: el router serverless puede ignorar src_lang/tgt_lang para mBART.
    result = client.translation(text, src_lang=src_lang, tgt_lang=tgt_lang)
    return result.translation_text


def translate(text, src_lang=HIGH_RESOURCE_LANG, tgt_lang="es_XX", mode="local"):
    """Traduce `text` de `src_lang` a `tgt_lang` usando el backend `mode`."""
    if mode == "api":
        return _translate_api(text, src_lang, tgt_lang)
    return _translate_local(text, src_lang, tgt_lang)


if __name__ == "__main__":
    phrase = "The weather is nice today and I want to go for a walk."
    print("EN:", phrase)
    print("ES:", translate(phrase, tgt_lang="es_XX"))
    print("XH:", translate(phrase, tgt_lang="xh_ZA"))
    print("SW:", translate(phrase, tgt_lang="sw_KE"))
