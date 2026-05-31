"""
Componente C3 — Simplificación de terminología médica.

Reformula una frase en inglés con lenguaje técnico médico a lenguaje llano que un
paciente pueda entender, preservando el significado (especialmente las negaciones).
Ejemplo: "edema" -> "swelling caused by fluid".

Usa siempre el mismo modelo, meta-llama/Llama-3.2-1B-Instruct, con dos backends:

    mode="api"    -> Hugging Face Inference API (chat completion). Requiere HF_TOKEN.
    mode="local"  -> el modelo descargado y ejecutado localmente (CPU).

Ambos backends son de Hugging Face, por lo que solo se necesita el token de HF.
El modelo es gated: requiere aceptar su licencia con la cuenta del token.

Solo se invoca en el sentido altos->bajos recursos (doctor -> paciente) y únicamente
cuando el clasificador detecta terminología médica.
"""

import os

# Solo PyTorch: evita que transformers importe TensorFlow/Flax (banners y avisos).
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import env  # noqa: F401  carga .env (HF_TOKEN) en os.environ

MODEL = "meta-llama/Llama-3.1-8B-Instruct"

# Instrucción compartida por ambos backends. Es crítico preservar la polaridad:
# eliminar un "no" en contexto médico puede tener consecuencias graves.
SYSTEM_PROMPT = (
    "You are a medical communication assistant. Rewrite the user's English sentence "
    "so that any layperson with no medical background can understand it. Replace "
    "technical medical terms with simple everyday descriptions (e.g. 'edema' -> "
    "'swelling caused by fluid', 'hypertension' -> 'high blood pressure'). "
    "Preserve the exact meaning and ALL negations (never drop a 'no', 'not', or "
    "'without'). Keep numbers, doses and units unchanged. Do not add information, "
    "warnings or explanations. Output ONLY the rewritten sentence, nothing else."
)

MESSAGES = lambda text: [  # noqa: E731
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": text},
]

# Singleton perezoso para el pipeline local de Llama.
_local_pipe = None


def _hf_token():
    return os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")


def _simplify_api(text):
    from huggingface_hub import InferenceClient

    token = _hf_token()
    if not token:
        raise RuntimeError(
            "Modo 'api' para la simplificación requiere un token de Hugging Face en "
            "HF_TOKEN o HUGGING_FACE_HUB_TOKEN. Usa mode='local' para el modelo offline."
        )
    client = InferenceClient(model=MODEL, token=token)
    response = client.chat_completion(messages=MESSAGES(text), max_tokens=128)
    return response.choices[0].message.content.strip()


def _load_local():
    global _local_pipe
    if _local_pipe is None:
        import torch
        from transformers import pipeline

        _local_pipe = pipeline(
            "text-generation",
            model=MODEL,
            torch_dtype=torch.float32,  # CPU-only: sin float16
            token=_hf_token(),
        )
    return _local_pipe


def _simplify_local(text):
    pipe = _load_local()
    out = pipe(MESSAGES(text), max_new_tokens=128, do_sample=False)
    # El pipeline de chat devuelve la conversación completa; el último mensaje es la
    # respuesta del asistente.
    return out[0]["generated_text"][-1]["content"].strip()


def simplify(text, mode="api"):
    """Simplifica terminología médica en `text` usando el backend `mode`."""
    if mode == "local":
        return _simplify_local(text)
    return _simplify_api(text)


if __name__ == "__main__":
    samples = [
        "The patient presents with acute myocardial infarction and dyspnea.",
        "Do not administer the medication if the patient has no prior history of hypertension.",
    ]
    backend = os.getenv("SIMPLIFY_MODE", "api")
    print(f"Backend: {backend}\n")
    for s in samples:
        print("IN: ", s)
        print("OUT:", simplify(s, mode=backend))
        print()
