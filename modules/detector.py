"""
Componente C2 — Inferencia del detector de dominio médico (para el pipeline).

Carga el modelo TF-IDF + SVM entrenado por `medical_detector.py`
(models/medical_detector.pkl) y expone `is_medical()` y `classify()`.

Se mantiene SEPARADO del script de entrenamiento (medical_detector.py) para que
importar estas funciones NO dispare la descarga de datasets ni el reentrenamiento:
aquí solo se hace un joblib.load barato. Para (re)entrenar el modelo:
    python medical_detector.py
"""

import joblib, math, os

# Ruta absoluta al .pkl (robusta sin importar desde dónde se importe el módulo).
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "medical_detector.pkl")


def _load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"No se encontró el modelo en {MODEL_PATH}. "
            "Entrénalo ejecutando: python medical_detector.py"
        )
    return joblib.load(MODEL_PATH)


# El .pkl se carga una sola vez al importar el módulo (operación barata).
_model = _load_model()


def is_medical(text: str) -> bool:
    return bool(_model.predict([text])[0])


def classify(text: str):
    """Devuelve (is_medical, confidence).

    LinearSVC no da probabilidades calibradas, pero `decision_function` da el margen
    con signo respecto al hiperplano (positivo = médico). Se aproxima una confianza
    en [0, 1] aplicando una sigmoide al margen y tomando la probabilidad de la clase
    predicha. Es orientativa (no calibrada): margen grande -> más seguro.
    """
    score = float(_model.decision_function([text])[0])
    is_med = score > 0
    prob_medical = 1.0 / (1.0 + math.exp(-score))
    confidence = prob_medical if is_med else 1.0 - prob_medical
    return is_med, confidence


if __name__ == "__main__":
    for t in [
        "She was diagnosed with stage 3 hypertension.",
        "What are you doing this weekend?",
    ]:
        med, conf = classify(t)
        print(f"{'médico' if med else 'cotidiano':>9}  {conf*100:5.1f}%  {t}")
