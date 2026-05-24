# Traducción Automática de Lenguas de Bajos Recursos
**Máster Universitario en Lógica, Computación e Inteligencia Artificial (MULCIA)**  
Procesamiento del Lenguaje Natural · Curso 2025–26

> Herramienta de traducción automática en el ámbito médico para Médicos Sin Fronteras,
> con detección y simplificación de terminología médica previa a la traducción. Esta traducción será implementada para el ingés y el suajili.

---

## Instalación y uso

### Requisitos
- Python 3.10+

### Instalación

```bash
git clone https://github.com/vuestro-repo/traduccion-bajos-recursos.git
cd traduccion-bajos-recursos
python -m venv .venv

# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

pip install -r requirements.txt
```

### Dependencias principales
- datasets
- scikit-learn
- joblib

## Arquitectura del sistema

El sistema sigue el siguiente flujo según la dirección de la traducción:

### C1 · Captación y transcripción de voz a texto (STT)
*Pendiente de documentar*

### C2 · Detección de contexto médico (TF-IDF + SVM)

Clasificador binario que determina si una frase contiene terminología médica.
Solo se activa en la dirección de traducción inglés → suajili.

**Datasets utilizados:**
- [ChatDoctor / avaliev](https://huggingface.co/datasets/avaliev/chat_doctor): 95.588 conversaciones reales médico-paciente (HealthCareMagic). Se usan como clase positiva (médico).
- [DailyDialog / roskoN](https://huggingface.co/datasets/roskoN/dailydialog): 11.118 diálogos cotidianos. Se usan como clase negativa (no médico).

**Modelo:** Pipeline de scikit-learn con `TfidfVectorizer` (10.000 features, unigramas y bigramas) + `LinearSVC`.

**Resultados:**
| Métrica | Valor |
|---------|-------|
| Accuracy | 0.9969 |
| F1-Macro | 1.00 |

Se ha definido la función is_medical() que trabaja con este modelo. De forma que se podrá comprobar si un texto es médico o no como se indica con estos ejemplos:
```python
is_medical("Doctor, I have chest pain and difficulty breathing.")  # → True
is_medical("What are you doing this weekend?")                    # → False
```

---

### C3 · Simplificación de lenguaje médico (Llama-3-8B / GPT-4)
*Pendiente de documentar*

### C4 · Traducción multilingüe (mBART)
*Pendiente de documentar*

### C5 · Síntesis de voz (TTS)
*Pendiente de documentar*

---

## Estructura del repositorio
```
/
├── modules/
│   ├── medical_detector.py   # C2: Detección de contexto médico
│   ├── simplifier.py         # C3: Simplificación de lenguaje médico
│   ├── translator.py         # C4: Traducción con mBART
│   ├── stt.py                # C1: Speech to Text
│   └── tts.py                # C5: Text to Speech
├   └──  models/
│          └── medical_detector.pkl  # Modelo entrenado C2
├── pipeline.py               # Orquestador principal
├── config.yaml               # Configuración del sistema
├── requirements.txt          # Dependencias
└── README.md
```

## Autores

- Mario Vázquez Lechuga
- Cristian Caballero Sánchez
- Natalia Olmo Villegas
- Manuel Enciso Martínez

---
