from datasets import load_dataset
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import joblib, os, random

# ── 1. CARGAR DATASETS ──────────────────────────────────────────────────────

print("Cargando ChatDoctor (médico)...")
medical_ds = load_dataset("avaliev/chat_doctor", split="train")

print("Cargando DailyDialog (cotidiano)...")
daily_ds = load_dataset("agentlans/li2017dailydialog", split="train")

# ── 2. EXTRAER TEXTO ────────────────────────────────────────────────────────

medical_texts = [row["input"] for row in medical_ds if row.get("input")]

def extract_daily_text(row):
    convs = row.get("conversations", [])
    texts = [c["value"] for c in convs if c.get("from") == "human"]
    return " ".join(texts) if texts else ""

daily_texts = [extract_daily_text(row) for row in daily_ds]
daily_texts = [t for t in daily_texts if t.strip()]

# ── 3. EQUILIBRAR CLASES ────────────────────────────────────────────────────

random.seed(42)
n = min(len(medical_texts), len(daily_texts))
medical_texts = random.sample(medical_texts, n)
daily_texts   = random.sample(daily_texts, n)
print(f"Muestras por clase: {n} (total: {2*n})")

# ── 4. SPLIT ────────────────────────────────────────────────────────────────

texts  = medical_texts + daily_texts
labels = [1] * n + [0] * n

X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)

# ── 5. ENTRENAR ─────────────────────────────────────────────────────────────

model = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
    ("svm",   LinearSVC(class_weight="balanced", max_iter=2000))
])

print("Entrenando...")
model.fit(X_train, y_train)

# ── 6. EVALUAR (accuracy + F1-Macro) ────────────────────────────────────────

y_pred = model.predict(X_test)
print("\n── Resultados ──")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred, target_names=["cotidiano", "médico"]))

# ── 7. GUARDAR MODELO ───────────────────────────────────────────────────────

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/medical_detector.pkl")
print("Modelo guardado en models/medical_detector.pkl")

# ── 8. FUNCIÓN PARA USAR DESDE EL PIPELINE ──────────────────────────────────

_model = joblib.load("models/medical_detector.pkl")

def is_medical(text: str) -> bool:
    return bool(_model.predict([text])[0])


# ── 9. TESTS BÁSICOS ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    casos = [
        ("Doctor, I have been experiencing chest pain and difficulty breathing.", True),
        ("She was diagnosed with stage 3 hypertension.",                         True),
        ("Administer 500mg of paracetamol twice daily.",                         True),
        ("What are you doing this weekend?",                                     False),
        ("I really enjoyed the movie last night.",                               False),
    ]
    print("\n── Tests básicos ──")
    for texto, esperado in casos:
        resultado = is_medical(texto)
        estado = "✓" if resultado == esperado else "✗"
        print(f"{estado} '{texto[:50]}...' → {'médico' if resultado else 'cotidiano'}")