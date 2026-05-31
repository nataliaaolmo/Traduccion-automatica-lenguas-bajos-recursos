"""
Orquestador del pipeline de traducción médica.

Une los tres componentes según la lógica direccional de la Entrega 2:

    Doctor (en_XX) -> Paciente (bajos recursos)   [altos -> bajos]
        1. Clasificar: ¿contiene terminología médica?  (detector.classify)
        2. Si es médica: simplificar a lenguaje llano   (simplifier.simplify)
        3. Traducir                                     (translator.translate)

    Paciente (bajos recursos) -> Doctor (en_XX)    [bajos -> altos]
        Traducir directamente, SIN clasificar ni simplificar.

El clasificador y el simplificador solo operan en inglés y solo en el sentido
altos->bajos, tal y como justifica el informe (C2 y C3).
"""

from detector import classify
from simplifier import simplify
from translator import translate, HIGH_RESOURCE_LANG


def process_turn(text, src_lang, tgt_lang, simplify_mode="api", translate_mode="local"):
    """Procesa un turno de conversación.

    `simplify_mode` y `translate_mode` (cada uno "api" o "local") eligen el backend de
    cada componente por separado. Recomendado: simplificador en "api" (HF Llama, buena
    calidad) y traductor en "local" (mBART fiable; su modo "api" no fuerza el idioma
    destino, ver translator.py).

    Devuelve (translation, steps), donde `steps` documenta lo ocurrido para que la
    interfaz pueda mostrar anotaciones (médica / simplificada).
    """
    steps = {
        "src_lang": src_lang,
        "tgt_lang": tgt_lang,
        "original": text,
        "medical": None,
        "confidence": None,
        "simplified": None,
    }

    working = text

    # El módulo médico (clasificar + simplificar) solo se activa de altos a bajos
    # recursos, es decir, cuando el origen es la lengua del doctor (inglés).
    if src_lang == HIGH_RESOURCE_LANG:
        medical, confidence = classify(text)
        steps["medical"] = medical
        steps["confidence"] = confidence
        if medical:
            working = simplify(text, mode=simplify_mode)
            steps["simplified"] = working

    translation = translate(working, src_lang=src_lang, tgt_lang=tgt_lang, mode=translate_mode)
    steps["translation"] = translation
    return translation, steps


if __name__ == "__main__":
    # Demostración de las dos direcciones.
    print("── Doctor -> Paciente (médico) ──")
    _, steps = process_turn(
        "Administer 500mg of paracetamol twice daily.",
        src_lang="en_XX", tgt_lang="sw_KE",
        simplify_mode="api", translate_mode="local",
    )
    print(steps)

    print("\n── Paciente -> Doctor (directo) ──")
    _, steps = process_turn(
        "Habari, nina maumivu ya kichwa.",
        src_lang="sw_KE", tgt_lang="en_XX",
        simplify_mode="api", translate_mode="local",
    )
    print(steps)
