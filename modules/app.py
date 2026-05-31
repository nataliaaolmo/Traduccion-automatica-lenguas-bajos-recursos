"""
Interfaz web (Gradio) — chat médico bidireccional Doctor <-> Paciente.

Diseño de "paneles bilingües": dos columnas lado a lado que muestran la MISMA
conversación, cada una en su idioma. El doctor (izquierda) la ve toda en inglés; el
paciente (derecha) la ve toda en su lengua. En cada panel, los mensajes propios se
alinean a la derecha y los del interlocutor a la izquierda (efecto espejo).

Cada turno pasa por el pipeline:

    Doctor (EN)   -> [clasificar -> simplificar si es médico] -> traducir -> Paciente
    Paciente (XX) -> traducir directamente                                  -> Doctor

Ejecutar:  python app.py   (abre una URL local)
"""

import html
import os

# Solo usamos PyTorch: evita que transformers importe TensorFlow/Flax (banners de
# oneDNN, aviso de Keras) y acelera el arranque. Debe ir antes de importar transformers.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import gradio as gr

from translator import LOW_RESOURCE_LANGS, HIGH_RESOURCE_LANG
from pipeline import process_turn

LANG_NAMES = list(LOW_RESOURCE_LANGS.keys())

# Tema base con acento médico (verde azulado) y neutros suaves.
THEME = gr.themes.Soft(primary_hue="teal", neutral_hue="slate")

# Estética de los elementos NO-chat (cabecera, controles, títulos de panel).
CUSTOM_CSS = """
.gradio-container { max-width: 1140px !important; margin: auto !important; }

#app-header { text-align: center; padding: 22px 16px 6px; }
#app-header .title { font-size: 1.85rem; font-weight: 700; letter-spacing: -0.01em;
    background: linear-gradient(90deg, #0d9488, #2563eb);
    -webkit-background-clip: text; background-clip: text; color: transparent; margin: 0; }
#app-header .subtitle { color: var(--body-text-color-subdued); font-size: 0.95rem;
    max-width: 720px; margin: 10px auto 0; line-height: 1.5; }

/* Tarjeta que agrupa los controles */
.controls-card { border-radius: 14px !important; padding: 6px 14px 12px !important;
    box-shadow: var(--shadow-drop-lg); }

/* Títulos de cada panel, con fondo tintado por rol (sin borde que se curve) */
.panel-title { font-weight: 650; font-size: 1.05rem !important; padding: 10px 16px !important;
    border-radius: 10px; margin-bottom: 6px !important; }
.panel-title.doctor  { background: rgba(37, 99, 235, 0.12); }
.panel-title.patient { background: rgba(13, 148, 136, 0.12); }

/* Botón de limpiar a lo ancho, discreto */
#clear-btn { color: var(--body-text-color-subdued); }

/* Aviso de prueba de concepto al pie */
#poc-disclaimer { text-align: center; color: var(--body-text-color-subdued);
    font-size: 0.82rem; line-height: 1.45; max-width: 780px; margin: 16px auto 6px;
    padding-top: 12px; border-top: 1px solid var(--border-color-primary); }

/* Estructura interna de cada mensaje (envuelto en .msg con número de turno) */
.msg { transition: box-shadow 0.12s ease; border-radius: 8px; }
.msg .turn-no { font-size: 0.72rem; font-weight: 700; opacity: 0.45; margin-right: 7px; }
.msg .msg-tag { font-size: 0.78rem; opacity: 0.7; margin-top: 6px; }
.msg .msg-err { font-size: 0.85rem; margin-top: 6px; color: #ef4444; }
.msg details.msg-simpl { margin-top: 8px; }
.msg details.msg-simpl summary { cursor: pointer; font-size: 0.85rem; opacity: 0.85; }

/* Resaltado vinculado: al pasar el ratón por un mensaje se tiñe ese mensaje y su
   traducción (misma clase turn-N) en el otro panel. Gestionado por LINK_HOVER_JS. */
.msg.linked-hl { box-shadow: inset 0 0 0 9999px rgba(45, 212, 191, 0.18); }

/* ── Estabilidad de tamaños ─────────────────────────────────────────────────
   Textos largos se parten (no ensanchan la columna) y los paneles mantienen una
   altura fija con scroll interno, sin importar lo larga que sea la conversación. */
.msg, .msg .msg-text, .msg .msg-tag, .msg blockquote, .msg summary {
    overflow-wrap: anywhere; word-break: break-word; }
#doctor-chat, #patient-chat { height: 460px !important; max-height: 460px !important;
    flex-grow: 0 !important; }
/* Las dos columnas se reparten el ancho a partes iguales y no se encogen */
#chat-row > .column, #chat-row > div { flex: 1 1 0 !important; min-width: 0 !important; }

/* Oculta el botón de papelera "Limpiar chat" propio del Chatbot de Gradio (no se
   elimina con buttons=[]). Se cubren las etiquetas en español e inglés. */
#doctor-chat [aria-label*="Limpiar"], #patient-chat [aria-label*="Limpiar"],
#doctor-chat [aria-label*="Clear"],   #patient-chat [aria-label*="Clear"],
#doctor-chat [title*="Limpiar"],      #patient-chat [title*="Limpiar"] {
    display: none !important; }
"""

# Script de la interfaz, inyectado en <head> al lanzar (launch(head=...)). Se usa head
# y NO el parámetro js= de launch porque, en esta versión de Gradio, el js= no llega a
# ejecutarse; un <script> en head sí (verificado con Selenium).
#  (1) Hover vinculado: al pasar el ratón por un mensaje (.msg con clase turn-N) se
#      resalta él y el mensaje turn-N del otro panel. Empareja por la CLASE turn-N
#      (el saneador de Gradio elimina los atributos data-*, pero conserva las clases).
#      Delegación a nivel de documento -> sobrevive a los re-render de los chats.
#  (2) Scroll sincronizado (proporcional) entre los dos paneles.
LINK_HOVER_HEAD = """
<script>
(function(){
  function setup(){
    // (0) Copiar solo el texto visible, no el HTML del envoltorio. El botón de copiar
    // de Gradio llama a navigator.clipboard.writeText(content), y content es nuestro
    // <div class="msg">...</div>. Lo interceptamos y extraemos el texto de .msg-text.
    if(navigator.clipboard && navigator.clipboard.writeText && !navigator.clipboard.__msgPatched){
      var origWrite = navigator.clipboard.writeText.bind(navigator.clipboard);
      navigator.clipboard.writeText = function(t){
        try{
          if(typeof t === 'string' && t.indexOf('class="msg') !== -1){
            var tmp = document.createElement('div');
            tmp.innerHTML = t;
            var el = tmp.querySelector('.msg-text');
            t = el ? el.textContent : (tmp.textContent || t);
          }
        }catch(e){}
        return origWrite(t);
      };
      navigator.clipboard.__msgPatched = true;
    }

    var clearHL = function(){
      document.querySelectorAll('.msg.linked-hl').forEach(function(e){
        e.classList.remove('linked-hl'); });
    };
    if(!window.__msgHoverBound){
      window.__msgHoverBound = true;
      document.addEventListener('mouseover', function(e){
        var m = e.target.closest ? e.target.closest('.msg') : null;
        clearHL();
        if(!m) return;
        var tc = null;
        m.classList.forEach(function(c){ if(c.indexOf('turn-')===0) tc = c; });
        if(!tc) return;
        document.querySelectorAll('.msg.'+tc).forEach(function(x){
          x.classList.add('linked-hl'); });
      });
    }
    var scroller = function(root){
      if(!root) return null;
      var pref = root.querySelector('.bubble-wrap');
      if(pref && pref.scrollHeight - pref.clientHeight > 4) return pref;
      var els = root.querySelectorAll('*');
      for(var i=0;i<els.length;i++){
        var oy = getComputedStyle(els[i]).overflowY;
        if((oy==='auto'||oy==='scroll') && els[i].scrollHeight - els[i].clientHeight > 4) return els[i];
      }
      return pref;
    };
    var lock = false;
    var link = function(src, dst){
      if(!src || !dst || src.__syncBound) return;
      src.__syncBound = true;
      src.addEventListener('scroll', function(){
        if(lock) return;
        lock = true;
        var denom = (src.scrollHeight - src.clientHeight) || 1;
        dst.scrollTop = (src.scrollTop/denom) * ((dst.scrollHeight - dst.clientHeight) || 1);
        requestAnimationFrame(function(){ lock = false; });
      });
    };
    setInterval(function(){
      var A = scroller(document.querySelector('#doctor-chat'));
      var B = scroller(document.querySelector('#patient-chat'));
      if(A && B){ link(A, B); link(B, A); }
    }, 1000);
  }
  if(document.readyState !== 'loading') setup();
  else document.addEventListener('DOMContentLoaded', setup);
})();
</script>
"""


# El contenido de cada mensaje se construye como HTML puro (no markdown) para poder
# envolverlo en un <div class="msg" data-turn="N"> sin perder formato. Ese envoltorio
# permite (a) numerar los turnos y (b) vincular el hover entre paneles por id de turno.

def _text_html(text):
    """Texto plano de un mensaje, escapado, como HTML."""
    return f'<span class="msg-text">{html.escape(text)}</span>'


def _error_html(text, err):
    return (f'<span class="msg-text">{html.escape(text)}</span>'
            f'<div class="msg-err">⚠️ <strong>Error:</strong> {html.escape(str(err))}</div>')


def _doctor_bubble(message, steps):
    """Contenido HTML de un mensaje del doctor: frase original, etiqueta médico/no
    médico (con confianza) y, si hubo simplificación, un desplegable colapsado."""
    parts = [f'<span class="msg-text">{html.escape(message)}</span>']

    conf = steps.get("confidence")
    conf_txt = f" · {conf * 100:.0f}% conf." if conf is not None else ""
    if steps.get("medical") is True:
        parts.append(f'<div class="msg-tag">🏥 médico{conf_txt}</div>')
    elif steps.get("medical") is False:
        parts.append(f'<div class="msg-tag">💬 no médico{conf_txt}</div>')

    if steps.get("simplified"):
        simp = html.escape(steps["simplified"])
        parts.append(
            '<details class="msg-simpl"><summary>✏️ Ver simplificación</summary>'
            f"<blockquote>{simp}</blockquote></details>"
        )

    return "".join(parts)


def _wrap(inner, turn_no):
    """Envuelve el contenido con su número de turno.

    El id del turno va en la CLASE (turn-N), no en un atributo data-*, porque el
    saneador de Gradio ELIMINA los atributos data-* pero conserva las clases. Así el
    JS empareja el mismo turno en ambos paneles para el resaltado vinculado.
    """
    return (f'<div class="msg turn-{turn_no}">'
            f'<span class="turn-no">#{turn_no}</span>{inner}</div>')


def _render(convo):
    """Convierte la conversación lógica en los dos historiales de chat.

    Cada turno guarda su versión en inglés (`en`) y en la lengua del paciente (`xx`),
    además de quién habló. En el panel del doctor, el doctor es "user" (derecha) y el
    paciente "assistant" (izquierda); en el panel del paciente, al revés.
    """
    doctor_view, patient_view = [], []
    for i, t in enumerate(convo, start=1):
        # De-espejado: el doctor SIEMPRE a la izquierda (assistant) y el paciente
        # SIEMPRE a la derecha (user) en AMBOS paneles, para que el turno i quede a la
        # misma altura y lado en los dos. El número de turno y el data-turn (en _wrap)
        # alinean visualmente y permiten vincular el hover por id.
        role = "assistant" if t["speaker"] == "doctor" else "user"
        doctor_view.append({"role": role, "content": _wrap(t["en"], i)})
        patient_view.append({"role": role, "content": _wrap(t["xx"], i)})
    return doctor_view, patient_view


def _doctor_turn(message, lang_name, simplify_mode, translate_mode, convo):
    if not message.strip():
        d, p = _render(convo)
        return convo, d, p, ""
    lang_code = LOW_RESOURCE_LANGS[lang_name]
    try:
        translation, steps = process_turn(
            message, src_lang=HIGH_RESOURCE_LANG, tgt_lang=lang_code,
            simplify_mode=simplify_mode, translate_mode=translate_mode,
        )
        convo = convo + [{
            "speaker": "doctor",
            "en": _doctor_bubble(message, steps),
            "xx": _text_html(translation),
        }]
    except Exception as e:
        convo = convo + [{
            "speaker": "doctor",
            "en": _error_html(message, e),
            "xx": '<span class="msg-text">⚠️ (error de traducción)</span>',
        }]
    d, p = _render(convo)
    return convo, d, p, ""


def _patient_turn(message, lang_name, translate_mode, convo):
    if not message.strip():
        d, p = _render(convo)
        return convo, d, p, ""
    lang_code = LOW_RESOURCE_LANGS[lang_name]
    try:
        # Sentido bajos->altos: sin clasificar ni simplificar.
        translation, _ = process_turn(
            message, src_lang=lang_code, tgt_lang=HIGH_RESOURCE_LANG,
            translate_mode=translate_mode,
        )
        convo = convo + [{
            "speaker": "patient",
            "en": _text_html(translation),
            "xx": _text_html(message),
        }]
    except Exception as e:
        convo = convo + [{
            "speaker": "patient",
            "en": '<span class="msg-text">⚠️ (translation error)</span>',
            "xx": _error_html(message, e),
        }]
    d, p = _render(convo)
    return convo, d, p, ""


def _clear():
    return [], [], []


def _patient_header(lang_name):
    return f"🧑 Paciente · {lang_name}"


with gr.Blocks(title="Traductor médico — lenguas de bajos recursos") as demo:
    gr.HTML(
        '<div id="app-header">'
        '<div class="title">🌍 Traductor médico para lenguas de bajos recursos</div>'
        '<div class="subtitle">Cada panel muestra la misma conversación en su propio '
        "idioma. El doctor escribe en inglés (se clasifica y, si es médico, se "
        "simplifica antes de traducir); el paciente responde en su lengua.</div>"
        "</div>"
    )

    with gr.Group(elem_classes="controls-card"):
        with gr.Row():
            lang = gr.Dropdown(
                choices=LANG_NAMES, value=LANG_NAMES[0],
                label="Idioma objetivo",
                info="Lengua de bajos recursos del paciente"
            )
            simplify_mode = gr.Radio(
                choices=["api", "local"], value="api",
                label="Simplificador",
                info="(Llama-3.1-8B Instruct)"
            )
            translate_mode = gr.Radio(
                choices=["api", "local"], value="api",
                label="Traductor",
                info="(mBART)"
            )

    convo = gr.State([])

    with gr.Row(equal_height=True, elem_id="chat-row"):
        # ── Panel del doctor (inglés) ──────────────────────────────────────────
        with gr.Column():
            gr.Markdown("🩺 Doctor · English", elem_classes=["panel-title", "doctor"])
            doctor_chat = gr.Chatbot(
                elem_id="doctor-chat",
                height=460, show_label=False, group_consecutive_messages=False,
                buttons=[],  # quita copy/share/copy_all de la esquina superior derecha
                placeholder="La conversación en inglés.",
            )
            with gr.Row():
                doctor_box = gr.Textbox(
                    placeholder="Type in English…", show_label=False,
                    scale=8, container=False, lines=1, max_lines=4,
                )
                doctor_btn = gr.Button("Send", variant="primary", scale=1, min_width=80)

        # ── Panel del paciente (lengua seleccionada) ───────────────────────────
        with gr.Column():
            patient_header = gr.Markdown(
                _patient_header(LANG_NAMES[0]), elem_classes=["panel-title", "patient"],
            )
            patient_chat = gr.Chatbot(
                elem_id="patient-chat",
                height=460, show_label=False, group_consecutive_messages=False,
                buttons=[],  # quita copy/share/copy_all de la esquina superior derecha
                placeholder="La conversación en la lengua elegida.",
            )
            with gr.Row():
                patient_box = gr.Textbox(
                    placeholder="Escribe en tu idioma…", show_label=False,
                    scale=8, container=False, lines=1, max_lines=4,
                )
                patient_btn = gr.Button("➤", variant="primary", scale=1, min_width=80)

    clear_btn = gr.Button(
        "🗑️ Limpiar conversación", variant="secondary", elem_id="clear-btn",
    )

    gr.HTML(
        '<div id="poc-disclaimer">⚠️ <strong>Prueba de concepto.</strong> '
        "Prototipo académico (MULCIA · PLN 2025-26) con fines demostrativos; "
        "no debe utilizarse para la toma de decisiones clínicas reales.</div>"
    )

    # El encabezado del panel del paciente sigue al idioma seleccionado.
    lang.change(_patient_header, inputs=lang, outputs=patient_header)

    # Doctor -> Paciente (simplificador + traductor)
    doctor_inputs = [doctor_box, lang, simplify_mode, translate_mode, convo]
    doctor_outputs = [convo, doctor_chat, patient_chat, doctor_box]
    doctor_btn.click(_doctor_turn, doctor_inputs, doctor_outputs)
    doctor_box.submit(_doctor_turn, doctor_inputs, doctor_outputs)

    # Paciente -> Doctor (solo traductor)
    patient_inputs = [patient_box, lang, translate_mode, convo]
    patient_outputs = [convo, doctor_chat, patient_chat, patient_box]
    patient_btn.click(_patient_turn, patient_inputs, patient_outputs)
    patient_box.submit(_patient_turn, patient_inputs, patient_outputs)

    clear_btn.click(_clear, outputs=[convo, doctor_chat, patient_chat])


if __name__ == "__main__":
    # En Gradio 6, theme/css/footer_links se pasan a launch(). footer_links=[] elimina
    # el pie ("Usar vía API · Construido con Gradio · Configuración").
    demo.launch(theme=THEME, css=CUSTOM_CSS, head=LINK_HOVER_HEAD, footer_links=[])
