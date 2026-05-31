"""
Carga las claves del archivo .env situado junto a este módulo.

Importar este módulo (en cualquier script del sistema) garantiza que HF_TOKEN esté
disponible en os.environ. `load_dotenv` NO sobrescribe variables ya definidas en el
entorno, así que una clave exportada manualmente tiene prioridad sobre el .env (lee
del .env solo "si no está ya" definida).
"""

import os
from dotenv import load_dotenv

_ENV_PATH = os.path.join(os.path.dirname(__file__), ".env")

# override=False: respeta cualquier variable ya presente en el entorno.
load_dotenv(_ENV_PATH, override=False)
