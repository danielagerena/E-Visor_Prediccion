# Crea la estructura base del repositorio E-Visor.
# Solo arma carpetas y archivos de texto; los datos y modelos se copian aparte.
import os

# Los 16 bloques del campus, uno por carpeta de modelo
BLOQUES = [
    'B10_ARQ', 'B12_DERE', 'B15_BIBL', 'B17_POLI', 'B18_PARQ',
    'B3_RECT', 'B4_PRIM', 'B5_BACH', 'B7_CTIC', 'B7_TAC',
    'B8_AA', 'B8_CPA', 'B8_LABS', 'B9_SFA1', 'B9_SFA2', 'ECOVILLA'
]

# Carpeta raiz del repo
RAIZ = 'e-visor-app'


def crear_estructura():
    # Carpeta de datos
    os.makedirs(os.path.join(RAIZ, 'datos'), exist_ok=True)

    # Una carpeta por bloque dentro de modelos_predictivos
    for b in BLOQUES:
        os.makedirs(os.path.join(RAIZ, 'modelos_predictivos', b), exist_ok=True)

    # requirements.txt con las dependencias
    requirements = "streamlit\ndarts[torch]\npandas\nmatplotlib\njoblib\n"
    with open(os.path.join(RAIZ, 'requirements.txt'), 'w', encoding='utf-8') as f:
        f.write(requirements)

    # .gitignore para no subir basura al repo
    gitignore = "__pycache__/\n*.pyc\n.DS_Store\n.ipynb_checkpoints/\n"
    with open(os.path.join(RAIZ, '.gitignore'), 'w', encoding='utf-8') as f:
        f.write(gitignore)

    # README con recordatorios de que falta pegar
    readme = (
        "# E-Visor: prediccion de consumo por bloque\n\n"
        "App de Streamlit que muestra la prediccion de 24h por edificio.\n\n"
        "## Pendiente por copiar\n"
        "- app.py y predictor.py en la raiz\n"
        "- El CSV en datos/datos_preprocesados_10min.csv\n"
        "- Los 4 artefactos de cada bloque en modelos_predictivos/<bloque>/\n"
    )
    with open(os.path.join(RAIZ, 'README.md'), 'w', encoding='utf-8') as f:
        f.write(readme)

    # Archivos vacios de marca para que git conserve las carpetas vacias.
    # Git no sube carpetas sin archivos; este truco las mantiene hasta que
    # copies los modelos reales.
    with open(os.path.join(RAIZ, 'datos', '.gitkeep'), 'w') as f:
        f.write('')
    for b in BLOQUES:
        ruta = os.path.join(RAIZ, 'modelos_predictivos', b, '.gitkeep')
        with open(ruta, 'w') as f:
            f.write('')

    print(f"Estructura creada en la carpeta: {RAIZ}")
    print(f"Carpetas de bloques: {len(BLOQUES)}")


if __name__ == '__main__':
    crear_estructura()