# E-Visor — Pronóstico de consumo eléctrico del Ecocampus Laureles

Sistema de pronóstico de demanda eléctrica a 24 horas para los 16 bloques del
Ecocampus Laureles de la Universidad Pontificia Bolivariana, desarrollado con el
UPB Smart Energy Center (SEC).

**Aplicación publicada:** https://evisor-prediccion.streamlit.app/

Este archivo es el punto de entrada para cualquier persona que necesite
entender, operar o continuar el proyecto. `OPERACION.md` explica cómo se opera
el sistema día a día, y el informe metodológico en `docs/` documenta cada
decisión de análisis.

---

## 1. Qué hace el sistema

Dos procesos independientes que solo comparten la fuente de datos:

| Proceso | Qué responde | Cuándo se ejecuta |
|---|---|---|
| **Pronóstico** | Cuánto va a consumir cada bloque en las próximas 24 horas | Diario y automático |
| **Agrupamiento** | Qué bloques comparten ritmo de uso, calidad de energía o condiciones de voltaje | Puntual, al cerrar un período |

El pronóstico predice de forma conjunta tres variables —potencia activa, factor
de potencia y voltaje promedio— en pasos de 10 minutos, con un único modelo para
los 16 bloques. La energía importada no tiene modelo propio: se deriva de la
potencia, porque un intervalo de 10 minutos es un sexto de hora.

**Error medido sobre agosto de 2026**, un mes que el modelo no vio durante el
entrenamiento, agregado sobre quince bloques (B8_CPA queda excluido por sus
problemas de medición): 24,48 % en potencia activa, 2,41 % en factor de potencia
y 0,43 % en voltaje. La referencia de control —repetir la semana anterior— da
29,23 % en potencia activa.

Las métricas provienen de `modelo_global`, el modelo del experimento, que nunca
vio agosto. El que corre en producción es `modelo_produccion`, reentrenado con
todo el período incluido agosto. Las cifras son por tanto una estimación del
desempeño esperado del modelo desplegado, no una medición sobre él.

---

## 2. Estado actual y lo que falta

| Etapa | Estado |
|---|---|
| Extracción de datos desde la fuente | **Manual.** Único paso sin automatizar |
| Preprocesamiento a rejilla de 10 minutos | Automatizado (`src/preprocesar_crudo.py`) |
| Lectura desde base de datos | Construida, sin fuente conectada (`src/ingesta.py`) |
| Entrenamiento del modelo | Reproducible (`notebooks/`, en Colab con GPU) |
| Inferencia diaria | Automatizada (GitHub Actions, 6:00 a. m. Colombia) |
| Publicación del tablero | Automática al actualizar `main` |
| Detección de anomalías y alertas | **No implementado.** Siguiente fase |

Mientras no exista la conexión automática a la fuente, la inferencia diaria
opera sobre un **reloj simulado**: `datos/reloj_simulado.txt` guarda el instante
del histórico que el sistema asume como presente, y cada ejecución lo adelanta
un día. Esto permite verificar el circuito completo y comparar cada predicción
contra los datos reales de ese día.

---

## 3. Estructura del repositorio

```
app/                      Aplicación Streamlit (no carga el modelo, solo lee resultados)
  Tablero.py                Punto de entrada: predicción a 24 horas por bloque
  pages/                    Páginas adicionales (perfiles de consumo)
datos/                    Datos de entrada y salidas del proceso diario
  datos_preprocesados_10min.csv   Rejilla de 10 minutos, insumo del modelo
  predicciones.csv                Última predicción publicada
  historia_reciente.csv           Medidas recientes que muestra el tablero
  reloj_simulado.txt              Instante simulado actual
docs/                     Informe metodológico y capturas
modelos/v4/               Modelo entrenado, límites de escalado y métricas
notebooks/                Entrenamiento y análisis exploratorio
resultados/               Salidas de los análisis (agrupamiento, energía)
scripts/                  Análisis y utilidades ejecutables
src/                      Paquete principal
tests/                    Pruebas del preprocesamiento, calidad, base y alertas
.github/workflows/        Automatización
```

### Archivos principales de `src/`

| Archivo | Función |
|---|---|
| `evisor_lstm.py` | Construcción de ventanas, modelo, métricas. Compartido por entrenamiento e inferencia |
| `inferir_v4.py` | Genera la predicción de 24 horas. Es lo que corre en producción |
| `preprocesar_crudo.py` | Convierte el consolidado crudo en la rejilla de 10 minutos |
| `preprocesamiento.py` | Funciones puras de limpieza y remuestreo |
| `config.py` | Configuración central. Los valores sensibles vienen de variables de entorno |
| `ingesta.py` | Lectura incremental desde una fuente: archivos, base de datos o API |
| `cargar_historico.py` | Carga el CSV preprocesado en la tabla `mediciones` |
| `db.py`, `esquema.sql` | Acceso a base de datos y esquema |
| `calidad.py` | Compuerta de calidad previa al entrenamiento y a la predicción |
| `alertas.py`, `vigilante.py` | Motor de alertas y verificación de salud del proceso |
| `promover.py` | Compuertas de promoción de modelos |
| `features.py` | Construcción de series de la arquitectura anterior. Solo lo usa `promover.py` |

---

## 4. Cómo ejecutar cada pieza

Todos los comandos se corren desde la raíz del repositorio.

### Preprocesar datos crudos

```bash
python -m src.preprocesar_crudo \
    --entrada datos/datos_crudos_consolidados.csv \
    --salida  datos/datos_preprocesados_10min.csv
```

Imprime un reporte de validación: valores fuera de rango convertidos a nulo,
reinicios del contador corregidos, filas y nulos por mes, y los vacíos de más de
una hora. Conviene revisarlo antes de entrenar o predecir sobre datos nuevos.

### Generar la predicción de 24 horas

Desde un archivo:

```bash
pip install -r requirements-inferencia.txt
python -m src.inferir_v4 \
    --artefactos modelos/v4 \
    --csv datos/datos_preprocesados_10min.csv \
    --salida datos/predicciones.csv \
    --salida-historia datos/historia_reciente.csv
```

Desde la base de datos, que es como corre en un servidor:

```bash
pip install -r requirements-servidor.txt
python -m src.inferir_v4 --artefactos modelos/v4 --desde-bd --guardar-bd
```

Acepta `--origen "AAAA-MM-DD HH:MM"` para predecir desde un instante concreto
del histórico, que es lo que usa el reloj simulado. Sin ese argumento toma el
último dato disponible, que es el comportamiento correcto en producción.

### Ejecutar el tablero localmente

```bash
pip install -r requirements.txt
streamlit run app/Tablero.py
```

### Análisis de agrupamiento

```bash
pip install -r requirements-analisis.txt
python scripts/clustering.py
```

### Medir el error por bloque

```bash
python scripts/evaluar_por_bloque.py \
    --artefactos modelos/v4 --nombre modelo_global \
    --csv datos/datos_preprocesados_10min.csv
```

Usa `modelo_global`, que es el modelo del experimento y no vio agosto. El modelo
de producción sí lo vio, así que medir sobre ese mes con él no significa nada.
Imprime cuatro escenarios, incluido el error sin B8_CPA, que es el que reporta
el informe.

---

## 5. Automatización

| Flujo | Programación | Qué hace |
|---|---|---|
| `predicciones_diarias.yml` | 11:00 UTC (6:00 a. m. Colombia) | Predice 24 horas, adelanta el reloj simulado y publica los resultados |
| `pruebas.yml` | En cada push | Ejecuta las pruebas de `tests/` |
| `ingesta.yml` | Sin horario | Ingesta incremental. Se reactiva cuando exista la fuente automática |
| `vigilante.yml` | Sin horario | Verificación de salud. Se reactiva junto con la ingesta |

Los flujos programados solo se ejecutan desde la rama por defecto. La rama de
trabajo es `pipeline-produccion`; `main` es la que publica.

**Si el pronóstico pasa a correr en el servidor del SEC, hay que apagar la
programación de `predicciones_diarias.yml`.** De lo contrario dos procesos
generarían la predicción del mismo día.

---

## 6. Requisitos por entorno

El proyecto separa las dependencias para que cada entorno instale solo lo suyo:

| Archivo | Para qué |
|---|---|
| `requirements.txt` | Aplicación Streamlit. Sin TensorFlow a propósito |
| `requirements-inferencia.txt` | Predicción desde un CSV, fuera de Colab |
| `requirements-servidor.txt` | Predicción en el servidor, leyendo de la base |
| `requirements-pipeline.txt` | Ingesta y procesos automáticos |
| `requirements-analisis.txt` | Agrupamiento y análisis de energía |

La versión de TensorFlow está fijada en 2.20.0 porque es la del entrenamiento.
Un `.keras` de Keras 3 no carga con versiones anteriores y el error no menciona
la versión.

---

## 7. Variables de entorno

Ningún valor sensible está en el repositorio. `src/config.py` los lee del
entorno:

| Variable | Para qué |
|---|---|
| `EVISOR_DB_URL` | Conexión a la base de datos del proyecto |
| `EVISOR_FUENTE_URL` | Conexión de solo lectura a la base de origen (CrateDB) |
| `EVISOR_FUENTE_TABLA` | Nombre de la tabla de origen, si no es `mediciones` |
| `EVISOR_PROVEEDOR_URL`, `EVISOR_PROVEEDOR_TOKEN` | Acceso por API, si llegara a existir |
| `EVISOR_MODELOS` | Carpeta de artefactos. Por omisión `modelos/v4` |
| `EVISOR_SMTP_*`, `EVISOR_ALERTAS_A` | Envío de alertas |

---

## 8. Decisiones de diseño que conviene conocer antes de modificar

- **La aplicación no carga el modelo.** Solo lee los archivos que produjo la
  inferencia. Esto evita depender de TensorFlow en el entorno de publicación y
  mantiene el consumo de memoria dentro del límite del servicio.
- **Entrenamiento e inferencia comparten `evisor_lstm.py`.** Si se cambia la
  forma de construir las entradas, cambia en los dos lados a la vez. No
  duplicar esa lógica.
- **Un solo modelo para los 16 bloques**, con la identidad del bloque como
  entrada y escalado propio por bloque. Resultó más preciso que un modelo por
  bloque (24,48 % contra 27,13 %) y es un solo artefacto que mantener.
- **Agrupamiento y pronóstico son independientes.** No reconectarlos: se
  comprobó que encadenarlos no aporta y propaga errores.
- **El agrupamiento se hace por variable, no con todas a la vez.** Mezclarlas
  produce un mapa dominado por la variable de valores más extremos.
- **El relleno de vacíos no usa el parámetro `limit` de pandas.** Ese parámetro
  rellena el borde de los vacíos largos en vez de descartarlos. Ver el
  comentario en `preprocesamiento.interpolar_huecos_cortos` y las pruebas que
  lo cubren.
- **La ingesta vuelve a leer 72 horas en cada corrida**, no solo lo nuevo. El
  cálculo de los incrementos de energía necesita contexto anterior.

## 9. Limitaciones conocidas

- La extracción de datos es manual.
- La plataforma de adquisición presentó **cinco interrupciones de más de 12
  horas que afectaron al campus completo en siete meses**, aproximadamente una
  por mes. Dos de ellas, el 13 y el 20 de agosto, ocurrieron el mismo día de la
  semana y en el mismo horario, lo que sugiere un proceso programado cuyo
  origen no está identificado.
- `evisor_lstm.tabla_bloque` todavía usa `interpolate(limit=...)`, con el mismo
  defecto que se corrigió en `preprocesamiento`. Su efecto es menor (una hora
  en lugar de doce) pero conviene corregirlo y volver a medir las métricas.
- Los medidores de los bloques **B8_CPA** y **B8_AA** presentan problemas de
  registro confirmados. Por decisión del equipo, **B8_CPA queda excluido de los
  análisis** hasta que su medidor sea intervenido.
- El bloque **B15_BIBL** empezó a registrar el 2 de marzo de 2026, veinte días
  después que los demás.
- `src/features.py` pertenece a la arquitectura anterior y solo lo usa
  `promover.py`. Rehacer esa compuerta es trabajo pendiente.
- El repositorio versiona un CSV de 43 MB. Cada actualización guarda una copia
  completa en el historial; conviene pasarlo a formato comprimido.

## 10. Autoría

**Daniela Gerena Lopera** — Ingeniería en Ciencia de Datos, UPB
Docente líder: Ana Isabel Oviedo
Proyecto desarrollado en el UPB Smart Energy Center (SEC).
