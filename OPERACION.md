# Manual de operación de E-Visor

Este documento es para quien reciba el proyecto. Explica qué corre, cuándo,
cómo saber si funciona y qué hacer cuando algo falla. Si algo de aquí deja de
ser cierto, hay que actualizarlo: es el único documento que no puede quedar
desactualizado.

Última revisión: octubre de 2026.

---

## 1. Qué hace el sistema

Predice el consumo eléctrico de las próximas 24 horas de los 16 bloques del
Ecocampus, en pasos de 10 minutos, y publica el resultado en un tablero web.

Hoy hay un solo proceso automático:

| Proceso | Cuándo | Qué hace | Dónde corre |
|---|---|---|---|
| `src.inferir_v4` | diario, 11:00 UTC (6:00 a. m. Colombia) | predice 24 horas y publica los resultados | GitHub Actions |
| `tests/` | en cada push | corre las pruebas | GitHub Actions |
| `scripts/clustering.py` | a mano, al cerrar un período | recalcula los perfiles de consumo | máquina local |
| entrenamiento | a mano | reentrena el modelo | Colab con GPU |

Los flujos `ingesta.yml` y `vigilante.yml` existen pero **no tienen horario**:
se reactivan cuando haya conexión automática a la fuente de datos.

El tablero de Streamlit **no calcula nada**: solo lee los archivos de
resultados. Si el tablero se cae, las predicciones se siguen generando.

### El reloj simulado

Mientras la extracción de datos sea manual, el histórico termina el 31 de
agosto de 2026 y no hay datos nuevos que predecir. Para poder verificar el
circuito completo, el proceso diario opera sobre un **reloj simulado**:
`datos/reloj_simulado.txt` guarda el instante del histórico que el sistema
toma como presente, y cada ejecución lo adelanta un día.

La ventaja añadida es que cada predicción se puede comparar contra lo que
realmente ocurrió ese día, cosa imposible si se pronosticara más allá del
final de los datos.

Cuando exista la conexión real, se borra el paso del reloj en
`predicciones_diarias.yml` y la inferencia toma por omisión el último dato
disponible.

---

## 2. Cómo saber si está funcionando

En orden de rapidez:

1. Abrir https://evisor-prediccion.streamlit.app/ y mirar la fecha del último
   dato.
2. En GitHub, pestaña **Actions**: el flujo `Predicciones diarias` debe estar
   en verde y haber corrido hoy.
3. Revisar que `datos/predicciones.csv` tenga un commit reciente del usuario
   `e-visor-bot`.

Cuando el proceso corra contra la base de datos, estas tres consultas dicen lo
mismo:

```sql
-- que tan fresco esta el dato
SELECT MAX(ts) FROM mediciones;

-- cuantos bloques tienen pronostico reciente
SELECT COUNT(DISTINCT entity_id) FROM predicciones
WHERE generado_en > NOW() - INTERVAL '24 hours';

-- ultimas corridas de cada proceso
SELECT job, MAX(inicio) AS ultima, estado FROM ejecuciones GROUP BY job, estado;
```

---

## 3. Fallas comunes y qué hacer

**El flujo diario falla al cargar el modelo.**
Casi siempre es la versión de TensorFlow. El modelo se entrenó con 2.20.0 y un
archivo `.keras` de Keras 3 no carga con versiones anteriores; el error que da
no menciona la versión. Verificar que el entorno instaló
`requirements-inferencia.txt` o `requirements-servidor.txt` sin sustituciones.

**Un bloque no tiene pronóstico pero los demás sí.**
Es el comportamiento esperado cuando ese medidor dejó de reportar. El proceso
está hecho para que un bloque que falla no tumbe a los demás: imprime
`SIN PREDICCION <bloque>: <motivo>` y continúa.

**Un bloque aparece con calidad degradada.**
Significa que no había datos completos de la semana anterior y el modelo usó
una referencia más lejana, hasta cuatro semanas atrás, o el perfil del día
anterior. La predicción sirve, pero con menos confianza.

**El tablero muestra datos viejos.**
Revisar Actions. Si el flujo corrió en verde pero el tablero no cambió,
probablemente Streamlit Cloud no se redesplegó: forzar el redespliegue desde
su panel.

**Las predicciones salen planas o con nulos.**
`src/calidad.py` tiene una prueba de humo para eso. Una línea plana indica que
el modelo no está respondiendo, no que el consumo sea constante.

---

## 4. Poner el proceso a correr en un servidor

Es el camino previsto para dejar de depender de la extracción manual.

1. Clonar el repositorio en el servidor.
2. Instalar dependencias: `pip install -r requirements-servidor.txt`.
   Requiere Python 3.11 o superior.
3. Definir las variables de entorno:
   - `EVISOR_DB_URL`: base donde viven las tablas del proyecto.
   - `EVISOR_FUENTE_URL`: conexión de solo lectura a la base de origen.
     CrateDB habla el protocolo de PostgreSQL, así que sirve el mismo driver.
   - `EVISOR_FUENTE_TABLA`: solo si la tabla de origen no se llama
     `mediciones`.
4. Crear el esquema y cargar el histórico una vez:

```bash
python -m src.cargar_historico --csv datos/datos_preprocesados_10min.csv
```

5. Programar dos tareas diarias, la ingesta antes que la predicción:

```bash
python -m src.ingesta --fuente sql
python -m src.inferir_v4 --artefactos modelos/v4 --desde-bd --guardar-bd \
       --salida datos/predicciones.csv --salida-historia datos/historia_reciente.csv
```

6. **Apagar la programación de `predicciones_diarias.yml`** en GitHub Actions.
   Si no, dos procesos generarían la predicción del mismo día.

### A qué hora programarlo

La hora depende de con cuánto retraso llega el dato a la base, que es un
número que hay que medir, no suponer. El modelo lee las últimas 48 horas, de
modo que si el borde de esa ventana viene incompleto se degrada justo en la
parte que más pesa.

Si el dato llega con pocos minutos de retraso, correr a las 00:00 hace que el
horizonte coincida exactamente con el día calendario. Si llega con media hora
o más, conviene correr a las 00:30 o a la 1:00: se pierde una hora de
anticipación y se gana una ventana de entrada completa.

---

## 5. Reentrenamiento

El modelo actual es un único artefacto para los 16 bloques, entrenado en Colab
con GPU desde `notebooks/E_Visor_v4_entrenamiento.ipynb`. No hay
reentrenamiento automático y **no conviene añadirlo en esta fase**: sin
detección de anomalías en operación, un reentrenamiento automático puede
aprender un período defectuoso sin que nadie lo note.

Cuándo vale la pena reentrenar:

- Cuando haya un semestre completo de datos nuevos.
- Cuando el error medido sobre un mes reciente se aparte de forma sostenida de
  lo que reporta el informe.
- Cuando se corrija un defecto del preprocesamiento que cambie la rejilla.

El procedimiento está documentado en el informe metodológico, sección 4. Tras
entrenar, medir siempre con `scripts/evaluar_por_bloque.py` usando
`modelo_global` sobre un mes que el modelo no haya visto, y comparar contra la
regla de referencia antes de reemplazar nada.

---

## 6. Secretos y dónde viven

Ninguno está en el repositorio. Todos deberían estar además en el gestor de
contraseñas institucional, con al menos dos personas que puedan recuperarlos.

| Secreto | Para qué | Dónde se configura |
|---|---|---|
| `EVISOR_DB_URL` | base del proyecto | GitHub Secrets, servidor |
| `EVISOR_FUENTE_URL` | base de origen, solo lectura | servidor |
| `EVISOR_SMTP_HOST/USER/PASS` | envío de correo | GitHub Secrets |
| `EVISOR_ALERTAS_A` | destinatarios de avisos | GitHub Secrets |

---

## 7. Responsables

| Rol | Persona | Contacto |
|---|---|---|
| Responsable técnico | (por definir) | |
| Relevo | (por definir) | |
| Contacto en Sistemas | (por definir) | |
| Contacto en el SEC | (por definir) | |

Estas cuatro casillas vacías son el mayor riesgo del proyecto. Llenarlas es
más importante que cualquier mejora del modelo.

---

## 8. Decisiones tomadas y por qué

- **Un solo modelo para los 16 bloques, no uno por bloque.** Resultó más
  preciso (24,48 % contra 27,13 %) y es un solo artefacto que mantener.
- **La aplicación no carga el modelo.** Solo lee los archivos de resultados.
  Así arranca en segundos y no se acerca al límite de memoria del servicio.
- **El error se reporta como WAPE, no como MAPE.** El MAPE de la potencia
  activa se dispara de madrugada, cuando el consumo se acerca a cero.
- **La energía se evalúa sobre el consumo por intervalo, no sobre el contador
  acumulado.** Un error del 0,2 % sobre un contador de 45 millones no dice nada
  sobre la calidad del pronóstico.
- **Las cifras agregadas excluyen B8_CPA.** Su medidor tiene problemas
  confirmados y sus registros no son comparables con los del resto.
- **La ingesta vuelve a leer 72 horas en cada corrida.** No es solo margen por
  datos que lleguen tarde: el cálculo de los incrementos de energía necesita
  contexto anterior para detectar los reinicios del contador.
- **Los vacíos largos no se rellenan.** El relleno de huecos evita a propósito
  el parámetro `limit` de pandas, que rellena el borde de los vacíos largos en
  lugar de descartarlos. Hay pruebas que lo cubren; no cambiarlo sin leerlas.
