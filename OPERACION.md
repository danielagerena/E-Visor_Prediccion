# Manual de operacion de E-Visor

Este documento es para quien reciba el proyecto. Explica que corre, cuando,
como saber si funciona y que hacer cuando algo falla. Si algo de aqui deja de
ser cierto, hay que actualizarlo: es el unico documento que no puede quedar
desactualizado.

---

## 1. Que hace el sistema

Predice el consumo electrico de las proximas 24 horas de 16 bloques del
Ecocampus, guarda esos pronosticos en una base de datos, los muestra en un
tablero y avisa cuando alguno cruza un umbral.

El sistema son tres relojes que giran a velocidades distintas:

| Job | Cuando | Que hace | Donde corre |
|---|---|---|---|
| `src.ingesta` | cada 15 min | trae mediciones nuevas y las guarda | GitHub Actions |
| `src.inferir` + `src.alertas` | cada hora | predice 24 h y evalua reglas | GitHub Actions |
| `src.vigilante` | diario 12:00 UTC | revisa salud y poda datos viejos | GitHub Actions |
| `src.entrenar` + `src.promover` | mensual o a mano | reentrena y decide si publicar | maquina con GPU |

El tablero de Streamlit **no calcula nada**: solo lee las tablas. Si el
tablero se cae, los pronosticos siguen generandose.

---

## 2. Como saber si esta funcionando

En orden de rapidez:

1. Abrir el tablero. Arriba dice "Datos actualizados hace X min". Si esta en
   verde, todo bien.
2. En GitHub, pestana **Actions**: los ultimos workflows deben estar en verde.
3. En la base de datos:

```sql
-- ultimas corridas de cada job
SELECT job, MAX(inicio) AS ultima, estado FROM ejecuciones GROUP BY job, estado;

-- que tan fresco esta el dato
SELECT MAX(ts) FROM mediciones;

-- cuantos bloques tienen pronostico reciente
SELECT COUNT(DISTINCT entity_id) FROM predicciones
WHERE generado_en > NOW() - INTERVAL '6 hours';
```

El job `vigilante` hace estas tres revisiones solo, todos los dias, y envia
correo si algo falla.

---

## 3. Fallas comunes y que hacer

**El tablero dice que los datos estan viejos.**
Revisar Actions: probablemente fallo la ingesta. La causa mas frecuente es que
expiro el token del proveedor. Se renueva en `Settings > Secrets > Actions`,
secreto `EVISOR_PROVEEDOR_TOKEN`.

**Un bloque no tiene pronostico pero los demas si.**
Normal si ese medidor dejo de reportar. `src.inferir` esta hecho para que un
bloque que falla no tumbe a los demas. Revisar el detalle en la tabla
`ejecuciones`.

**Llegaron muchas alertas de golpe.**
Poner la regla en modo sombra mientras se calibra, sin desplegar nada:

```sql
UPDATE reglas_alerta SET modo_sombra = 1 WHERE id = <id>;
```

**Un modelo nuevo empeoro las predicciones.**
Revertir al anterior. Es un cambio de puntero, no un despliegue:

```bash
python -m src.promover --bloque B10_ARQ --version v2026.08.01 --revertir
```

**La base se acerca al limite del plan gratuito.**
Bajar la retencion de pronosticos en `config.DIAS_RETENCION_PREDICCIONES` y
correr `python -m src.vigilante --podar`.

---

## 4. Reentrenamiento

Dos modalidades:

- **Refresco mensual (warm start).** Continua el entrenamiento desde los pesos
  actuales. Barato, 1 a 2 horas de GPU.
- **Completo trimestral.** Reentrena desde cero con validacion de origen
  deslizante. 10 a 25 horas de GPU.

```bash
python -m src.entrenar --bloque B10_ARQ --salida artefactos/B10_ARQ \
       --warm-start modelos_predictivos/B10_ARQ/modelo_blockrnn.pt

python -m src.promover --bloque B10_ARQ --version v2026.09.01 \
       --carpeta artefactos/B10_ARQ
```

`promover` aplica cuatro compuertas antes de publicar: datos validos,
entrenamiento terminado, desempeno no peor que el modelo actual sobre el mismo
conjunto de prueba, y prueba de humo cargando el artefacto en limpio. Si alguna
falla, **no se toca nada** y el modelo viejo sigue en produccion.

---

## 5. Secretos y donde viven

Ninguno esta en el repositorio. Todos deben estar ademas en el gestor de
contrasenas institucional, con al menos dos personas que puedan recuperarlos.

| Secreto | Para que | Donde se configura |
|---|---|---|
| `EVISOR_DB_URL` | conexion a la base | GitHub Secrets y Streamlit Secrets |
| `EVISOR_PROVEEDOR_URL` | endpoint del proveedor | GitHub Secrets |
| `EVISOR_PROVEEDOR_TOKEN` | credencial del proveedor | GitHub Secrets |
| `EVISOR_SMTP_HOST/USER/PASS` | envio de correo | GitHub Secrets |
| `EVISOR_ALERTAS_A` | destinatarios de avisos | GitHub Secrets |

---

## 6. Responsables

| Rol | Persona | Contacto |
|---|---|---|
| Responsable tecnico | (por definir) | |
| Relevo | (por definir) | |
| Contacto en Sistemas | (por definir) | |
| Contacto del proveedor | (por definir) | |

Estas cuatro casillas vacias son el mayor riesgo del proyecto. Llenarlas es
mas importante que cualquier mejora del modelo.

---

## 7. Decisiones tomadas y por que

- **La base de datos guarda la rejilla de 10 minutos, no el crudo de 30
  segundos.** El crudo ocupa quince veces mas y el modelo no lo usa. Los
  archivos originales se conservan aparte como respaldo.
- **La inferencia corre cada hora, no cada 15 minutos.** El horizonte es de 24
  horas: cuatro corridas por hora producen curvas casi identicas.
- **El error se reporta como WAPE, no como MAPE.** El MAPE de la potencia
  activa se dispara de madrugada, cuando el consumo se acerca a cero.
- **La energia se evalua sobre el consumo por intervalo, no sobre el contador
  acumulado.** Un error del 0,2 % sobre un contador de 45 millones no dice nada
  sobre la calidad del pronostico.
- **El modelo en produccion es un puntero en una tabla.** Cambiarlo o revertirlo
  no requiere desplegar codigo.
