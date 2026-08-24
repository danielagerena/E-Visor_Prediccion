-- Esquema de la base de datos de E-Visor.
-- Cinco tablas sostienen todo el sistema. Funciona igual en PostgreSQL y SQLite.

-- Rejilla de 10 minutos ya limpia. Es la unica fuente de verdad de mediciones.
-- No se guarda el crudo de 30 segundos: ocupa 15 veces mas y no aporta al modelo.
CREATE TABLE IF NOT EXISTS mediciones (
  entity_id                     TEXT NOT NULL,
  ts                            TIMESTAMP NOT NULL,
  activepower                   DOUBLE PRECISION,
  totalpowerfactor              DOUBLE PRECISION,
  voltaje_promedio              DOUBLE PRECISION,
  activeenergyimport_absoluto   DOUBLE PRECISION,
  activeenergyimport_incremento DOUBLE PRECISION,
  ingerido_en                   TIMESTAMP,
  PRIMARY KEY (entity_id, ts)
);
CREATE INDEX IF NOT EXISTS idx_mediciones_ts ON mediciones (ts);

-- Pronosticos generados. Formato ancho: una fila por instante predicho.
-- 144 pasos x 16 bloques = 2304 filas por corrida.
CREATE TABLE IF NOT EXISTS predicciones (
  entity_id                   TEXT NOT NULL,
  ts_origen                   TIMESTAMP NOT NULL,
  ts_objetivo                 TIMESTAMP NOT NULL,
  activepower                 DOUBLE PRECISION,
  totalpowerfactor            DOUBLE PRECISION,
  voltaje_promedio            DOUBLE PRECISION,
  activeenergyimport_absoluto DOUBLE PRECISION,
  modelo_version              TEXT NOT NULL,
  generado_en                 TIMESTAMP,
  PRIMARY KEY (entity_id, ts_origen, ts_objetivo)
);
CREATE INDEX IF NOT EXISTS idx_pred_objetivo ON predicciones (entity_id, ts_objetivo);

-- Registro de modelos. El puntero a produccion vive aqui, no en el codigo.
CREATE TABLE IF NOT EXISTS modelos (
  version      TEXT NOT NULL,
  bloque       TEXT NOT NULL,
  estado       TEXT NOT NULL,          -- produccion | archivada | rechazada
  url          TEXT,                   -- release de GitHub
  metricas     TEXT,                   -- JSON
  datos_desde  TIMESTAMP,
  datos_hasta  TIMESTAMP,
  commit_sha   TEXT,
  creado_en    TIMESTAMP,
  PRIMARY KEY (version, bloque)
);

-- Reglas de alerta configurables. Cambiar un umbral es un UPDATE, no un despliegue.
CREATE TABLE IF NOT EXISTS reglas_alerta (
  id                   INTEGER PRIMARY KEY,
  nombre               TEXT NOT NULL,
  entity_id            TEXT,            -- nulo = aplica a todos los bloques
  variable             TEXT NOT NULL,
  operador             TEXT NOT NULL,   -- '>' | '<'
  umbral               DOUBLE PRECISION NOT NULL,
  umbral_salida        DOUBLE PRECISION,-- histeresis: por donde se apaga
  pasos_consecutivos   INTEGER DEFAULT 1,
  potencia_minima      DOUBLE PRECISION DEFAULT 0,  -- ignora el bloque si esta casi apagado
  severidad            TEXT DEFAULT 'media',
  destinatarios        TEXT,            -- correos separados por coma
  referencia           TEXT,            -- norma o justificacion del umbral
  activa               INTEGER DEFAULT 1,
  modo_sombra          INTEGER DEFAULT 1  -- 1 = registra pero no notifica
);

-- Historial de eventos. Permite responder "cuantas alertas fueron falsas".
CREATE TABLE IF NOT EXISTS eventos_alerta (
  id             INTEGER PRIMARY KEY,
  regla_id       INTEGER NOT NULL,
  entity_id      TEXT NOT NULL,
  iniciado_en    TIMESTAMP NOT NULL,
  ts_prevista    TIMESTAMP,             -- cuando ocurriria segun el pronostico
  valor_extremo  DOUBLE PRECISION,
  estado         TEXT DEFAULT 'activa', -- activa | reconocida | resuelta
  ultimo_aviso   TIMESTAMP,
  resuelto_en    TIMESTAMP,
  reconocido_por TEXT
);
CREATE INDEX IF NOT EXISTS idx_eventos_estado ON eventos_alerta (estado, regla_id, entity_id);

-- Auditoria de cada corrida de los jobs. Es lo que revisa el vigilante.
CREATE TABLE IF NOT EXISTS ejecuciones (
  id          INTEGER PRIMARY KEY,
  job         TEXT NOT NULL,            -- ingesta | inferencia | alertas | reentrenamiento
  inicio      TIMESTAMP,
  fin         TIMESTAMP,
  estado      TEXT,                     -- ok | error
  filas       INTEGER,
  detalle     TEXT
);
CREATE INDEX IF NOT EXISTS idx_ejec_job ON ejecuciones (job, inicio);
