# Clasificador de ocupaciones ESCO

Aplicación explicable para relacionar una descripción libre en español con las
ocupaciones de ESCO. Devuelve el nombre (`preferredLabel_esp`), código ESCO,
códigos relacionados CIUO-08/CIUO-08 CL, descripción y una frase de evidencia con
las palabras coincidentes resaltadas.

## Enfoque

Este problema es de **recuperación de información / clasificación semántica**, no
necesita un agente. La aplicación ofrece dos modos:

- **Léxico (offline):** BM25 sobre etiqueta preferida, etiquetas alternativas y
  descripción, más cobertura de términos de la consulta y evaluación por frases.
  Esto evita que el contexto secundario oculte una tarea profesional muy
  específica. Cada sinónimo se indexa por separado para que una ocupación con
  muchos nombres alternativos no sea penalizada. No requiere API ni descarga de
  modelos.
- **Semántico:** usa BM25 para generar una lista corta de candidatos y luego
  reordena los mejores 20 con embeddings multilingües usando
  `paraphrase-multilingual-MiniLM-L12-v2`. Esto ayuda cuando la descripción del
  usuario y la descripción ESCO expresan la misma ocupación con palabras
  distintas, sin dejar que candidatos semánticamente parecidos pero
  léxicamente débiles salten desde cualquier parte del catálogo.

Un LLM podría reranquear los primeros resultados más adelante, pero no conviene
usarlo como fuente de códigos: aumenta costo y variabilidad, y puede inventar
categorías. La lista cerrada del JSON enriquecido debe seguir siendo la fuente de
verdad.

## Fuente de datos

La aplicación usa `data/codes_enriched.json` como fuente principal. El JSON se
aplana internamente a una fila por ocupación ESCO y conserva como metadatos:

- Código ESCO.
- Código CIUO-08 internacional, cuando está disponible.
- Ruta jerárquica CIUO-08 CL.
- Etiquetas y descripciones ESCO en español e inglés.

El resultado muestra estos códigos unidos en un formato como:

```text
ESCO:7231.10 | CIUO08:7231 | CIUO08_CL:7 > 72 > 723 > 7231
```

El loader aún soporta `data/data_oc.csv` para compatibilidad, pero la interfaz y
la CLI usan el JSON enriquecido.

Nota de calidad de datos: la ruta `CIUO-08 CL` se muestra tal como viene en el
JSON enriquecido. Algunas glosas parentales pueden requerir auditoría antes de
usarse como etiquetas oficiales en reportes; por eso actualmente se usan como
metadatos de salida y no como señal principal del ranking.

## Requisitos

- Python 3.10 o superior.
- El archivo `data/codes_enriched.json`, incluido en este repositorio.

## Instalación y ejecución

Clonar el repositorio y entrar en su carpeta:

```bash
git clone https://github.com/dmanzanoa/occupation_categorization.git
cd occupation_categorization
```

Crear y activar un entorno virtual:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

En Windows PowerShell, la activación es:

```powershell
.\.venv\Scripts\Activate.ps1
```

Instalar las dependencias básicas e iniciar la interfaz:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
```

Streamlit mostrará en la terminal la dirección local, normalmente
`http://localhost:8501`.

### Modo semántico

El modo básico funciona completamente offline con BM25. Para habilitar la
recuperación híbrida con embeddings multilingües:

```bash
python -m pip install -r requirements-semantic.txt
streamlit run app.py
```

La primera activación del interruptor **Usar modelo semántico** descarga el modelo.
Después queda en la caché local.

### Error relacionado con torchvision

Este proyecto procesa texto y no necesita `torchvision`. Algunas versiones de
`transformers` comprueban módulos opcionales de visión durante la importación. Si
existe una instalación incompatible de `torchvision`, se puede eliminar y
reinstalar las dependencias semánticas:

```bash
python -m pip uninstall -y torchvision
python -m pip install --upgrade --force-reinstall -r requirements-semantic.txt
python -m pip check
```

Para evitar mezclar paquetes globales con el entorno del proyecto, ejecutar
Streamlit mediante el mismo intérprete:

```bash
python -m streamlit run app.py
```

## Interfaz de terminal

También se puede consultar el sistema sin Streamlit:

```bash
python cli.py "Instalo sistemas eléctricos y reparo averías en edificios"
python cli.py --semantic "Ayudo a pacientes con ejercicios de rehabilitación"
```

## Pruebas

```bash
python -m unittest discover -s tests -v
```

## Estructura

```text
.
├── app.py                    # Interfaz Streamlit
├── cli.py                    # Interfaz de terminal
├── data/codes_enriched.json  # ESCO + CIUO-08 + CIUO-08 CL
├── data/data_oc.csv          # Catálogo ESCO original, mantenido como referencia
├── src/esco_matcher/         # Carga, normalización y ranking
└── tests/                    # Pruebas de regresión
```

## Limitaciones

Este proyecto recupera candidatos para homologación ocupacional. Los puntajes
sirven para ordenar resultados, pero no son probabilidades calibradas. Cuando
varias ocupaciones sean similares, deben presentarse los primeros candidatos y
solicitar información adicional en lugar de asumir que el primer resultado es
definitivo.
