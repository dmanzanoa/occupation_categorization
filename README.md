# Clasificador de ocupaciones ESCO

Aplicación explicable para relacionar una descripción libre en español con las
ocupaciones de ESCO. Devuelve el nombre (`preferredLabel_esp`), código, jerarquía,
descripción y una frase de evidencia con las palabras coincidentes resaltadas.

## Enfoque

Este problema es de **recuperación de información / clasificación semántica**, no
necesita un agente. La aplicación ofrece dos modos:

- **Léxico (offline):** BM25 sobre etiqueta preferida, etiquetas alternativas y
  descripción, más cobertura de términos de la consulta y evaluación por frases.
  Esto evita que el contexto secundario oculte una tarea profesional muy
  específica. Cada sinónimo se indexa por separado para que una ocupación con
  muchos nombres alternativos no sea penalizada. No requiere API ni descarga de
  modelos.
- **Híbrido:** combina BM25 (35 %) con embeddings multilingües (65 %) usando
  `paraphrase-multilingual-MiniLM-L12-v2`. Es mejor para expresiones con pocas
  palabras exactas en común.

Un LLM podría reranquear los primeros resultados más adelante, pero no conviene
usarlo como fuente de códigos ESCO: aumenta costo y variabilidad, y puede inventar
categorías. La lista cerrada del CSV debe seguir siendo la fuente de verdad.

## Requisitos

- Python 3.10 o superior.
- El archivo `data/data_oc.csv`, incluido en este repositorio.

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
├── data/data_oc.csv          # Catálogo de ocupaciones ESCO
├── src/esco_matcher/         # Carga, normalización y ranking
└── tests/                    # Pruebas de regresión
```

## Limitaciones

Este proyecto recupera candidatos para homologación ocupacional. Los puntajes
sirven para ordenar resultados, pero no son probabilidades calibradas. Cuando
varias ocupaciones sean similares, deben presentarse los primeros candidatos y
solicitar información adicional en lugar de asumir que el primer resultado es
definitivo.
