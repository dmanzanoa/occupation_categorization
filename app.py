from __future__ import annotations

from pathlib import Path
import sys

import streamlit as st


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from esco_matcher import EscoMatcher  # noqa: E402


st.set_page_config(page_title="Clasificador de ocupaciones ESCO")


@st.cache_resource(show_spinner="Construyendo el índice de ocupaciones...")
def get_matcher(use_semantic: bool) -> EscoMatcher:
    return EscoMatcher(
        ROOT / "data" / "codes_enriched.json", use_semantic=use_semantic
    )


st.title("Clasificador de ocupaciones ESCO")
st.write(
    "Describe libremente tu trabajo en español. El sistema devolverá las "
    "ocupaciones ESCO más cercanas, sus códigos relacionados CIUO/Chile y "
    "una frase de evidencia para revisarlas."
)

with st.sidebar:
    st.header("Configuración")
    use_semantic = st.toggle(
        "Usar modelo semántico",
        value=False,
        help=(
            "Mejora las coincidencias conceptuales. La primera ejecución puede "
            "descargar el modelo multilingual MiniLM."
        ),
    )
    top_k = st.slider("Número de resultados", min_value=1, max_value=10, value=5)
    st.caption(
        "El modo básico funciona sin API ni LLM. El modo semántico usa embeddings "
        "locales y tampoco envía las descripciones a un servicio externo."
    )

query = st.text_area(
    "Descripción de la ocupación",
    height=150,
    placeholder=(
        "Ejemplo: Instalo y mantengo sistemas eléctricos en viviendas y edificios, "
        "diagnostico averías y leo planos técnicos."
    ),
)

if st.button("Buscar ocupación", type="primary", use_container_width=True):
    try:
        matcher = get_matcher(use_semantic)
        if use_semantic and not matcher.semantic_enabled:
            st.warning(
                "No se pudo cargar el modelo semántico. Se muestran resultados "
                f"léxicos. Detalle: {matcher.semantic_error}"
            )
        results = matcher.search(query, top_k=top_k)
    except (ValueError, FileNotFoundError) as error:
        st.error(str(error))
    else:
        mode = "híbrida (semántica + léxica)" if matcher.semantic_enabled else "léxica"
        st.caption(f"Estrategia usada: {mode}. Los puntajes sirven para ordenar.")
        for position, result in enumerate(results, start=1):
            with st.container(border=True):
                st.subheader(f"{position}. {result.label}")
                left, right = st.columns(2)
                left.metric("Código ESCO", result.code)
                right.metric("Puntaje de similitud", f"{result.score:.1%}")
                if result.ciuo08_code:
                    st.markdown(
                        f"**CIUO-08:** `{result.ciuo08_code}` - {result.ciuo08_label}"
                    )
                if result.ciuo08_cl_path:
                    st.markdown(f"**CIUO-08 CL:** {result.ciuo08_cl_path}")
                st.markdown("**Evidencia en la descripción ESCO**")
                st.markdown(result.evidence_html, unsafe_allow_html=True)
                with st.expander("Ver detalles"):
                    st.write(result.description)
                    st.write("Códigos unidos:", result.joined_codes)
                    st.write(f"Puntaje léxico: {result.lexical_score:.3f}")
                    if result.semantic_score is not None:
                        st.write(f"Puntaje semántico: {result.semantic_score:.3f}")
                    if result.concept_uri:
                        st.link_button("Abrir concepto ESCO", result.concept_uri)
