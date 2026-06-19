from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from esco_matcher import EscoMatcher  # noqa: E402
from esco_matcher.data import load_enriched_json, load_esco_csv  # noqa: E402
from esco_matcher.text import highlighted_html, tokenize  # noqa: E402


class EscoMatcherTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.csv_path = ROOT / "data" / "data_oc.csv"
        cls.json_path = ROOT / "data" / "codes_enriched.json"
        cls.matcher = EscoMatcher(cls.json_path)

    def test_loads_windows_encoded_csv_and_preserves_accents(self) -> None:
        frame = load_esco_csv(self.csv_path)
        self.assertEqual(len(frame), 3039)
        self.assertIn("Ejército", frame.iloc[0]["preferredLabel_esp"])

    def test_loads_enriched_json_and_joins_codes(self) -> None:
        frame = load_enriched_json(self.json_path)
        self.assertEqual(len(frame), 3039)
        row = frame.loc[frame["code"].eq("7231.10")].iloc[0]
        self.assertEqual(row["ciuo08_code"], "7231")
        self.assertIn("ESCO:7231.10", row["joined_codes"])
        self.assertIn("CIUO08:7231", row["joined_codes"])
        self.assertIn("CIUO08_CL:", row["joined_codes"])

    def test_match_result_exposes_joined_codes(self) -> None:
        result = self.matcher.search(
            "Diagnostica, mantiene y repara vehículos y motores.", top_k=5
        )[0]
        self.assertTrue(result.joined_codes.startswith("ESCO:"))
        self.assertTrue(result.ciuo08_cl_path)

    def test_electrician_description_returns_relevant_result(self) -> None:
        results = self.matcher.search(
            "Instalo sistemas eléctricos, reparo averías y leo planos de edificios.",
            top_k=10,
        )
        labels = " ".join(result.label.lower() for result in results)
        self.assertIn("electric", labels)

    def test_query_requires_informative_content(self) -> None:
        with self.assertRaises(ValueError):
            self.matcher.search("soy yo")

    def test_highlight_escapes_untrusted_html(self) -> None:
        rendered = highlighted_html(
            "<script>electricidad</script>", set(tokenize("electricidad"))
        )
        self.assertNotIn("<script>", rendered)
        self.assertIn("<mark>electricidad</mark>", rendered)

    def test_tokenizer_is_accent_insensitive(self) -> None:
        self.assertEqual(tokenize("Instalación eléctrica"), ["instal", "electric"])

    def test_tokenizer_reduces_common_inflections(self) -> None:
        self.assertEqual(tokenize("cuido cuidados cuidar"), ["cuid", "cuid", "cuid"])
        self.assertEqual(tokenize("soldador soldadura soldar"), ["sold", "sold", "sold"])

    def test_security_description_retrieves_security_candidates(self) -> None:
        results = self.matcher.search(
            "Protege personas, instalaciones y bienes mediante rondas de vigilancia, "
            "control de accesos y monitoreo de cámaras. También actúa ante situaciones "
            "de emergencia siguiendo protocolos establecidos.",
            top_k=10,
        )
        self.assertIn("5414.1", [result.code for result in results])

    def test_receptionist_description_retrieves_receptionist_candidates(self) -> None:
        results = self.matcher.search(
            "Atiende a visitantes y clientes, responde llamadas telefónicas, coordina "
            "citas y proporciona información. Es la primera persona de contacto en "
            "empresas, hoteles, clínicas y otras organizaciones.",
            top_k=10,
        )
        self.assertIn("4226.1", [result.code for result in results])

    def test_welder_description_retrieves_welder_candidates(self) -> None:
        results = self.matcher.search(
            "Realiza la unión y reparación de piezas metálicas utilizando diferentes "
            "técnicas de soldadura. Trabaja en industrias, construcciones, talleres y "
            "proyectos de infraestructura, asegurando que las estructuras sean "
            "resistentes y seguras.",
            top_k=10,
        )
        self.assertIn("7212.3", [result.code for result in results])

    def test_automotive_description_retrieves_vehicle_mechanic_candidates(self) -> None:
        results = self.matcher.search(
            "Diagnostica, mantiene y repara vehículos. Se encarga de sistemas "
            "mecánicos, eléctricos y electrónicos para garantizar el correcto "
            "funcionamiento y la seguridad de los automóviles.",
            top_k=10,
        )
        self.assertIn("7231.10", [result.code for result in results])

    def test_semantic_rerank_can_promote_candidates_within_lexical_shortlist(self) -> None:
        query = (
            "Realiza la unión y reparación de piezas metálicas utilizando diferentes "
            "técnicas de soldadura."
        )
        query_tokens = tokenize(query)
        lexical = self.matcher._lexical_scores(query_tokens)
        lexical_top20 = np.argsort(lexical)[::-1][:20]
        promoted_index = int(lexical_top20[-1])
        semantic = np.zeros(len(self.matcher.frame))
        semantic[promoted_index] = 1.0
        original_semantic_scores = self.matcher._semantic_scores
        self.matcher._semantic_scores = lambda query: semantic
        try:
            results = self.matcher.search(query, top_k=1)
        finally:
            self.matcher._semantic_scores = original_semantic_scores
        self.assertEqual(results[0].code, self.matcher.frame.iloc[promoted_index]["code"])

    def test_semantic_rerank_is_restricted_to_lexical_candidates(self) -> None:
        query = (
            "Realiza la unión y reparación de piezas metálicas utilizando diferentes "
            "técnicas de soldadura."
        )
        query_tokens = tokenize(query)
        lexical = self.matcher._lexical_scores(query_tokens)
        lexical_top20 = set(np.argsort(lexical)[::-1][:20])
        outside_index = next(
            index
            for index, code in enumerate(self.matcher.frame["code"])
            if index not in lexical_top20 and code != "7212.3"
        )
        semantic = np.zeros(len(self.matcher.frame))
        semantic[outside_index] = 1.0
        original_semantic_scores = self.matcher._semantic_scores
        self.matcher._semantic_scores = lambda query: semantic
        try:
            results = self.matcher.search(query, top_k=5)
        finally:
            self.matcher._semantic_scores = original_semantic_scores
        self.assertNotIn(
            self.matcher.frame.iloc[outside_index]["code"],
            [result.code for result in results],
        )


if __name__ == "__main__":
    unittest.main()
