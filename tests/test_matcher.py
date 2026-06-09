from __future__ import annotations

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from esco_matcher import EscoMatcher  # noqa: E402
from esco_matcher.data import load_esco_csv  # noqa: E402
from esco_matcher.text import highlighted_html, tokenize  # noqa: E402


class EscoMatcherTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.csv_path = ROOT / "data" / "data_oc.csv"
        cls.matcher = EscoMatcher(cls.csv_path)

    def test_loads_windows_encoded_csv_and_preserves_accents(self) -> None:
        frame = load_esco_csv(self.csv_path)
        self.assertEqual(len(frame), 3039)
        self.assertIn("Ejército", frame.iloc[0]["preferredLabel_esp"])

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

    def test_security_description_prefers_security_guard(self) -> None:
        results = self.matcher.search(
            "Protege personas, instalaciones y bienes mediante rondas de vigilancia, "
            "control de accesos y monitoreo de cámaras. También actúa ante situaciones "
            "de emergencia siguiendo protocolos establecidos.",
            top_k=3,
        )
        self.assertEqual(results[0].code, "5414.1")

    def test_welder_description_prefers_general_welder(self) -> None:
        results = self.matcher.search(
            "Realiza la unión y reparación de piezas metálicas utilizando diferentes "
            "técnicas de soldadura. Trabaja en industrias, construcciones, talleres y "
            "proyectos de infraestructura, asegurando que las estructuras sean "
            "resistentes y seguras.",
            top_k=3,
        )
        self.assertEqual(results[0].code, "7212.3")

    def test_automotive_description_retrieves_general_vehicle_mechanic(self) -> None:
        results = self.matcher.search(
            "Diagnostica, mantiene y repara vehículos. Se encarga de sistemas "
            "mecánicos, eléctricos y electrónicos para garantizar el correcto "
            "funcionamiento y la seguridad de los automóviles.",
            top_k=5,
        )
        self.assertIn("7231.10", [result.code for result in results])


if __name__ == "__main__":
    unittest.main()
