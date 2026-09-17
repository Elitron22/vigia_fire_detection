"""Comprueba el inventario y la documentación mínima de los notebooks entregados."""

from pathlib import Path
import unittest

import nbformat

from tools.document_delivery_notebooks import GUIDE_TAG, GUIDES


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"


class DeliveryNotebookDocumentationTests(unittest.TestCase):
    def test_inventory_matches_documented_delivery(self):
        actual = {path.name for path in NOTEBOOK_DIR.glob("*.ipynb")}
        self.assertEqual(actual, set(GUIDES))
        self.assertNotIn("04_DFire_auditoria_escenas.ipynb", actual)

    def test_every_notebook_has_one_complete_guide_and_no_saved_errors(self):
        required_labels = (
            "Papel en el proyecto",
            "Cuándo abrirlo",
            "Comportamiento por defecto",
            "Entradas principales",
            "Salidas principales",
            "Continuación",
        )
        for path in sorted(NOTEBOOK_DIR.glob("*.ipynb")):
            with self.subTest(notebook=path.name):
                notebook = nbformat.read(path, as_version=4)
                nbformat.validate(notebook)
                guides = [
                    cell
                    for cell in notebook.cells
                    if GUIDE_TAG in cell.metadata.get("tags", [])
                ]
                self.assertEqual(len(guides), 1)
                for label in required_labels:
                    self.assertIn(label, guides[0].source)
                errors = [
                    output
                    for cell in notebook.cells
                    if cell.cell_type == "code"
                    for output in cell.get("outputs", [])
                    if output.get("output_type") == "error"
                ]
                self.assertFalse(errors)
                combined_source = "\n".join(cell.source for cell in notebook.cells)
                self.assertNotIn("see_notebook_04", combined_source)
                self.assertNotIn("notebook 04", combined_source.lower())


if __name__ == "__main__":
    unittest.main()
