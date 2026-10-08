"""Contract checks against the saved model; separate from reported test metrics."""
import unittest
from pathlib import Path
import torch
from .extractor import IDExtractor

class ExtractionContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.extractor=IDExtractor(Path(__file__).resolve().parents[1]/"artifacts/identifier_extraction")

    def test_requested_cases(self):
        examples=[
            ('Show overpayment details for case ID 12345', {'case_id':'12345','employee_id':'','sr_no':''}),
            ('Show overpayment details for employee ID 12345', {'case_id':'','employee_id':'12345','sr_no':''}),
            ('Show overpayment details for OP SR 12345', {'case_id':'','employee_id':'','sr_no':'12345'}),
            ('Show overpayment details', {'case_id':'','employee_id':'','sr_no':''}),
            ('Get me the recent letter generated for the employee 009876', {'case_id':'','employee_id':'009876','sr_no':''}),
            ('Show case ID 00123 and employee ID 00456 and SR number 00789', {'case_id':'00123','employee_id':'00456','sr_no':'00789'}),
            ('There are 12 records', {'case_id':'','employee_id':'','sr_no':''}),
            ('', {'case_id':'','employee_id':'','sr_no':''}),
        ]
        for query, expected in examples:
            with self.subTest(query=query): self.assertEqual(self.extractor.predict(query),expected)

    def test_unknown_id_values(self):
        for value in ['00001','876543210123','7']:
            self.assertEqual(self.extractor.predict('Show case ID '+value)['case_id'],value)

    def test_invalid_input(self):
        with self.assertRaises(TypeError): self.extractor.predict(None)
        with self.assertRaises(ValueError): self.extractor.predict('word '*513)

if __name__=='__main__': unittest.main()
