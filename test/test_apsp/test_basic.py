"""
Simple tests for APSP
package unittest is used because it helps with test management
and also provided formatted results for tests.
"""
import os
from subprocess import run, PIPE
from unittest import TestCase
from pathlib import Path


class TestBasic(TestCase):
    make_path = None
    make_process = None
    exec_name = 'cuda_floyd-warshall'

    @classmethod
    def setUpClass(cls):
        cls.make_path = Path(os.path.dirname(os.path.abspath(__file__))) / '../..'
        cls.make_process = run(['make', 'clean', 'all'],
                               cwd=cls.make_path,
                               stdout=PIPE, stderr=PIPE)

    def setUp(self):
        self.exec_path = self.make_path / self.exec_name

    def test_GIVEN_source_code_WHEN_compiling_THEN_compile_success(self):
        self.assertEqual(self.make_process.returncode, 0)

    def test_GIVEN_source_code_WHEN_compiling_THEN_exec_exist(self):
        self.assertTrue(self.exec_path.exists())

    def test_GIVEN_source_code_WHEN_compiling_THEN_no_error_message(self):
        self.assertEqual(self.make_process.stderr, '')
