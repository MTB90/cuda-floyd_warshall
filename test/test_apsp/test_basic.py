"""
Simple tests for APSP
package unittest is used because it helps with test management
and also provided formatted results for tests.
"""
import os
from subprocess import run, PIPE
from unittest import TestCase
from pathlib import Path

from test_apsp.helpers import APSP
from test_apsp.helpers import execute_algorithm
from test_apsp.helpers import gen_graph_with_diagonal_zeros
from test_apsp.helpers import gen_empty_graph
from test_apsp.helpers import gen_k1_predecessors, gen_k1_graph
from test_apsp.helpers import gen_kn_predecessors


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

    @classmethod
    def tearDownClass(cls):
        cls.make_path = Path(os.path.dirname(os.path.abspath(__file__))) / '../..'
        cls.make_process = run(['make', 'clean'],
                               cwd=cls.make_path,
                               stdout=PIPE, stderr=PIPE)

    def setUp(self):
        self.exec_path = self.make_path / self.exec_name
        self.assertTrue(self.exec_path.exists(), f"Can't find executable {self.exec_name}")

    def test_GIVEN_source_code_WHEN_compiling_THEN_compile_success(self):
        self.assertEqual(self.make_process.returncode, 0)

    def test_GIVEN_source_code_WHEN_compiling_THEN_no_error_message(self):
        self.assertFalse(bool(self.make_process.stderr))

    def test_GIVEN_graph_empty_WHEN_naive_fw_THEN_return_result_empty(self):
        result, stderr = execute_algorithm(self.exec_path, APSP.NAIVE_FW, "0 0")
        self.assertEqual(stderr, '')
        self.assertEqual(result['graph'], [])
        self.assertEqual(result['predecessors'], [])

    def test_GIVEN_graph_k0_WHEN_naive_fw_THEN_return_k0_result_path(self):
        data, stderr = execute_algorithm(self.exec_path, APSP.NAIVE_FW, "100 0")
        self.assertEqual(stderr, '')
        self.assertListEqual(data['graph'], gen_graph_with_diagonal_zeros(100))
        self.assertListEqual(data['predecessors'], gen_empty_graph(100, -1))

    def test_GIVEN_graph_k1_WHEN_naive_fw_THEN_return_k1_result_path(self):
        input_graph = "100 100"
        for i in range(50):
            input_graph += f" {i*2} {i*2+1} 1"
            input_graph += f" {i*2+1} {i*2} 1"

        data, stderr = execute_algorithm(self.exec_path, APSP.NAIVE_FW, input_graph)
        self.assertEqual(stderr, '')
        self.assertListEqual(data['graph'], gen_k1_graph(100))
        self.assertListEqual(data['predecessors'], gen_k1_predecessors(100))

    def test_GIVEN_graph_kn_WHEN_naive_fw_THEN_return_kn_result_path(self):
        input_graph = "10 100"
        for i in range(10):
            for j in range(10):
                if j != i:
                    input_graph += f" {i} {j} 1"

        data, stderr = execute_algorithm(self.exec_path, APSP.NAIVE_FW, input_graph)
        self.assertEqual(stderr, '')
        self.assertListEqual(data['graph'], gen_graph_with_diagonal_zeros(10, 1))
        self.assertListEqual(data['predecessors'], gen_kn_predecessors(10))

    def test_GIVEN_graph_empty_WHEN_cuda_naive_fw_THEN_return_result_empty(self):
        result, stderr = execute_algorithm(self.exec_path, APSP.CUDA_NAIVE_FW, "0 0")
        self.assertEqual(stderr, '')
        self.assertEqual(result['graph'], [])
        self.assertEqual(result['predecessors'], [])

    def test_GIVEN_graph_k0_WHEN_cuda_naive_fw_THEN_return_k0_result_path(self):
        data, stderr = execute_algorithm(self.exec_path, APSP.CUDA_NAIVE_FW, "100 0")
        self.assertEqual(stderr, '')
        self.assertListEqual(data['graph'], gen_graph_with_diagonal_zeros(100))
        self.assertListEqual(data['predecessors'], gen_empty_graph(100, -1))

    def test_GIVEN_graph_k1_WHEN_cuda_naive_fw_THEN_return_k1_result_path(self):
        input_graph = "100 100"
        for i in range(50):
            input_graph += f" {i*2} {i*2+1} 1"
            input_graph += f" {i*2+1} {i*2} 1"

        data, stderr = execute_algorithm(self.exec_path, APSP.CUDA_NAIVE_FW, input_graph)
        self.assertEqual(stderr, '')
        self.assertListEqual(data['graph'], gen_k1_graph(100))
        self.assertListEqual(data['predecessors'], gen_k1_predecessors(100))

    def test_GIVEN_graph_kn_WHEN_cuda_naive_fw_THEN_return_kn_result_path(self):
        input_graph = "10 100"
        for i in range(10):
            for j in range(10):
                if j != i:
                    input_graph += f" {i} {j} 1"

        data, stderr = execute_algorithm(self.exec_path, APSP.CUDA_NAIVE_FW, input_graph)
        self.assertEqual(stderr, '')
        self.assertListEqual(data['graph'], gen_graph_with_diagonal_zeros(10, 1))
        self.assertListEqual(data['predecessors'], gen_kn_predecessors(10))

