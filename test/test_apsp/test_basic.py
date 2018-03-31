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
from test_apsp.helpers import gen_graph_out
from test_apsp.helpers import gen_k1_predecessors_out, gen_k1_graph_out, gen_k1_graph_in
from test_apsp.helpers import gen_kn_predecessors_out, gen_kn_graph_in
from test_apsp.helpers import gen_graph_dicircle_in, gen_kn_graph_for_dcircle_out, gen_kn_pred_for_dcircle_out


class TestBasic(TestCase):
    MAKE_PATH = None
    MAKE_PROCESS = None
    EXEC_NAME = 'cuda_floyd-warshall'
    SIZE = 1000

    @classmethod
    def setUpClass(cls):
        cls.MAKE_PATH = Path(os.path.dirname(os.path.abspath(__file__))) / '../..'
        cls.MAKE_PROCESS = run(['make', 'clean', 'all'],
                               cwd=cls.MAKE_PATH,
                               stdout=PIPE, stderr=PIPE)

    @classmethod
    def tearDownClass(cls):
        cls.MAKE_PATH = Path(os.path.dirname(os.path.abspath(__file__))) / '../..'
        cls.MAKE_PROCESS = run(['make', 'clean'],
                               cwd=cls.MAKE_PATH,
                               stdout=PIPE, stderr=PIPE)

    def setUp(self):
        self.exec_path = self.MAKE_PATH / self.EXEC_NAME
        self.assertTrue(self.exec_path.exists(), f"Can't find executable {self.EXEC_NAME}")
        self.algorithms = [APSP.NAIVE_FW, APSP.CUDA_NAIVE_FW, APSP.CUDA_BLOCKED_FW]

    def _compare_result_graph_from_two_algorithms(self, input_graph: str, first: str, second: str):
        results = {}
        for algorithm in [first, second]:
            results[algorithm], stderr = execute_algorithm(self.exec_path, algorithm, input_graph)
        self.assertListEqual(results[first]['graph'], results[second]['graph'])

    def test_GIVEN_source_code_WHEN_compiling_THEN_compile_success(self):
        self.assertEqual(self.MAKE_PROCESS.returncode, 0)

    def test_GIVEN_source_code_WHEN_compiling_THEN_no_error_message(self):
        self.assertFalse(bool(self.MAKE_PROCESS.stderr))

    def test_GIVEN_graph_empty_WHEN_fw_THEN_return_result_empty(self):
        for algorithm in self.algorithms:
            result, stderr = execute_algorithm(self.exec_path, algorithm, "0 0")
            self.assertEqual(stderr, '')
            self.assertEqual(result['graph'], [])
            self.assertEqual(result['predecessors'], [])

    def test_GIVEN_graph_k0_WHEN_fw_THEN_return_k0_result_path(self):
        for algorithm in self.algorithms:
            data, stderr = execute_algorithm(self.exec_path, algorithm, f"{self.SIZE} 0")
            self.assertEqual(stderr, '')
            self.assertListEqual(data['graph'], gen_graph_out(self.SIZE, diagonal=0))
            self.assertListEqual(data['predecessors'], gen_graph_out(self.SIZE, -1))

    def test_GIVEN_graph_k1_WHEN_fw_THEN_return_k1_result_path(self):
        for algorithm in self.algorithms:
            data, stderr = execute_algorithm(self.exec_path, algorithm, gen_k1_graph_in(self.SIZE))
            self.assertEqual(stderr, '')
            self.assertListEqual(data['graph'], gen_k1_graph_out(self.SIZE))
            self.assertListEqual(data['predecessors'], gen_k1_predecessors_out(self.SIZE))

    def test_GIVEN_graph_kn_WHEN_fw_THEN_return_kn_result_path(self):
        for algorithm in self.algorithms:
            data, stderr = execute_algorithm(self.exec_path, algorithm, gen_kn_graph_in(self.SIZE))
            self.assertEqual(stderr, '')
            self.assertListEqual(data['graph'], gen_graph_out(self.SIZE, 1, 0))
            self.assertListEqual(data['predecessors'], gen_kn_predecessors_out(self.SIZE))

    def test_GIVEN_graph_dicircle_WHEN_fw_THEN_return_kn_correct_result_path(self):
        for algorithm in self.algorithms:
            data, stderr = execute_algorithm(self.exec_path, algorithm, gen_graph_dicircle_in(self.SIZE))
            self.assertEqual(stderr, '')
            self.assertListEqual(data['graph'], gen_kn_graph_for_dcircle_out(self.SIZE))
            self.assertListEqual(data['predecessors'], gen_kn_pred_for_dcircle_out(self.SIZE))

    def test_GIVEN_all_small_graphs_WHEN_compare_navie_fw_with_cuda_naive_fw_THEN_results_path_are_the_same(self):
        inputs_small = Path(os.path.dirname(os.path.abspath(__file__))) / '../../input/small'
        for input_small in inputs_small.glob("*"):
            with input_small.open() as f:
                self._compare_result_graph_from_two_algorithms(f.read(), APSP.NAIVE_FW, APSP.CUDA_NAIVE_FW)

    def test_GIVEN_all_small_graphs_WHEN_compare_navie_fw_with_cuda_blocked_fw_THEN_results_path_are_the_same(self):
        inputs_small = Path(os.path.dirname(os.path.abspath(__file__))) / '../../input/small'
        for input_small in inputs_small.glob("*"):
            with input_small.open() as f:
                self._compare_result_graph_from_two_algorithms(f.read(), APSP.NAIVE_FW, APSP.CUDA_BLOCKED_FW)

    def test_GIVEN_one_big_graph_WHEN_compare_navie_fw_with_cuda_naive_fw_THEN_results_path_are_the_same(self):
        big_graph = Path(os.path.dirname(os.path.abspath(__file__))) / '../../input/big/V1000-E150000'
        with big_graph.open() as f:
            self._compare_result_graph_from_two_algorithms(f.read(), APSP.NAIVE_FW, APSP.CUDA_NAIVE_FW)

    def test_GIVEN_one_big_graph_WHEN_compare_navie_fw_with_cuda_blocked_fw_THEN_results_path_are_the_same(self):
        big_graph = Path(os.path.dirname(os.path.abspath(__file__))) / '../../input/big/V1000-E150000'
        with big_graph.open() as f:
            self._compare_result_graph_from_two_algorithms(f.read(), APSP.NAIVE_FW, APSP.CUDA_BLOCKED_FW)
