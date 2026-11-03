from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy
import onnx
import pytest
from onnx import helper, numpy_helper
from onnxruntime import InferenceSession

from facefusion.common_helper import is_macos
from facefusion.execution import create_inference_providers, get_available_execution_providers, has_execution_provider


def test_has_execution_provider() -> None:
	assert has_execution_provider('cpu') is True
	assert has_execution_provider('openvino') is False


def test_get_available_execution_providers() -> None:
	assert 'cpu' in get_available_execution_providers()


def test_coreml_is_preferred_when_available() -> None:
	with patch('facefusion.execution.onnxruntime.get_available_providers', return_value = [ 'CPUExecutionProvider', 'CoreMLExecutionProvider' ]):
		assert get_available_execution_providers() == [ 'coreml', 'cpu' ]


@pytest.mark.skipif(not is_macos(), reason = 'CoreML is only available on macOS')
def test_coreml_provider_is_available_on_macos() -> None:
	assert has_execution_provider('coreml') is True


@pytest.mark.skipif(not is_macos(), reason = 'CoreML is only available on macOS')
def test_coreml_can_run_inference_on_macos() -> None:
	input_info = helper.make_tensor_value_info('input', onnx.TensorProto.FLOAT, [ 1, 2 ])
	output_info = helper.make_tensor_value_info('output', onnx.TensorProto.FLOAT, [ 1, 2 ])
	bias = numpy_helper.from_array(numpy.array([ [ 1.0, 2.0 ] ], dtype = numpy.float32), 'bias')
	node = helper.make_node('Add', [ 'input', 'bias' ], [ 'output' ])
	graph = helper.make_graph([ node ], 'coreml-smoke', [ input_info ], [ output_info ], [ bias ])
	model = helper.make_model(graph, opset_imports = [ helper.make_opsetid('', 13) ])
	model.ir_version = 8

	with TemporaryDirectory() as cache_directory:
		with patch('facefusion.execution.resolve_cache_path', return_value = cache_directory):
			inference_providers = create_inference_providers(0, [ 'coreml' ])
		inference_session = InferenceSession(model.SerializeToString(), providers = inference_providers)
		output = inference_session.run(None, { 'input': numpy.array([ [ 1.0, 1.0 ] ], dtype = numpy.float32) })[0]

	assert inference_session.get_providers()[0] == 'CoreMLExecutionProvider'
	numpy.testing.assert_allclose(output, [ [ 2.0, 3.0 ] ])


def test_create_inference_providers() -> None:
	inference_providers =\
	[
		('CUDAExecutionProvider',
		{
			'device_id': 1,
			'cudnn_conv_algo_search': 'EXHAUSTIVE'
		}),
		'CPUExecutionProvider'
	]

	assert create_inference_providers(1, [ 'cpu', 'cuda' ]) == inference_providers
