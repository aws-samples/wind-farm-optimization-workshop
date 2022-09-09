# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: agent.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x0b\x61gent.proto\x12\x12\x41WS.SageMaker.Edge\"^\n\x0eTensorMetadata\x12\x0c\n\x04name\x18\x01 \x01(\x0c\x12/\n\tdata_type\x18\x02 \x01(\x0e\x32\x1c.AWS.SageMaker.Edge.DataType\x12\r\n\x05shape\x18\x03 \x03(\x05\"F\n\x12SharedMemoryHandle\x12\x0c\n\x04size\x18\x01 \x01(\x04\x12\x0e\n\x06offset\x18\x02 \x01(\x04\x12\x12\n\nsegment_id\x18\x03 \x01(\x04\"\xaa\x01\n\x06Tensor\x12;\n\x0ftensor_metadata\x18\x01 \x01(\x0b\x32\".AWS.SageMaker.Edge.TensorMetadata\x12\x13\n\tbyte_data\x18\x04 \x01(\x0cH\x00\x12\x46\n\x14shared_memory_handle\x18\x05 \x01(\x0b\x32&.AWS.SageMaker.Edge.SharedMemoryHandleH\x00\x42\x06\n\x04\x64\x61ta\"\xab\x01\n\x05Model\x12\x0b\n\x03url\x18\x01 \x01(\t\x12\x0c\n\x04name\x18\x02 \x01(\t\x12\x42\n\x16input_tensor_metadatas\x18\x03 \x03(\x0b\x32\".AWS.SageMaker.Edge.TensorMetadata\x12\x43\n\x17output_tensor_metadatas\x18\x04 \x03(\x0b\x32\".AWS.SageMaker.Edge.TensorMetadata\"K\n\x0ePredictRequest\x12\x0c\n\x04name\x18\x01 \x01(\t\x12+\n\x07tensors\x18\x02 \x03(\x0b\x32\x1a.AWS.SageMaker.Edge.Tensor\">\n\x0fPredictResponse\x12+\n\x07tensors\x18\x01 \x03(\x0b\x32\x1a.AWS.SageMaker.Edge.Tensor\"-\n\x10LoadModelRequest\x12\x0b\n\x03url\x18\x01 \x01(\t\x12\x0c\n\x04name\x18\x02 \x01(\t\"=\n\x11LoadModelResponse\x12(\n\x05model\x18\x01 \x01(\x0b\x32\x19.AWS.SageMaker.Edge.Model\"\"\n\x12UnLoadModelRequest\x12\x0c\n\x04name\x18\x01 \x01(\t\"\x15\n\x13UnLoadModelResponse\"\x13\n\x11ListModelsRequest\"?\n\x12ListModelsResponse\x12)\n\x06models\x18\x01 \x03(\x0b\x32\x19.AWS.SageMaker.Edge.Model\"$\n\x14\x44\x65scribeModelRequest\x12\x0c\n\x04name\x18\x01 \x01(\t\"A\n\x15\x44\x65scribeModelResponse\x12(\n\x05model\x18\x01 \x01(\x0b\x32\x19.AWS.SageMaker.Edge.Model\"\xb1\x01\n\x0c\x41uxilaryData\x12\x0c\n\x04name\x18\x01 \x01(\t\x12.\n\x08\x65ncoding\x18\x02 \x01(\x0e\x32\x1c.AWS.SageMaker.Edge.Encoding\x12\x13\n\tbyte_data\x18\x03 \x01(\x0cH\x00\x12\x46\n\x14shared_memory_handle\x18\x04 \x01(\x0b\x32&.AWS.SageMaker.Edge.SharedMemoryHandleH\x00\x42\x06\n\x04\x64\x61ta\"\xc4\x02\n\x12\x43\x61ptureDataRequest\x12\x12\n\nmodel_name\x18\x01 \x01(\t\x12\x12\n\ncapture_id\x18\x02 \x01(\t\x12:\n\x13inference_timestamp\x18\x03 \x01(\x0b\x32\x1d.AWS.SageMaker.Edge.Timestamp\x12\x31\n\rinput_tensors\x18\x04 \x03(\x0b\x32\x1a.AWS.SageMaker.Edge.Tensor\x12\x32\n\x0eoutput_tensors\x18\x05 \x03(\x0b\x32\x1a.AWS.SageMaker.Edge.Tensor\x12\x30\n\x06inputs\x18\x06 \x03(\x0b\x32 .AWS.SageMaker.Edge.AuxilaryData\x12\x31\n\x07outputs\x18\x07 \x03(\x0b\x32 .AWS.SageMaker.Edge.AuxilaryData\"\x15\n\x13\x43\x61ptureDataResponse\"1\n\x1bGetCaptureDataStatusRequest\x12\x12\n\ncapture_id\x18\x01 \x01(\t\"U\n\x1cGetCaptureDataStatusResponse\x12\x35\n\x06status\x18\x01 \x01(\x0e\x32%.AWS.SageMaker.Edge.CaptureDataStatus\"+\n\tTimestamp\x12\x0f\n\x07seconds\x18\x01 \x01(\x03\x12\r\n\x05nanos\x18\x02 \x01(\x05*]\n\x08\x44\x61taType\x12\t\n\x05UINT8\x10\x00\x12\t\n\x05INT16\x10\x01\x12\t\n\x05INT32\x10\x02\x12\t\n\x05INT64\x10\x03\x12\x0b\n\x07\x46LOAT16\x10\x04\x12\x0b\n\x07\x46LOAT32\x10\x05\x12\x0b\n\x07\x46LOAT64\x10\x06*3\n\x08\x45ncoding\x12\x07\n\x03\x43SV\x10\x00\x12\x08\n\x04JSON\x10\x01\x12\x08\n\x04NONE\x10\x02\x12\n\n\x06\x42\x41SE64\x10\x03*M\n\x11\x43\x61ptureDataStatus\x12\x0b\n\x07\x46\x41ILURE\x10\x00\x12\x0b\n\x07SUCCESS\x10\x01\x12\x0f\n\x0bIN_PROGRESS\x10\x02\x12\r\n\tNOT_FOUND\x10\x03\x32\xb3\x05\n\x05\x41gent\x12R\n\x07Predict\x12\".AWS.SageMaker.Edge.PredictRequest\x1a#.AWS.SageMaker.Edge.PredictResponse\x12X\n\tLoadModel\x12$.AWS.SageMaker.Edge.LoadModelRequest\x1a%.AWS.SageMaker.Edge.LoadModelResponse\x12^\n\x0bUnLoadModel\x12&.AWS.SageMaker.Edge.UnLoadModelRequest\x1a\'.AWS.SageMaker.Edge.UnLoadModelResponse\x12[\n\nListModels\x12%.AWS.SageMaker.Edge.ListModelsRequest\x1a&.AWS.SageMaker.Edge.ListModelsResponse\x12\x64\n\rDescribeModel\x12(.AWS.SageMaker.Edge.DescribeModelRequest\x1a).AWS.SageMaker.Edge.DescribeModelResponse\x12^\n\x0b\x43\x61ptureData\x12&.AWS.SageMaker.Edge.CaptureDataRequest\x1a\'.AWS.SageMaker.Edge.CaptureDataResponse\x12y\n\x14GetCaptureDataStatus\x12/.AWS.SageMaker.Edge.GetCaptureDataStatusRequest\x1a\x30.AWS.SageMaker.Edge.GetCaptureDataStatusResponseb\x06proto3')

_DATATYPE = DESCRIPTOR.enum_types_by_name['DataType']
DataType = enum_type_wrapper.EnumTypeWrapper(_DATATYPE)
_ENCODING = DESCRIPTOR.enum_types_by_name['Encoding']
Encoding = enum_type_wrapper.EnumTypeWrapper(_ENCODING)
_CAPTUREDATASTATUS = DESCRIPTOR.enum_types_by_name['CaptureDataStatus']
CaptureDataStatus = enum_type_wrapper.EnumTypeWrapper(_CAPTUREDATASTATUS)
UINT8 = 0
INT16 = 1
INT32 = 2
INT64 = 3
FLOAT16 = 4
FLOAT32 = 5
FLOAT64 = 6
CSV = 0
JSON = 1
NONE = 2
BASE64 = 3
FAILURE = 0
SUCCESS = 1
IN_PROGRESS = 2
NOT_FOUND = 3


_TENSORMETADATA = DESCRIPTOR.message_types_by_name['TensorMetadata']
_SHAREDMEMORYHANDLE = DESCRIPTOR.message_types_by_name['SharedMemoryHandle']
_TENSOR = DESCRIPTOR.message_types_by_name['Tensor']
_MODEL = DESCRIPTOR.message_types_by_name['Model']
_PREDICTREQUEST = DESCRIPTOR.message_types_by_name['PredictRequest']
_PREDICTRESPONSE = DESCRIPTOR.message_types_by_name['PredictResponse']
_LOADMODELREQUEST = DESCRIPTOR.message_types_by_name['LoadModelRequest']
_LOADMODELRESPONSE = DESCRIPTOR.message_types_by_name['LoadModelResponse']
_UNLOADMODELREQUEST = DESCRIPTOR.message_types_by_name['UnLoadModelRequest']
_UNLOADMODELRESPONSE = DESCRIPTOR.message_types_by_name['UnLoadModelResponse']
_LISTMODELSREQUEST = DESCRIPTOR.message_types_by_name['ListModelsRequest']
_LISTMODELSRESPONSE = DESCRIPTOR.message_types_by_name['ListModelsResponse']
_DESCRIBEMODELREQUEST = DESCRIPTOR.message_types_by_name['DescribeModelRequest']
_DESCRIBEMODELRESPONSE = DESCRIPTOR.message_types_by_name['DescribeModelResponse']
_AUXILARYDATA = DESCRIPTOR.message_types_by_name['AuxilaryData']
_CAPTUREDATAREQUEST = DESCRIPTOR.message_types_by_name['CaptureDataRequest']
_CAPTUREDATARESPONSE = DESCRIPTOR.message_types_by_name['CaptureDataResponse']
_GETCAPTUREDATASTATUSREQUEST = DESCRIPTOR.message_types_by_name['GetCaptureDataStatusRequest']
_GETCAPTUREDATASTATUSRESPONSE = DESCRIPTOR.message_types_by_name['GetCaptureDataStatusResponse']
_TIMESTAMP = DESCRIPTOR.message_types_by_name['Timestamp']
TensorMetadata = _reflection.GeneratedProtocolMessageType('TensorMetadata', (_message.Message,), {
  'DESCRIPTOR' : _TENSORMETADATA,
  '__module__' : 'agent_pb2'
  # @@protoc_insertion_point(class_scope:AWS.SageMaker.Edge.TensorMetadata)
  })
_sym_db.RegisterMessage(TensorMetadata)

SharedMemoryHandle = _reflection.GeneratedProtocolMessageType('SharedMemoryHandle', (_message.Message,), {
  'DESCRIPTOR' : _SHAREDMEMORYHANDLE,
  '__module__' : 'agent_pb2'
  # @@protoc_insertion_point(class_scope:AWS.SageMaker.Edge.SharedMemoryHandle)
  })
_sym_db.RegisterMessage(SharedMemoryHandle)

Tensor = _reflection.GeneratedProtocolMessageType('Tensor', (_message.Message,), {
  'DESCRIPTOR' : _TENSOR,
  '__module__' : 'agent_pb2'
  # @@protoc_insertion_point(class_scope:AWS.SageMaker.Edge.Tensor)
  })
_sym_db.RegisterMessage(Tensor)

Model = _reflection.GeneratedProtocolMessageType('Model', (_message.Message,), {
  'DESCRIPTOR' : _MODEL,
  '__module__' : 'agent_pb2'
  # @@protoc_insertion_point(class_scope:AWS.SageMaker.Edge.Model)
  })
_sym_db.RegisterMessage(Model)

PredictRequest = _reflection.GeneratedProtocolMessageType('PredictRequest', (_message.Message,), {
  'DESCRIPTOR' : _PREDICTREQUEST,
  '__module__' : 'agent_pb2'
  # @@protoc_insertion_point(class_scope:AWS.SageMaker.Edge.PredictRequest)
  })
_sym_db.RegisterMessage(PredictRequest)

PredictResponse = _reflection.GeneratedProtocolMessageType('PredictResponse', (_message.Message,), {
  'DESCRIPTOR' : _PREDICTRESPONSE,
  '__module__' : 'agent_pb2'
  # @@protoc_insertion_point(class_scope:AWS.SageMaker.Edge.PredictResponse)
  })
_sym_db.RegisterMessage(PredictResponse)

LoadModelRequest = _reflection.GeneratedProtocolMessageType('LoadModelRequest', (_message.Message,), {
  'DESCRIPTOR' : _LOADMODELREQUEST,
  '__module__' : 'agent_pb2'
  # @@protoc_insertion_point(class_scope:AWS.SageMaker.Edge.LoadModelRequest)
  })
_sym_db.RegisterMessage(LoadModelRequest)

LoadModelResponse = _reflection.GeneratedProtocolMessageType('LoadModelResponse', (_message.Message,), {
  'DESCRIPTOR' : _LOADMODELRESPONSE,
  '__module__' : 'agent_pb2'
  # @@protoc_insertion_point(class_scope:AWS.SageMaker.Edge.LoadModelResponse)
  })
_sym_db.RegisterMessage(LoadModelResponse)

UnLoadModelRequest = _reflection.GeneratedProtocolMessageType('UnLoadModelRequest', (_message.Message,), {
  'DESCRIPTOR' : _UNLOADMODELREQUEST,
  '__module__' : 'agent_pb2'
  # @@protoc_insertion_point(class_scope:AWS.SageMaker.Edge.UnLoadModelRequest)
  })
_sym_db.RegisterMessage(UnLoadModelRequest)

UnLoadModelResponse = _reflection.GeneratedProtocolMessageType('UnLoadModelResponse', (_message.Message,), {
  'DESCRIPTOR' : _UNLOADMODELRESPONSE,
  '__module__' : 'agent_pb2'
  # @@protoc_insertion_point(class_scope:AWS.SageMaker.Edge.UnLoadModelResponse)
  })
_sym_db.RegisterMessage(UnLoadModelResponse)

ListModelsRequest = _reflection.GeneratedProtocolMessageType('ListModelsRequest', (_message.Message,), {
  'DESCRIPTOR' : _LISTMODELSREQUEST,
  '__module__' : 'agent_pb2'
  # @@protoc_insertion_point(class_scope:AWS.SageMaker.Edge.ListModelsRequest)
  })
_sym_db.RegisterMessage(ListModelsRequest)

ListModelsResponse = _reflection.GeneratedProtocolMessageType('ListModelsResponse', (_message.Message,), {
  'DESCRIPTOR' : _LISTMODELSRESPONSE,
  '__module__' : 'agent_pb2'
  # @@protoc_insertion_point(class_scope:AWS.SageMaker.Edge.ListModelsResponse)
  })
_sym_db.RegisterMessage(ListModelsResponse)

DescribeModelRequest = _reflection.GeneratedProtocolMessageType('DescribeModelRequest', (_message.Message,), {
  'DESCRIPTOR' : _DESCRIBEMODELREQUEST,
  '__module__' : 'agent_pb2'
  # @@protoc_insertion_point(class_scope:AWS.SageMaker.Edge.DescribeModelRequest)
  })
_sym_db.RegisterMessage(DescribeModelRequest)

DescribeModelResponse = _reflection.GeneratedProtocolMessageType('DescribeModelResponse', (_message.Message,), {
  'DESCRIPTOR' : _DESCRIBEMODELRESPONSE,
  '__module__' : 'agent_pb2'
  # @@protoc_insertion_point(class_scope:AWS.SageMaker.Edge.DescribeModelResponse)
  })
_sym_db.RegisterMessage(DescribeModelResponse)

AuxilaryData = _reflection.GeneratedProtocolMessageType('AuxilaryData', (_message.Message,), {
  'DESCRIPTOR' : _AUXILARYDATA,
  '__module__' : 'agent_pb2'
  # @@protoc_insertion_point(class_scope:AWS.SageMaker.Edge.AuxilaryData)
  })
_sym_db.RegisterMessage(AuxilaryData)

CaptureDataRequest = _reflection.GeneratedProtocolMessageType('CaptureDataRequest', (_message.Message,), {
  'DESCRIPTOR' : _CAPTUREDATAREQUEST,
  '__module__' : 'agent_pb2'
  # @@protoc_insertion_point(class_scope:AWS.SageMaker.Edge.CaptureDataRequest)
  })
_sym_db.RegisterMessage(CaptureDataRequest)

CaptureDataResponse = _reflection.GeneratedProtocolMessageType('CaptureDataResponse', (_message.Message,), {
  'DESCRIPTOR' : _CAPTUREDATARESPONSE,
  '__module__' : 'agent_pb2'
  # @@protoc_insertion_point(class_scope:AWS.SageMaker.Edge.CaptureDataResponse)
  })
_sym_db.RegisterMessage(CaptureDataResponse)

GetCaptureDataStatusRequest = _reflection.GeneratedProtocolMessageType('GetCaptureDataStatusRequest', (_message.Message,), {
  'DESCRIPTOR' : _GETCAPTUREDATASTATUSREQUEST,
  '__module__' : 'agent_pb2'
  # @@protoc_insertion_point(class_scope:AWS.SageMaker.Edge.GetCaptureDataStatusRequest)
  })
_sym_db.RegisterMessage(GetCaptureDataStatusRequest)

GetCaptureDataStatusResponse = _reflection.GeneratedProtocolMessageType('GetCaptureDataStatusResponse', (_message.Message,), {
  'DESCRIPTOR' : _GETCAPTUREDATASTATUSRESPONSE,
  '__module__' : 'agent_pb2'
  # @@protoc_insertion_point(class_scope:AWS.SageMaker.Edge.GetCaptureDataStatusResponse)
  })
_sym_db.RegisterMessage(GetCaptureDataStatusResponse)

Timestamp = _reflection.GeneratedProtocolMessageType('Timestamp', (_message.Message,), {
  'DESCRIPTOR' : _TIMESTAMP,
  '__module__' : 'agent_pb2'
  # @@protoc_insertion_point(class_scope:AWS.SageMaker.Edge.Timestamp)
  })
_sym_db.RegisterMessage(Timestamp)

_AGENT = DESCRIPTOR.services_by_name['Agent']
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _DATATYPE._serialized_start=1764
  _DATATYPE._serialized_end=1857
  _ENCODING._serialized_start=1859
  _ENCODING._serialized_end=1910
  _CAPTUREDATASTATUS._serialized_start=1912
  _CAPTUREDATASTATUS._serialized_end=1989
  _TENSORMETADATA._serialized_start=35
  _TENSORMETADATA._serialized_end=129
  _SHAREDMEMORYHANDLE._serialized_start=131
  _SHAREDMEMORYHANDLE._serialized_end=201
  _TENSOR._serialized_start=204
  _TENSOR._serialized_end=374
  _MODEL._serialized_start=377
  _MODEL._serialized_end=548
  _PREDICTREQUEST._serialized_start=550
  _PREDICTREQUEST._serialized_end=625
  _PREDICTRESPONSE._serialized_start=627
  _PREDICTRESPONSE._serialized_end=689
  _LOADMODELREQUEST._serialized_start=691
  _LOADMODELREQUEST._serialized_end=736
  _LOADMODELRESPONSE._serialized_start=738
  _LOADMODELRESPONSE._serialized_end=799
  _UNLOADMODELREQUEST._serialized_start=801
  _UNLOADMODELREQUEST._serialized_end=835
  _UNLOADMODELRESPONSE._serialized_start=837
  _UNLOADMODELRESPONSE._serialized_end=858
  _LISTMODELSREQUEST._serialized_start=860
  _LISTMODELSREQUEST._serialized_end=879
  _LISTMODELSRESPONSE._serialized_start=881
  _LISTMODELSRESPONSE._serialized_end=944
  _DESCRIBEMODELREQUEST._serialized_start=946
  _DESCRIBEMODELREQUEST._serialized_end=982
  _DESCRIBEMODELRESPONSE._serialized_start=984
  _DESCRIBEMODELRESPONSE._serialized_end=1049
  _AUXILARYDATA._serialized_start=1052
  _AUXILARYDATA._serialized_end=1229
  _CAPTUREDATAREQUEST._serialized_start=1232
  _CAPTUREDATAREQUEST._serialized_end=1556
  _CAPTUREDATARESPONSE._serialized_start=1558
  _CAPTUREDATARESPONSE._serialized_end=1579
  _GETCAPTUREDATASTATUSREQUEST._serialized_start=1581
  _GETCAPTUREDATASTATUSREQUEST._serialized_end=1630
  _GETCAPTUREDATASTATUSRESPONSE._serialized_start=1632
  _GETCAPTUREDATASTATUSRESPONSE._serialized_end=1717
  _TIMESTAMP._serialized_start=1719
  _TIMESTAMP._serialized_end=1762
  _AGENT._serialized_start=1992
  _AGENT._serialized_end=2683
# @@protoc_insertion_point(module_scope)
