# Copyright 2020 The FedLearner Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# coding: utf-8
# pylint: disable=protected-access

import logging
import time
from collections import defaultdict

import google.protobuf.any_pb2
import grpc
import tensorflow.compat.v1 as tf

from fedlearner.common import common_pb2 as common_pb
from fedlearner.common import metrics
from fedlearner.common import trainer_worker_service2_pb2 as tws2_pb
from fedlearner.common import trainer_worker_service2_pb2_grpc as tws2_grpc
from fedlearner.proxy.bridge import Bridge, BridgeClientManager
from fedlearner.proxy.channel import make_insecure_channel, ChannelType


class TrainerBridge(Bridge):
    class TrainerServicer(tws2_grpc.TrainerWorkerServiceServicer):
        def __init__(self, bridge):
            super().__init__()
            self._bridge = bridge

        def LoadDataBlock(self, request, context):
            return self._bridge._data_block_handler(request)

    def __init__(self,
                 role,
                 listen_port,
                 remote_address,
                 app_id=None,
                 rank=0,
                 streaming_mode=True,
                 compression=grpc.Compression.NoCompression):
        super().__init__(role, listen_port, remote_address, app_id, rank,
                         streaming_mode, compression)
        self._data_block_handler_fn = None

        # data transmit
        self._current_iter_id = None
        self._next_iter_id = 0
        self._receive_buffer = defaultdict(dict)
        self._trainer_client = BridgeClientManager(
            tws2_grpc.TrainerWorkerServiceStub,
            lambda: make_insecure_channel(
                remote_address, ChannelType.REMOTE,
                options=self._grpc_options, compression=self._compression
            )
        )
        tws2_grpc.add_TrainerWorkerServiceServicer_to_server(
            TrainerBridge.TrainerServicer(self), self._server
        )

    def get(self, *args):
        raise NotImplementedError('TrainerWorker does not implement `get`.')

    def _data_block_handler(self, request):
        assert self._connected, "Cannot load data before connect"
        if not self._data_block_handler_fn:
            raise RuntimeError("Received DataBlockMessage but " \
                               "no handler registered")
        metrics.emit_counter('load_data_block_counter', 1)
        if self._data_block_handler_fn(request):
            logging.info('Succeeded to load data block %s',
                         request.block_id)
            return common_pb.Status(code=common_pb.STATUS_SUCCESS)
        metrics.emit_counter('load_data_block_fail_counter', 1)
        logging.info('Failed to load data block %s', request.block_id)
        return common_pb.Status(code=common_pb.STATUS_INVALID_DATA_BLOCK)

    def _receive_data_arranger(self, request):
        data = request.payload
        with self._transmit_condition:
            self._receive_buffer[data.iter_id][data.name] = data
            self._transmit_condition.notifyAll()

    @property
    def current_iter_id(self):
        return self._current_iter_id

    def new_iter_id(self):
        iter_id = self._next_iter_id
        self._next_iter_id += 1
        return iter_id

    def start(self, iter_id):
        assert self._current_iter_id is None, "Last iter not finished"
        self._current_iter_id = iter_id
        logging.debug("Starting iter %d", iter_id)

    def commit(self):
        assert self._current_iter_id is not None, "Not started yet"
        with self._transmit_condition:
            last_iter_id = self._current_iter_id
            self._current_iter_id = None
            if last_iter_id in self._receive_buffer:
                del self._receive_buffer[last_iter_id]
        logging.debug("iter %d committed", last_iter_id)

    def register_data_block_handler(self, func):
        assert self._data_block_handler_fn is None, \
            "DataBlock handler already registered"
        self._data_block_handler_fn = func

    def load_data_block(self, count, block_id):
        msg = tws2_pb.LoadDataBlockRequest(count=count, block_id=block_id)
        logging.debug("sending DataBlock with id %s", block_id)
        status = self._trainer_client.rpc_with_retry(
            lambda: self._client.LoadDataBlock(msg),
            "Failed to send load data block request"
        )
        if status.code == common_pb.STATUS_SUCCESS:
            logging.info('Remote succeeded to load data block %s', block_id)
            return True
        logging.info('Remote failed to load data block %s. code: %d',
                     block_id, status.code)
        return False

    def send_proto(self, iter_id, name, proto):
        any_proto = google.protobuf.any_pb2.Any()
        any_proto.Pack(proto)
        msg = tws2_pb.TrainerWorkerMessage(
            data=tws2_pb.DataMessage(iter_id=iter_id, name=name,
                                     any_data=any_proto)
        )
        self.transmit(msg)
        logging.debug('Data: send protobuf %s for iter %d. seq_num=%d.',
                      name, iter_id, msg.seq_num)

    def send(self, iter_id, name, x):
        msg = tws2_pb.TrainerWorkerMessage(data=tws2_pb.DataMessage(
            iter_id=iter_id, name=name, tensor=tf.make_tensor_proto(x)))
        self.transmit(msg)
        logging.debug('Data: send %s for iter %d. seq_num=%d.',
                      name, iter_id, msg.seq_num)

    def send_op(self, name, x):
        def func(x):
            assert self._current_iter_id is not None, "Bridge not started"
            self.send(self._current_iter_id, name, x.numpy())

        out = tf.py_function(func=func, inp=[x], Tout=[], name='send_' + name)
        return out

    def receive_proto(self, iter_id, name):
        logging.debug('Data: Waiting to receive proto %s for iter %d.',
                      name, iter_id)
        with self._transmit_condition:
            while iter_id not in self._receive_buffer \
                    or name not in self._receive_buffer[iter_id]:
                self._transmit_condition.wait()
            data = self._receive_buffer[iter_id][name]
        logging.debug('Data: received %s for iter %d.', name, iter_id)
        return data.any_data

    def receive(self, iter_id, name):
        logging.debug('Data: Waiting to receive %s for iter %d.', name,
                      iter_id)
        start_time = time.time()
        with self._transmit_condition:
            while iter_id not in self._receive_buffer \
                    or name not in self._receive_buffer[iter_id]:
                self._transmit_condition.wait()
            data = self._receive_buffer[iter_id][name]
        duration = time.time() - start_time
        metrics.emit_timer('receive_timer', duration)
        logging.debug('Data: received %s for iter %d after %f sec.',
                      name, iter_id, duration)
        return tf.make_ndarray(data.tensor)

    def receive_op(self, name, dtype):
        def func():
            assert self._current_iter_id is not None, "Bridge not started"
            x = self.receive(self._current_iter_id, name)
            return tf.convert_to_tensor(x, dtype=dtype)

        return tf.py_function(func=func, inp=[], Tout=[dtype])[0]
