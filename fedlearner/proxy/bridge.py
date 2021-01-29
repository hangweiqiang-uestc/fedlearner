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

import collections
import logging
import os
import threading
import time
from concurrent import futures

import grpc

from fedlearner.common import bridge_pb2 as bridge_pb
from fedlearner.common import bridge_pb2_grpc as bridge_grpc
from fedlearner.common import common_pb2 as common_pb
from fedlearner.common import metrics
from fedlearner.proxy.channel import make_insecure_channel, ChannelType


def make_ready_client(channel, stop_event=None,
                      client_fn=bridge_grpc.BridgeServiceStub):
    channel_ready = grpc.channel_ready_future(channel)
    wait_secs = 0.5
    start_time = time.time()
    while (stop_event is None) or (not stop_event.is_set()):
        try:
            channel_ready.result(timeout=wait_secs)
            break
        except grpc.FutureTimeoutError:
            logging.warning(
                'Channel has not been ready for %.2f seconds',
                time.time() - start_time)
            if wait_secs < 5.0:
                wait_secs *= 1.2
        except Exception as e:  # pylint: disable=broad-except
            logging.warning('Waiting channel ready: %s', repr(e))
    return client_fn(channel)


class BridgeClientManager:
    def __init__(self, client_fn, channel_fn):
        self._client_fn = client_fn
        self._channel_fn = channel_fn
        self._channel = channel_fn()
        self._client = client_fn(self._channel)

    def rpc_with_retry(self, sender, err_log):
        while True:
            with self._client_lock:
                try:
                    return sender()
                except Exception as e:  # pylint: disable=broad-except
                    logging.warning(
                        "%s: %s. Retry in 1s...", err_log, repr(e))
                    metrics.emit_counter('reconnect_counter', 1)
                    self._channel.close()
                    time.sleep(1)
                    self._channel = self._channel_fn()
                    self._client = make_ready_client(self._channel,
                                                     client_fn=self._client_fn)

    def __getattr__(self, item):
        return getattr(self._client, item)

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__ = state


class _MessageQueue(object):
    def __init__(self, window_size=100):
        super(_MessageQueue, self).__init__()
        self._window_size = window_size
        self._condition = threading.Condition()
        self._queue = collections.deque()
        self._next = 0

    def size(self):
        with self._condition:
            return len(self._queue)

    def confirm(self, next_seq_num):
        with self._condition:
            while self._queue and self._queue[0].seq_num < next_seq_num:
                self._queue.popleft()
                if self._next > 0:
                    self._next -= 1
            self._condition.notifyAll()

    def resend(self, seq_num):
        with self._condition:
            while self._next > 0 and \
                (self._next >= len(self._queue) or
                 self._queue[self._next].seq_num > seq_num):
                self._next -= 1
            if self._queue:
                logging.warning(
                    'Message with seq_num=%d missing. Resending from %d',
                    seq_num, self._queue[self._next].seq_num)
            self._condition.notifyAll()

    def put(self, msg):
        with self._condition:
            while len(self._queue) >= self._window_size:
                self._condition.wait()
            self._queue.append(msg)
            self._condition.notifyAll()

    def get(self):
        with self._condition:
            while self._next == len(self._queue):
                if not self._condition.wait(10.0) and self._queue:
                    logging.warning(
                        'Timeout waiting for confirmation. Resending from %d',
                        self._queue[0].seq_num)
                    self._next = 0
            assert self._next < len(self._queue)
            msg = self._queue[self._next]
            self._next += 1
            return msg


class Bridge(object):
    class BridgeServicer(bridge_grpc.BridgeServiceServicer):
        def __init__(self, bridge):
            super().__init__()
            self._bridge = bridge

        def Transmit(self, request, context):
            return self._bridge._transmit_handler(request)

        def StreamTransmit(self, request_iterator, context):
            for request in request_iterator:
                yield self._bridge._transmit_handler(request)

        def Connect(self, request, context):
            return self._bridge._connect_handler(request)

        def Heartbeat(self, request, context):
            return self._bridge._heartbeat_handler(request)

        def Terminate(self, request, context):
            return self._bridge._terminate_handler(request)

    def __init__(self,
                 role,
                 listen_port,
                 remote_address,
                 app_id=None,
                 rank=0,
                 streaming_mode=True,
                 compression=grpc.Compression.NoCompression):
        self._role = role
        self._listen_port = listen_port
        self._remote_address = remote_address
        if app_id is None:
            app_id = 'test_trainer'
        self._app_id = app_id
        self._rank = rank
        self._streaming_mode = streaming_mode
        self._compression = compression

        # Connection related
        self._connected = False
        self._connected_at = 0
        self._terminated = False
        self._terminated_at = 0
        self._peer_terminated = False
        self._identifier = '%s-%s-%d-%d' % (
            app_id, role, rank, int(time.time()))  # Ensure unique per run
        self._peer_identifier = ''

        # data transmit
        self._transmit_condition = threading.Condition()
        self._next_iter_id = 0

        # grpc client
        self._transmit_send_lock = threading.Lock()
        self._client_lock = threading.Lock()
        self._grpc_options = [
            ('grpc.max_send_message_length', 2 ** 31 - 1),
            ('grpc.max_receive_message_length', 2 ** 31 - 1)
        ]
        self._client = BridgeClientManager(
            bridge_grpc.BridgeServiceStub,
            lambda: make_insecure_channel(
                remote_address, ChannelType.REMOTE,
                options=self._grpc_options, compression=self._compression
            )
        )
        self._next_send_seq_num = int(role.lower() == 'leader')
        self._transmit_queue = _MessageQueue()
        self._client_daemon = None
        self._client_daemon_shutdown_fn = None

        # server
        self._transmit_receive_lock = threading.Lock()
        self._next_receive_seq_num = int(role.lower() == 'follower')
        self._receive_buffer = collections.deque()
        self._receive_buffer_size = 1 << 16

        self._server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=10),
            options=self._grpc_options,
            compression=self._compression)
        bridge_grpc.add_BridgeServiceServicer_to_server(
            Bridge.BridgeServicer(self), self._server
        )
        self._server.add_insecure_port('[::]:%d' % listen_port)

    def __del__(self):
        self.terminate()

    @property
    def role(self):
        return self._role

    @property
    def connected_at(self):
        if self._connected:
            return self._connected_at
        return None

    @property
    def terminated_at(self):
        if self._terminated:
            return self._terminated_at
        return None

    def connect(self):
        if self._connected:
            logging.warning("Bridge already connected!")
            return

        self._server.start()

        # Get ACK from peer
        msg = bridge_pb.ConnectRequest(app_id=self._app_id,
                                       worker_rank=self._rank,
                                       identifier=self._identifier)
        resp = self._client.rpc_with_retry(
            lambda: self._client.Connect(msg), "Bridge failed to connect"
        )
        logging.debug('Bridge now connected to peer.')

        # Ensure REQ from peer
        with self._transmit_condition:
            self._connected_at = max(self._connected_at, resp.timestamp)
            while not self._connected:
                self._transmit_condition.wait()
        logging.debug('Connect requested from peer.')

        if self._streaming_mode:
            logging.debug('Enter streaming mode.')
            self._client_daemon = threading.Thread(
                target=self._client_daemon_fn)
            self._client_daemon.start()
        logging.debug('Connect finish.')

    def transmit(self, payload):
        assert self._connected, "Cannot transmit before connect"
        metrics.emit_counter('send_counter', 1)
        msg = bridge_pb.BridgeRequest(payload=payload)
        with self._transmit_send_lock:
            msg.seq_num = self._next_send_seq_num
            self._next_send_seq_num += 2
            if self._streaming_mode:
                self._transmit_queue.put(msg)
                return

            def sender():
                rsp = self._client.Transmit(msg)
                assert rsp.status.code == common_pb.STATUS_SUCCESS, \
                    "Transmit error with code %d." % rsp.status.code
            self._client.rpc_with_retry(sender, "Bridge transmit failed")

    def get(self, *args):
        raise NotImplementedError

    def terminate(self, forced=False):
        if not self._connected or self._terminated:
            return
        self._terminated = True

        try:
            if self._client_daemon is not None:
                self._client_daemon_shutdown_fn()
                self._client_daemon.join()
        except Exception as e:  # pylint: disable=broad-except
            logging.warning(
                'Error during streaming shutdown: %s', repr(e))

        # Get ACK from peer
        resp = self._client.rpc_with_retry(
            lambda: self._client.Terminate(bridge_pb.TerminateRequest()),
            "Failed to send terminate message."
        )
        logging.debug('Waiting for peer to terminate.')

        # Ensure REQ from peer
        with self._transmit_condition:
            self._terminated_at = max(self._terminated_at, resp.timestamp)
            while not self._peer_terminated:
                self._transmit_condition.wait()

        self._server.stop(None)
        logging.debug("Bridge connection terminated.")

    def _connect_handler(self, request):
        assert request.app_id == self._app_id, \
            "Connection failed. Application id mismatch: %s vs %s" % (
                request.app_id, self._app_id)
        assert request.worker_rank == self._rank, \
            "Connection failed. Rank mismatch: %s vs %s" % (
                request.worker_rank, self._rank)
        assert len(request.identifier) > 0, \
            "Connection failed. An identifier should be offered!"

        with self._transmit_condition:
            if self._connected:
                # If a duplicated reqeust from peer, just ignore it.
                # If a new connect request from peer, suicide.
                if request.identifier != self._peer_identifier:
                    logging.error('Suicide as peer %s has restarted!',
                                  request.identifier)
                    os._exit(138)  # Tell Scheduler to restart myself
            else:
                self._peer_identifier = request.identifier
                self._connected = True
                self._connected_at = max(self._connected_at, int(time.time()))
                self._transmit_condition.notifyAll()

        return bridge_pb.ConnectResponse(app_id=self._app_id,
                                         worker_rank=self._rank,
                                         timestamp=self._connected_at)

    def _heartbeat_handler(self, request):
        return bridge_pb.HeartbeatResponse(app_id=self._app_id,
                                           worker_rank=self._rank)

    def _transmit_handler(self, request):
        assert self._connected, "Cannot transmit before connect"
        metrics.emit_counter('receive_counter', 1)
        with self._transmit_receive_lock:
            logging.debug("Received message seq_num=%d. "
                          "Wanted seq_num=%d.",
                          request.seq_num, self._next_receive_seq_num)
            if request.seq_num > self._next_receive_seq_num:
                return bridge_pb.BridgeResponse(
                    status=common_pb.Status(
                        code=common_pb.STATUS_MESSAGE_MISSING
                    ),
                    next_seq_num=self._next_receive_seq_num)
            if request.seq_num < self._next_receive_seq_num:
                return bridge_pb.BridgeResponse(
                    status=common_pb.Status(
                        code=common_pb.STATUS_MESSAGE_DUPLICATED
                    ),
                    next_seq_num=self._next_receive_seq_num)

            # request.seq_num == self._next_receive_seq_num
            self._next_receive_seq_num += 2
            self._receive_data_arranger(request)
            return bridge_pb.BridgeResponse(
                next_seq_num=self._next_receive_seq_num)

    def _receive_data_arranger(self, request):
        """
        Args:
            request: BridgeRequest from peer.

        Can be overridden to arrange receive data differently. Please not that
            self.get() should also be overridden accordingly.
        """
        raise NotImplementedError

    def _terminate_handler(self, request):
        with self._transmit_condition:
            self._peer_terminated = True
            self._terminated_at = max(self._terminated_at, int(time.time()))
            self._transmit_condition.notifyAll()
        return bridge_pb.TerminateResponse(timestamp=self._terminated_at)

    @staticmethod
    def _check_remote_heartbeat(client):
        try:
            rsp = client.Heartbeat(bridge_pb.HeartbeatRequest())
            logging.debug("Heartbeat success: %s:%d at iteration %s.",
                          rsp.app_id, rsp.worker_rank, rsp.current_iter_id)
            return True
        except Exception as e:  # pylint: disable=broad-except
            logging.warning("Heartbeat request failed: %s", repr(e))
            return False

    def _client_daemon_fn(self):
        """
        Daemon function that routinely sends messages in the MessageQueue.
        """
        stop_event = threading.Event()
        generator = None
        channel = make_insecure_channel(
            self._remote_address, ChannelType.REMOTE,
            options=self._grpc_options, compression=self._compression)
        client = make_ready_client(channel, stop_event)

        def shutdown_fn():
            while self._transmit_queue.size():
                logging.debug(
                    "Waiting for message queue's being cleaned. "
                    "Queue size: %d", self._transmit_queue.size())
                time.sleep(1)

            stop_event.set()
            if generator is not None:
                generator.cancel()

        self._client_daemon_shutdown_fn = shutdown_fn

        while not stop_event.is_set():
            try:
                def iterator():
                    while True:
                        item = self._transmit_queue.get()
                        logging.debug("Streaming send message seq_num=%d",
                                      item.seq_num)
                        yield item

                generator = client.StreamTransmit(iterator())
                for response in generator:
                    if response.status.code == common_pb.STATUS_SUCCESS:
                        self._transmit_queue.confirm(response.next_seq_num)
                        logging.debug("Message with seq_num=%d is "
                                      "confirmed", response.next_seq_num - 2)
                    elif response.status.code == \
                        common_pb.STATUS_MESSAGE_DUPLICATED:
                        self._transmit_queue.confirm(response.next_seq_num)
                        logging.debug("Resent Message with seq_num=%d is "
                                      "confirmed", response.next_seq_num - 2)
                    elif response.status.code == \
                        common_pb.STATUS_MESSAGE_MISSING:
                        self._transmit_queue.resend(response.next_seq_num)
                    else:
                        raise RuntimeError("Transmit failed with %d" %
                                           response.status.code)
            except Exception as e:  # pylint: disable=broad-except
                if not stop_event.is_set():
                    logging.warning("Bridge streaming broken: %s.", repr(e))
                    metrics.emit_counter('reconnect_counter', 1)
            finally:
                generator.cancel()
                channel.close()
                time.sleep(1)
                self._transmit_queue.resend(-1)
                channel = make_insecure_channel(
                    self._remote_address, ChannelType.REMOTE,
                    options=self._grpc_options, compression=self._compression)
                client = make_ready_client(channel, stop_event)
                self._check_remote_heartbeat(client)
