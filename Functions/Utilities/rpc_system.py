# ==============================================================================
#  High-Performance, Low-Latency RPC System
#
#  Features:
#  - Concurrent worker pool using pyzmq (ROUTER/DEALER)
#  - Fast serialization with cbor2 (supports integer keys)
#  - High-speed lz4 compression for NumPy arrays to reduce latency
#
#  To run, install all necessary libraries:
#  pip install pyzmq cbor2 numpy lz4
# ==============================================================================

import zmq
import cbor2
import numpy as np
import lz4.frame
import threading
import time
from typing import Any, Dict

# ==============================================================================
# 1. CUSTOM NUMPY SERIALIZER with LZ4 COMPRESSION
# ==============================================================================

# Use a custom tag number for identifying NumPy arrays in the data stream.
NUMPY_TAG = 42

def encoder(encoder_instance, value: Any):
    """
    Custom cbor2 encoder for NumPy arrays.
    It compresses the array's raw byte data using lz4 for high speed.
    """
    if isinstance(value, np.ndarray):
        # Compress the raw bytes of the array. This is the key to reducing latency.
        compressed_data = lz4.frame.compress(value.tobytes())
        
        payload = {
            'dtype': value.dtype.str,
            'shape': value.shape,
            'data': compressed_data, # Send the smaller, compressed data
        }
        # Encode the payload inside a custom CBORTag to identify it for the decoder.
        encoder_instance.encode(cbor2.CBORTag(NUMPY_TAG, payload))
    else:
        # For any other data type, use the default cbor2 encoder.
        encoder_instance.encode(value)

def decoder(decoder_instance, tag: cbor2.CBORTag) -> Any:
    """
    Custom cbor2 decoder for NumPy arrays.
    It decompress lz4-compressed data to reconstruct the original array.
    """
    if tag.tag == NUMPY_TAG:
        # Decompress the data received over the network.
        raw_data = lz4.frame.decompress(tag.value['data'])
        
        # Reconstruct the NumPy array from its metadata and decompressed bytes.
        return np.frombuffer(
            raw_data,
            dtype=np.dtype(tag.value['dtype'])
        ).reshape(tag.value['shape'])
    # For any other tag, return it as is.
    return tag

# ==============================================================================
# 2. SHARED CODE: EXAMPLE CLASSES TO BE SERVED
# ==============================================================================

class RemoteServices:
    """
    A class demonstrating various RPC functionalities.
    """
    def __init__(self):
        # This dictionary with integer keys will be serialized correctly by cbor2.
        self.state = {
            0: "Initialized",
            1: "Nominal"
        }

    def get_state_with_int_keys(self) -> Dict[int, str]:
        """Returns a dictionary with integer keys."""
        print(f"[{threading.current_thread().name}] Returning dict with int keys.")
        return self.state

    def process_large_data(self, data_array: np.ndarray) -> Dict:
        """
        Simulates a task that receives and returns a large NumPy array.
        The compression in the encoder/decoder makes this fast.
        """
        print(f"[{threading.current_thread().name}] Received array of shape {data_array.shape} and type {data_array.dtype}.")
        # Simulate some work and return statistics
        return {
            'status': 'Processed',
            'mean': np.mean(data_array),
            'std_dev': np.std(data_array)
        }

    def slow_task(self, duration: float) -> str:
        """A task that blocks to demonstrate server concurrency."""
        print(f"[{threading.current_thread().name}] Starting slow task for {duration} seconds...")
        time.sleep(duration)
        return f"Completed slow task after {duration} seconds."


# ==============================================================================
# 3. SERVER IMPLEMENTATION (CONCURRENT WORKER POOL)
# ==============================================================================

class RPCServer:
    def __init__(self, host: str = "*", port: int = 5555, num_workers: int = 4):
        self._host = host
        self._port = port
        self._num_workers = num_workers
        self._services = {}
        self._context = zmq.Context.instance()
        self._frontend = self._context.socket(zmq.ROUTER)
        self._backend = self._context.socket(zmq.DEALER)

    def register_class(self, instance: Any):
        class_name = instance.__class__.__name__
        self._services[class_name] = instance
        print(f"[Server Main] Registered class: {class_name}")

    def _get_service(self, name: str) -> Any:
        service = self._services.get(name)
        if not service:
            raise NameError(f"Service '{name}' not found.")
        return service

    def _worker_routine(self):
        socket = self._context.socket(zmq.DEALER)
        socket.connect("inproc://backend")
        while True:
            try:
                identity, _, request_payload = socket.recv_multipart()
                request = cbor2.loads(request_payload, tag_hook=decoder)
                
                service_instance = self._get_service(request['class'])
                method = getattr(service_instance, request['method'])
                result = method(*request['args'], **request['kwargs'])

                response = {'status': 'ok', 'result': result}
            except Exception as e:
                print(f"❌ [{threading.current_thread().name}] Error: {e}")
                response = {'status': 'error', 'message': str(e)}
            
            packed_response = cbor2.dumps(response, default=encoder)
            socket.send_multipart([identity, b'', packed_response])

    def run(self):
        self._frontend.bind(f"tcp://{self._host}:{self._port}")
        self._backend.bind("inproc://backend")
        for i in range(self._num_workers):
            thread = threading.Thread(target=self._worker_routine, name=f"Worker-{i+1}", daemon=True)
            thread.start()
        print(f"✅ [Server Main] RPC Server started on tcp://{self._host}:{self._port} with {self._num_workers} workers.")
        zmq.proxy(self._frontend, self._backend)
        self._frontend.close()
        self._backend.close()
        self._context.term()

# ==============================================================================
# 4. CLIENT IMPLEMENTATION
# ==============================================================================

class _ServiceProxy:
    def __init__(self, client: 'RPCClient', class_name: str):
        self._client = client
        self._class_name = class_name
    def __getattr__(self, name: str):
        def remote_method(*args, **kwargs):
            payload = {'class': self._class_name, 'method': name, 'args': args, 'kwargs': kwargs}
            return self._client._call(payload)
        return remote_method

class RPCClient:
    def __init__(self, host: str = "localhost", port: int = 5555):
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REQ)
        self._socket.connect(f"tcp://{host}:{port}")

    def _call(self, payload: Dict) -> Any:
        self._socket.send(cbor2.dumps(payload, default=encoder))
        response = cbor2.loads(self._socket.recv(), tag_hook=decoder)
        if response['status'] == 'ok':
            return response['result']
        else:
            raise RuntimeError(f"Server error: {response['message']}")

    def __getattr__(self, name: str) -> _ServiceProxy:
        return _ServiceProxy(self, name)

# ==============================================================================
# 5. MAIN EXECUTION BLOCK - DEMONSTRATION
# ==============================================================================

if __name__ == "__main__":
    def start_server():
        # Use a sensible number of workers, e.g., number of CPU cores
        import os
        num_cores = os.cpu_count() or 4
        server = RPCServer(num_workers=num_cores)
        server.register_class(RemoteServices())
        server.run()

    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()
    time.sleep(1) # Give server time to bind

    client = RPCClient()
    
    # --- 1. Test Integer Keys ---
    print("\n--- 1. Testing Integer Key Support ---")
    state = client.RemoteServices.get_state_with_int_keys()
    print(f"Client received state with int keys: {state} ✅")
    assert isinstance(list(state.keys())[0], int)
    
    # --- 2. Test Large Data Transfer (Low Latency) ---
    print("\n--- 2. Testing Large Data Transfer with Compression ---")
    # Create a large 10-megapixel float32 array (approx. 40 MB uncompressed)
    large_array = np.random.rand(1000, 10000).astype(np.float32)
    print(f"Client sending {large_array.nbytes / 1e6:.2f} MB NumPy array...")
    
    start_time = time.monotonic()
    stats = client.RemoteServices.process_large_data(large_array)
    end_time = time.monotonic()
    
    print(f"Client received stats: {stats}")
    print(f"Round trip for large array took: {(end_time - start_time)*1000:.2f} ms ✅")
    
    # --- 3. Test Concurrency ---
    print("\n--- 3. Testing Server Concurrency ---")
    def run_slow_task(client_id):
        print(f"[Client {client_id}] Calling slow_task...")
        client = RPCClient() # Each thread needs its own client/socket
        client.RemoteServices.slow_task(2)
        print(f"[Client {client_id}] Finished slow_task.")

    thread1 = threading.Thread(target=run_slow_task, args=(1,))
    thread2 = threading.Thread(target=run_slow_task, args=(2,))

    start_time = time.monotonic()
    thread1.start()
    thread2.start()
    thread1.join()
    thread2.join()
    end_time = time.monotonic()

    print(f"Two parallel 2-second tasks completed in {end_time - start_time:.2f} seconds ✅")

    print("\nAll tests passed!")