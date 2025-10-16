# rpc_system.py
import zmq
import pickle
import threading
import time     
from Robot.robots import Robot
# ==============================================================================
# 1. SHARED CODE: DUMMY CLASSES TO BE SERVED
# ==============================================================================

class Calculator:
    """A simple calculator class whose methods will be called remotely."""
    def add(self, a, b):
        # The print statement will appear in the server's console output
        print(f"[Server Thread] Executing: Calculator.add({a}, {b})")
        return a + b

    def subtract(self, a, b):
        print(f"[Server Thread] Executing: Calculator.subtract({a}, {b})")
        return a - b



# ==============================================================================
# 2. SERVER IMPLEMENTATION
# ==============================================================================
class RPCServer:
    def __init__(self, host="*", port=5555):
        self._host = host
        self._port = port
        self._services = {}
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REP)

    def get_service(self, name):
        """Allows services to look up other registered services."""
        service = self._services.get(name)
        if not service:
            raise NameError(f"Service '{name}' not found.")
        return service

    def register_class(self, instance):
        """Registers an instance and performs dependency injection if needed."""
        class_name = instance.__class__.__name__
        self._services[class_name] = instance
        print(f"[Server Thread] Registered class: {class_name}")

        # This is the dependency injection logic for inter-service calls.
        if hasattr(instance, '_set_server_reference'):
            instance._set_server_reference(self)

    def run(self):
        self._socket.bind(f"tcp://{self._host}:{self._port}")
        print(f"✅ [Server Thread] RPC Server started on tcp://{self._host}:{self._port}")
        while True:
            try:
                request = pickle.loads(self._socket.recv())
                action = request.get('action')
                class_name = request['class']
                service_instance = self.get_service(class_name)

                if action == 'call':
                    method = getattr(service_instance, request['method'])
                    result = method(*request['args'], **request['kwargs'])
                elif action == 'get':
                    result = getattr(service_instance, request['attribute'])
                elif action == 'set':
                    setattr(service_instance, request['attribute'], request['value'])
                    result = None  # Acknowledge success
                else:
                    raise ValueError(f"Invalid action: {action}")

                response = {'status': 'ok', 'result': result}
            except Exception as e:
                print(f"❌ [Server Thread] Error processing request: {e}")
                response = {'status': 'error', 'message': str(e)}
            
            self._socket.send(pickle.dumps(response))

# ==============================================================================
# 3. CLIENT IMPLEMENTATION
# ==============================================================================

class _AttributeProxy:
    """Proxy for accessing and setting remote variables."""
    def __init__(self, client, class_name):
        # Use super() to set internal attributes without triggering our __setattr__
        super().__setattr__('_client', client)
        super().__setattr__('_class_name', class_name)

    def __getattr__(self, name):
        """Makes an RPC 'get' call when an attribute is accessed."""
        payload = {'action': 'get', 'class': self._class_name, 'attribute': name}
        return self._client._call(**payload)

    def __setattr__(self, name, value):
        """Makes an RPC 'set' call when an attribute is assigned."""
        if name.startswith('_'):
            super().__setattr__(name, value)
        else:
            payload = {'action': 'set', 'class': self._class_name, 'attribute': name, 'value': value}
            self._client._call(**payload)

class _ServiceProxy:
    """Proxy for a remote class, handling both methods and variables."""
    def __init__(self, client, class_name):
        self._client = client
        self._class_name = class_name
        self.vars = _AttributeProxy(client, class_name)

    def __getattr__(self, name):
        def remote_method(*args, **kwargs):
            payload = {'action': 'call', 'class': self._class_name, 'method': name, 'args': args, 'kwargs': kwargs}
            return self._client._call(**payload)
        return remote_method

class RPCClient:
    def __init__(self, host="localhost", port=5555):
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REQ)
        self._socket.connect(f"tcp://{host}:{port}")

    def _call(self, **kwargs):
        self._socket.send(pickle.dumps(kwargs))
        response = pickle.loads(self._socket.recv())
        if response['status'] == 'ok':
            return response['result']
        else:
            raise Exception(f"Server error: {response['message']}")

    def __getattr__(self, name):
        return _ServiceProxy(self, name)


# ==============================================================================
# 4. MAIN EXECUTION BLOCK
# ==============================================================================

if __name__ == "__main__":
    
    # --- Server Setup ---
    def start_server():
        """Function to run the server in a separate thread."""
        server = RPCServer()
        server.register_class(Calculator())
        server.register_class(Robot())
        server.run()

    # Run the server in a daemon thread.
    # A daemon thread will exit automatically when the main program finishes.
    print("Starting server in a background thread...")
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()
    while 1:
    # Give the server a moment to start up and bind to the port.
        time.sleep(0.5)

    # --- Client Usage ---
    print("\nInitializing client...")
    client = RPCClient()

    print("\n--- Calling remote methods ---")
    
    # Call the 'add' method of the 'Calculator' class
    add_result = client.Calculator.add(10, 5)
    print(f"client.Calculator.add(10, 5) => {add_result}")


    print("\n--- Testing error handling ---")
    try:
        # Call a method that doesn't exist on the server
        client.Calculator.multiply(5, 5)
    except Exception as e:
        print(f"Caught expected error: {e}")

    try:
        # Call a class that doesn't exist on the server
        client.FileManager.delete_everything()
    except Exception as e:
        print(f"Caught expected error: {e}")
        
    print("\nClient script finished.")