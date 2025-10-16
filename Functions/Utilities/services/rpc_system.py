# vision_dashboard/rpc_system.py

import zmq
import pickle
import threading

class RPCServer:
    """The central server that hosts and manages shared data and services."""
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
        """Registers an instance and performs dependency injection."""
        class_name = instance.__class__.__name__
        self._services[class_name] = instance
        print(f"[Server] Registered service: {class_name}")
        if hasattr(instance, '_set_server_reference'):
            instance._set_server_reference(self)

    def run(self):
        self._socket.bind(f"tcp://{self._host}:{self._port}")
        print(f"✅ [Server] RPC Server started on tcp://{self._host}:{self._port}")
        while True:
            try:
                req = pickle.loads(self._socket.recv())
                service = self.get_service(req['class'])
                action = req.get('action', 'call')

                if action == 'call':
                    method = getattr(service, req['method'])
                    result = method(*req['args'], **req['kwargs'])
                elif action == 'get':
                    result = getattr(service, req['attribute'])
                elif action == 'set':
                    setattr(service, req['attribute'], req['value'])
                    result = True
                else:
                    raise ValueError(f"Invalid action: {action}")
                
                response = {'status': 'ok', 'result': result}
            except Exception as e:
                response = {'status': 'error', 'message': str(e)}
            
            self._socket.send(pickle.dumps(response))

class _AttributeProxy:
    """Internal proxy for accessing remote variables (e.g., client.Data.vars.arena_corners)."""
    def __init__(self, client, class_name):
        super().__setattr__('_client', client)
        super().__setattr__('_class_name', class_name)

    def __getattr__(self, name):
        return self._client._call(action='get', class_name=self._class_name, attribute=name)

    def __setattr__(self, name, value):
        self._client._call(action='set', class_name=self._class_name, attribute=name, value=value)

class _ServiceProxy:
    """Internal proxy for calling remote methods (e.g., client.Robot.get_pose())."""
    def __init__(self, client, class_name):
        self._client = client
        self._class_name = class_name
        self.vars = _AttributeProxy(client, class_name)

    def __getattr__(self, name):
        def remote_method(*args, **kwargs):
            return self._client._call(action='call', class_name=self._class_name, method=name, args=args, kwargs=kwargs)
        return remote_method

class RPCClient:
    """The client used by all other processes to communicate with the server."""
    def __init__(self, host="localhost", port=5555):
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REQ)
        self._socket.connect(f"tcp://{host}:{port}")

    def _call(self, **kwargs):
        # The keys in kwargs must match what the server expects: class_name, action, etc.
        self._socket.send(pickle.dumps(kwargs))
        response = pickle.loads(self._socket.recv())
        if response['status'] == 'ok':
            return response['result']
        else:
            raise Exception(f"Server error in '{kwargs.get('class_name')}': {response['message']}")

    def __getattr__(self, name):
        # This allows for the natural syntax: client.ClassName.method()
        return _ServiceProxy(self, name)