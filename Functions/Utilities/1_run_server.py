# vision_dashboard/1_run_server.py

from rpc_system import RPCServer
from services.data_service import Data
from services.robot_service import Robot
from services.task_service import Tasks

def main():
    """Initializes and runs the central RPC server."""
    server = RPCServer()
    
    # Register all the services that clients can interact with
    server.register_class(Data())
    server.register_class(Robot())
    server.register_class(Tasks())
    
    # This call blocks and runs the server indefinitely
    server.run()

if __name__ == "__main__":
    main()