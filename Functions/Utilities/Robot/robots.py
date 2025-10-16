from . import motion
import threading
import time
class Robot:
    """Initialising the robot thread"""
    def __init__(self,
                 id_list=[0,1,2,782],
                 SERIAL_PORT = '/dev/ttyACM0',
                 BAUD_RATE = 115200):
        

        self.id_list=id_list
        self.SERIAL_PORT = SERIAL_PORT
        self.BAUD_RATE = BAUD_RATE 
        self.send_interval_s= 0.05
        self.stop_event=evt = threading.Event()
        self.command=len(self.id_list)*[90,90,'open']

        self.move_thread = threading.Thread(
            target=self.move_robot,
            kwargs={},
            daemon=True
        )
        try:
            self.move_thread.start()
        except KeyboardInterrupt:
            self.stop()


    def _set_server_reference(self, server_instance):
        """
        A special method the server will call upon registration.
        This is how we perform 'dependency injection'.
        """
        print("[Server Thread] Injecting server reference into AggregatorService.")
        self.server = server_instance


    def get_pose(self, id):
        print(f"Hello from Get Pos! You are robot - ",id)
        return [1,-1]
    
    def set_command(self,id,command):
        try:
            index = self.id_list.index(id)
            self.command[index*3:index*3+3]=command
        except ValueError:
            print("Invalid command Index!")
        

    def stop(self):
        self.command=len(self.id_list)*[90,90,'open']
        self.stop_event.set()

    def move_robot(self):

        while not self.stop_event.is_set(): 
            motion.move_robot(self.id_list, self.SERIAL_PORT, self.BAUD_RATE, self.command)
            time.sleep(0.05)

        



