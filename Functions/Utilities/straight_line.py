
import time
import rpc_system

k=rpc_system.RPCClient()
k.Robot.set_command(1,[90,90,'open'])
while 1:
    k.Robot.set_command(782,[95,94,'open'])
    time.sleep(3)
    k.Robot.set_command(782,[85,84,'open'])
    time.sleep(3)