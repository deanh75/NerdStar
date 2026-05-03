from werkzeug.serving import make_server
import threading
import subprocess

class ServerThread(threading.Thread):
    def __init__(self, app, name: str):
        super().__init__()
        self.daemon = False
        self.server = make_server('0.0.0.0', 5800, app, threaded=True)
        self.name = name
        
    def run(self):
        print(f"Server started: {self.name}")
        subprocess.run(["sudo", "scutil", "--set", "HostName", self.name], check=True)
        subprocess.run(["sudo", "scutil", "--set", "LocalHostName", self.name], check=True)
        subprocess.run(["sudo", "scutil", "--set", "ComputerName", self.name], check=True)
        self.server.serve_forever()

    def stop(self):
        self.server.shutdown()