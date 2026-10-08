import time
import os
import visdom
from threading import Thread
import argparse

class NvidiaSmiThread(Thread):
    flag = True
    def __init__(self, server, win_name):
        Thread.__init__(self)
        self.win_name = win_name
        self.vis = visdom.Visdom(server=server, use_incoming_socket=False, env="nvidia-smi")
    
    def run(self):
        print("start nvidia-smi thread....")
        while True:
            f = os.popen("nvidia-smi")
            nvidia_smi_text = f.read()

            nvidia_smi_text = "<pre>" + nvidia_smi_text +"</pre>"
            try:
                self.vis.text(nvidia_smi_text, win=self.win_name, opts={"title": self.win_name})
            except Exception as e:
                pass

            time.sleep(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--vis_server', type=str, default="192.168.1.246", help='the ip of the visdom server.')
    parser.add_argument('--win_name', type=str, required=True, help='the name of the window of the visdom')
    args = parser.parse_args()
    thread = NvidiaSmiThread(args.vis_server, args.win_name)
    thread.start()
