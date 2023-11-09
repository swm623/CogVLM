# @title ## 1.2. Start `File Explorer`
# @markdown This will work in real-time even when you run other cells
import threading
from imjoy_elfinder.app import main

open_in_new_tab = False  # @param {type:"boolean"}

def start_file_explorer(root_dir, port=8765):
    try:
        main(["--root-dir=" + root_dir, "--port=" + str(port)])
    except Exception as e:
        print("Error starting file explorer:", str(e))


def open_file_explorer(open_in_new_tab=False, root_dir="", port=8765):
    thread = threading.Thread(target=start_file_explorer, args=[root_dir, port])
    thread.start()



# Example usage
open_file_explorer(open_in_new_tab=open_in_new_tab, root_dir="/ML-A100/sshare-app/", port=8765)