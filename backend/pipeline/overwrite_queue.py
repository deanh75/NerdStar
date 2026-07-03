# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file at
# the root directory of this project.

import queue

class OverwriteQueue(queue.Queue):
    def put_overwrite(self, item):
        while True:
            try:
                self.put_nowait(item)
                return
            except queue.Full:
                try:
                    self.get_nowait()  # Discard oldest
                except queue.Empty:
                    pass