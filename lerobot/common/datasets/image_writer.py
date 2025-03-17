#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import multiprocessing
import queue
import threading
from pathlib import Path
import time

import numpy as np
import PIL.Image
import torch
import multiprocessing as mp
from queue import Empty


def safe_stop_image_writer(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            dataset = kwargs.get("dataset")
            image_writer = getattr(dataset, "image_writer", None) if dataset else None
            if image_writer is not None:
                print("Waiting for image writer to terminate...")
                image_writer.stop()
            raise e

    return wrapper


def image_array_to_pil_image(image_array: np.ndarray, range_check: bool = True) -> PIL.Image.Image:
    # TODO(aliberts): handle 1 channel and 4 for depth images
    if image_array.ndim != 3:
        raise ValueError(f"The array has {image_array.ndim} dimensions, but 3 is expected for an image.")

    if image_array.shape[0] == 3:
        # Transpose from pytorch convention (C, H, W) to (H, W, C)
        image_array = image_array.transpose(1, 2, 0)

    elif image_array.shape[-1] != 3:
        raise NotImplementedError(
            f"The image has {image_array.shape[-1]} channels, but 3 is required for now."
        )

    if image_array.dtype != np.uint8:
        if range_check:
            max_ = image_array.max().item()
            min_ = image_array.min().item()
            if max_ > 1.0 or min_ < 0.0:
                raise ValueError(
                    "The image data type is float, which requires values in the range [0.0, 1.0]. "
                    f"However, the provided range is [{min_}, {max_}]. Please adjust the range or "
                    "provide a uint8 image with values in the range [0, 255]."
                )

        image_array = (image_array * 255).astype(np.uint8)

    return PIL.Image.fromarray(image_array)


def write_image(image: np.ndarray | PIL.Image.Image, fpath: Path):
    try:
        if isinstance(image, np.ndarray):
            img = image_array_to_pil_image(image)
        elif isinstance(image, PIL.Image.Image):
            img = image
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")
        img.save(fpath)
    except Exception as e:
        print(f"Error writing image {fpath}: {e}")


def worker_thread_loop(queue: queue.Queue):
    while True:
        item = queue.get()
        if item is None:
            queue.task_done()
            break
        image_array, fpath = item
        write_image(image_array, fpath)
        queue.task_done()


def worker_process(queue: queue.Queue, num_threads: int):
    threads = []
    for _ in range(num_threads):
        t = threading.Thread(target=worker_thread_loop, args=(queue,))
        t.daemon = True
        t.start()
        threads.append(t)
    for t in threads:
        t.join()


class AsyncImageWriter:
    """
    This class abstract away the initialisation of processes or/and threads to
    save images on disk asynchrounously, which is critical to control a robot and record data
    at a high frame rate.

    When `num_processes=0`, it creates a threads pool of size `num_threads`.
    When `num_processes>0`, it creates processes pool of size `num_processes`, where each subprocess starts
    their own threads pool of size `num_threads`.

    The optimal number of processes and threads depends on your computer capabilities.
    We advise to use 4 threads per camera with 0 processes. If the fps is not stable, try to increase or lower
    the number of threads. If it is still not stable, try to use 1 subprocess, or more.
    """

    def __init__(self, num_processes: int = 0, num_threads: int = 4):
        self.num_threads = num_threads
        self.num_processes = num_processes
        self.queue = queue.Queue()
        self.processes = []
        self.threads = []
        self._graceful_shutdown = True
        self._start()

    def _start(self):
        if self.num_processes > 0:
            for i in range(self.num_processes):
                p = mp.Process(
                    target=worker_process,
                    args=(self.queue, self.num_threads // self.num_processes),
                    daemon=True,
                )
                p.start()
                self.processes.append(p)
        else:
            for i in range(self.num_threads):
                t = threading.Thread(target=worker_thread_loop, args=(self.queue,), daemon=True)
                t.start()
                self.threads.append(t)

    def save_image(self, image: torch.Tensor | np.ndarray | PIL.Image.Image, fpath: Path):
        if isinstance(image, torch.Tensor):
            # Convert tensor to numpy array to minimize main process time
            image = image.cpu().numpy()
        self.queue.put((image, fpath))

    def wait_until_done(self, timeout=None):
        """
        Wait until all image writing tasks are completed.
        
        Args:
            timeout (float, optional): Maximum time to wait in seconds.
                                     If None, wait indefinitely.
        
        Returns:
            bool: True if all tasks completed, False if timeout occurred.
        """
        if not self._graceful_shutdown:
            # Don't wait if we're not doing a graceful shutdown
            return True
            
        try:
            # Use a timeout to avoid blocking indefinitely
            wait_start = time.time()
            while not self.queue.empty():
                if timeout is not None and time.time() - wait_start > timeout:
                    return False
                time.sleep(0.1)
            self.queue.join()
            return True
        except KeyboardInterrupt:
            # Handle user interruption by falling back to non-graceful shutdown
            print("KeyboardInterrupt received while waiting for image writer. "
                  "Switching to immediate shutdown.")
            self.stop(graceful=False)
            return False

    def stop(self, graceful=True):
        """
        Stop the image writer.
        
        Args:
            graceful (bool): If True, wait for all queued tasks to complete.
                             If False, terminate immediately, discarding pending tasks.
        """
        self._graceful_shutdown = graceful
        
        if not graceful:
            # Clear the queue to prevent waiting on join
            while not self.queue.empty():
                try:
                    self.queue.get_nowait()
                    self.queue.task_done()
                except Empty:
                    break
        
        for p in self.processes:
            p.terminate()
            p.join(timeout=0.5)
        self.processes = []
        self.threads = []
