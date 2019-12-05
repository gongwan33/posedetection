import json
import numpy as np
import mxnet
from gpuinfo import GPUInfo
import _thread
import time

class JSONOutput:
    def __init__(self, fname):
        ofname = ''.join(fname.split(".")[:-1])

        if len(ofname) < 1:
            ofname = fname + '-res.json'
        else:
            ofname = ofname + '.json'

        print("Open %s to write json data"%ofname)
        self.f = open(ofname, 'w+')

        if self.f is not None:
            self.f.write("[")

        self.anno_count = 0
        self.gpu_percent = 0
        self.gpu_memory = 0
        self.gpu_thread = True
        self.thread_lock = _thread.allocate_lock()

        try:
            _thread.start_new_thread(self.get_gpu_info, ())
            print("GPUINFO Thread Started.")
        except Exception as e:
            print("Unable to start GPUINFO Thread")
            print(e)

        return

    def get_gpu_info(self):
        print("GPUINFO: Start loop")
        while self.gpu_thread:
            self.thread_lock.acquire()
            self.gpu_percent, self.gpu_memory = GPUInfo.gpu_usage()
            self.thread_lock.release()
            time.sleep(2)

        print("GPUINFO: End loop")
        return

    def write(self, confidence, coords, scores, idx):
        scores = scores.asnumpy()
        scores = scores[np.where(scores >= 0)]

        keypoints = mxnet.ndarray.concat(coords, confidence, dim = 2)

        for i, points in enumerate(keypoints):
            points = points.asnumpy()
            points_ary = points.flatten().tolist()
            scores_ary = scores[i].astype("float").round(5)
            instance_num = len(keypoints)

            self.thread_lock.acquire()
            key_annotation = {
                "image_id": idx,
                "category_id": 1,
                "keypoints": points_ary,
                "score": scores_ary,
                "instance_num": instance_num,
                "gpu_percentage": self.gpu_percent,
                "gpu_memory": self.gpu_memory
            }
            self.thread_lock.release()

            if self.f is not None:
                if self.anno_count > 0:
                    self.f.write(",\n")

                json.dump(key_annotation, self.f)
                self.anno_count += 1

        return

    def release(self):
        if self.f is not None:
            self.f.write("]")
            self.f.close()
            self.gpu_thread = False
        return

