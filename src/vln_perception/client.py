"""Small image-only Unix socket client; no Torch import in the ROS process."""
import hashlib
import base64
import json
import socket


class SamClient:
    def __init__(self,path,timeout=20.):
        self.path=str(path)
        self.timeout=timeout

    def segment(self,frame):
        if set(frame)!={'jpeg','boxes_xyxy'}:
            raise ValueError('Only an image and boxes may cross the SAM boundary')
        with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as channel:
            channel.settimeout(self.timeout)
            channel.connect(self.path)
            channel.sendall(json.dumps(frame,allow_nan=False).encode()+b'\n')
            with channel.makefile('rb') as stream:
                line=stream.readline(8*1024*1024+1)
            if len(line)>8*1024*1024 or not line.endswith(b'\n'):
                raise ValueError('Incomplete or oversized SAM response')
        result=json.loads(line)
        if result.get('valid') is not True:
            raise ValueError('SAM service rejected image: '+str(result.get('error')))
        if result['image_sha256']!=hashlib.sha256(base64.b64decode(frame['jpeg'])).hexdigest():
            raise ValueError('SAM response belongs to a different image')
        if len(result['masks'])!=len(frame['boxes_xyxy']):
            raise ValueError('SAM response box count differs')
        return result
