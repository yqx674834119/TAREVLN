"""Exercise the actual upstream/ported callback with a fixed API answer.

ROS transport and the external API are substituted; image processing, callback
branching and message publication run from the actual source method.
"""
import ast
import base64
import os
from pathlib import Path
import subprocess
import tempfile
import time
from types import SimpleNamespace as NS
import unittest
import cv2
import numpy as np
from pydantic import BaseModel

SOURCE=Path(__file__).resolve().parents[2]/'deps/sysnav-full'
FILE='src/vlm_node/vlm_node/vlm_reasoning_node.py'
BASE='478727e5a559be22822d5d093b57431449336287'


def callback(text):
    tree=ast.parse(text)
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='VLMNode')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='process_room_early_stop_1_query')
    namespace=dict(cv2=cv2,base64=base64,BaseModel=BaseModel,time=time,os=os,
                   RoomEarlyStop1=object,VlmAnswer=NS)
    exec(compile(ast.Module(body=[method],type_ignores=[]),FILE,'exec'),namespace)
    return namespace[method.name]


def invoke(method,enabled,decision,wrong_room):
    published=[];errors=[]
    answer=NS(early_stop=decision,steps=[])
    parse=lambda **kw:NS(choices=[NS(message=NS(parsed=answer))])
    self=NS(enable_room_early_stop=enabled,bridge=NS(imgmsg_to_cv2=lambda image,**kw:image),
        instruction='Find the screen in another meeting room',room_nav_vlm_model='fixture',
        room_early_stop_1_prompt='fixture',vlm_model=NS(beta=NS(chat=NS(completions=NS(parse=parse)))),
        get_logger=lambda:NS(info=lambda text:None,error=errors.append),
        room_navigation_publisher=NS(publish=published.append),publish_text_overlay=lambda text:None)
    msg=NS(room_id_1=2,room_id_2=1,anchor_point_1=NS(x=6.,y=-1.,z=.78),
           anchor_point_2=NS(x=0.,y=0.,z=.78),enter_wrong_room=wrong_room,
           image_1=np.zeros((16,16,3),dtype=np.uint8),image_2=np.zeros((16,16,3),dtype=np.uint8))
    previous=os.getcwd()
    with tempfile.TemporaryDirectory() as directory:
        try:os.chdir(directory);method(self,msg)
        finally:os.chdir(previous)
    if errors:raise AssertionError(errors)
    return [(p.room_id,p.answer_type,p.anchor_point.x,p.anchor_point.y) for p in published]


@unittest.skipUnless((SOURCE/'.git').exists(),'Pinned SysNav source required')
class EarlyStopCallbackTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original=staticmethod(callback(subprocess.check_output(['git','-C',str(SOURCE),'show',BASE+':'+FILE],text=True)))
        cls.current=staticmethod(callback((SOURCE/FILE).read_text()))

    def test_disabled_default_preserves_pinned_upstream_callback(self):
        for decision in (False,True):
            for wrong in (False,True):
                with self.subTest(decision=decision,wrong=wrong):
                    self.assertEqual(invoke(self.current,False,decision,wrong),invoke(self.original,False,decision,wrong))

    def test_true_answer_is_dropped_upstream_but_published_when_enabled(self):
        self.assertEqual(invoke(self.original,True,True,False),[])
        self.assertEqual(invoke(self.current,True,True,False),[(2,1,6.,-1.)])

    def test_false_answer_and_wrong_room_recovery(self):
        self.assertEqual(invoke(self.current,True,False,False),[])
        self.assertEqual(invoke(self.current,True,False,True),[(1,1,0.,0.)])


if __name__=='__main__':unittest.main()
