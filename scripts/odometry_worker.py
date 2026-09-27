"""JSON-pipe estimator, intended to run with no access to evaluator files."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import sys
from vln_localization.kiss_odometry import LidarOdometry

ap=argparse.ArgumentParser()
ap.add_argument('--deny-probe',required=True)
ap.add_argument('--voxel-size',type=float,default=.1)
ap.add_argument('--initial-threshold',type=float,default=.5)
ap.add_argument('--fixed-threshold',type=float)
ap.add_argument('--method',choices=('kiss','gicp'),default='kiss')
ap.add_argument('--prediction',choices=('constant_velocity','static'),default='constant_velocity')
ap.add_argument('--gicp-correspondence',type=float,default=.5)
args=ap.parse_args()
if Path(args.deny_probe).exists():raise RuntimeError('Evaluator directory is visible to estimator')
odometry=LidarOdometry(args.voxel_size,args.initial_threshold,args.fixed_threshold,args.method,args.prediction,args.gicp_correspondence)
print(json.dumps({'ready':True,'denied_directory_visible':False,'version':importlib.metadata.version('small-gicp' if args.method=='gicp' else 'kiss-icp'),
                  'config':odometry.config}),flush=True)
for line in sys.stdin:
    try:
        result=odometry.update(json.loads(line))
    except Exception as error:
        result={'valid':False,'control_ready':False,'reason':type(error).__name__,'latched':True}
        print(json.dumps(result),flush=True)
        raise
    print(json.dumps(result,allow_nan=False),flush=True)
