"""Evaluator-only mesh slab audit. Never supply its occupancy/path to policy."""
import argparse
import heapq
import json
import math
from pathlib import Path
import numpy as np
import cv2
from pxr import Usd, UsdGeom

ap=argparse.ArgumentParser()
ap.add_argument('--out',required=True,type=Path)
ap.add_argument('--radius',type=float,default=.35)
args=ap.parse_args()
root=Path(__file__).resolve().parents[1]
args.out.mkdir(parents=True,exist_ok=False)
stage=Usd.Stage.Open(str(root/'data/sim/office/private/scene.usda'))
xf=UsdGeom.XformCache()
resolution=.05
xmin,ymin,xmax,ymax=-25.,10.,6.,50.
zmin,zmax=.08,.8
mask=np.zeros((round((ymax-ymin)/resolution)+1,round((xmax-xmin)/resolution)+1),dtype=np.uint8)

def clip(poly,z,above):
    output=[]
    for a,b in zip(poly,poly[1:]+poly[:1]):
        ina=(a[2]>=z) if above else (a[2]<=z)
        inb=(b[2]>=z) if above else (b[2]<=z)
        if ina:output.append(a)
        if ina!=inb:output.append(a+(b-a)*(z-a[2])/(b[2]-a[2]))
    return output

triangles=0
for prim in stage.Traverse():
    if not prim.IsA(UsdGeom.Mesh):continue
    mesh=UsdGeom.Mesh(prim)
    p=np.asarray(mesh.GetPointsAttr().Get(),dtype=float)
    if not len(p):continue
    transform=np.asarray(xf.GetLocalToWorldTransform(prim))
    p=(np.c_[p,np.ones(len(p))]@transform)[:,:3]
    lo,hi=p.min(axis=0),p.max(axis=0)
    if hi[0]<xmin or lo[0]>xmax or hi[1]<ymin or lo[1]>ymax or hi[2]<zmin or lo[2]>zmax:continue
    counts=np.asarray(mesh.GetFaceVertexCountsAttr().Get())
    indices=np.asarray(mesh.GetFaceVertexIndicesAttr().Get())
    t=[];offset=0
    for n in counts:
        face=indices[offset:offset+n];offset+=n
        for k in range(1,n-1):t.append([face[0],face[k],face[k+1]])
    if not t:continue
    tri=p[np.asarray(t)]
    lo,hi=tri.min(axis=1),tri.max(axis=1)
    tri=tri[(hi[:,0]>=xmin)&(lo[:,0]<=xmax)&(hi[:,1]>=ymin)&(lo[:,1]<=ymax)&(hi[:,2]>=zmin)&(lo[:,2]<=zmax)]
    for face in tri:
        poly=clip(clip(list(face),zmin,True),zmax,False)
        if len(poly)<3:continue
        points=np.rint((np.asarray(poly)[:,:2]-[xmin,ymin])/resolution).astype(np.int32)
        cv2.fillConvexPoly(mask,points,1)
        triangles+=1
clearance=cv2.distanceTransform(1-mask,cv2.DIST_L2,cv2.DIST_MASK_PRECISE)*resolution
free=clearance>=args.radius+resolution/math.sqrt(2)

def ij(p):return (round((p[1]-ymin)/resolution),round((p[0]-xmin)/resolution))
def xy(p):return [p[1]*resolution+xmin,p[0]*resolution+ymin]
start=ij([-11.,29.5])
tasks=json.loads((root/'sim/fixtures/office-evaluation.json').read_text())['tasks']
goals={}
for name,task in tasks.items():
    for index,r in enumerate(task['goal_rectangles']):
        g=np.zeros_like(free)
        a,b=ij(r[:2]),ij(r[2:])
        g[a[0]:b[0]+1,a[1]:b[1]+1]=True
        goals[f'{name}-{index}']=g&free
queue=[(0.,start)] if free[start] else [];cost={start:0.};parent={};paths={}
while queue:
    d,current=heapq.heappop(queue)
    if d!=cost[current]:continue
    for name,goal in goals.items():
        if name not in paths and goal[current]:
            route=[current]
            while route[-1] in parent:route.append(parent[route[-1]])
            paths[name]={'length_m':d,'route':[xy(x) for x in route[::-1]]}
    if len(paths)==len(goals):break
    for dy,dx in ((0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)):
        n=current[0]+dy,current[1]+dx
        if not(0<=n[0]<free.shape[0] and 0<=n[1]<free.shape[1]) or not free[n]:continue
        if dx and dy and (not free[current[0]+dy,current[1]] or not free[current[0],current[1]+dx]):continue
        nd=d+math.hypot(dx,dy)*resolution
        if nd<cost.get(n,math.inf):cost[n]=nd;parent[n]=current;heapq.heappush(queue,(nd,n))
report={'scope':'evaluator-only full-mesh slab projection, not sensor replay or policy performance',
        'resolution':resolution,'radius':args.radius,'z_slab':[zmin,zmax],'start':xy(start),
        'start_free':bool(free[start]),'triangles':triangles,
        'goals':{n:{'reachable':n in paths,'length_m':paths[n]['length_m'] if n in paths else None} for n in goals}}
(args.out/'report.json').write_text(json.dumps(report,indent=2))
(args.out/'private-routes.json').write_text(json.dumps(paths))
np.savez_compressed(args.out/'private-occupancy.npz',occupied=mask,clearance=clearance,free=free)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,ax=plt.subplots(figsize=(8,10))
ax.imshow(mask,extent=[xmin,xmax,ymin,ymax],origin='lower',cmap='Greys',alpha=.7)
ax.contour(clearance,levels=[args.radius],extent=[xmin,xmax,ymin,ymax],origin='lower',colors=['lightgray'],linewidths=.3)
for name,path in paths.items():
    p=np.array(path['route']);ax.plot(p[:,0],p[:,1],label=name)
ax.scatter([-11],[29.5],marker='*',s=100,color='red');ax.legend();ax.set_aspect('equal')
ax.set_title('Evaluator geometry audit; paths hidden from policy')
fig.savefig(args.out/'routes.png',dpi=160)
print(json.dumps(report),flush=True)
