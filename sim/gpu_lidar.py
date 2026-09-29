"""Simulator-private CUDA ray queries over the actual static collision meshes.

Dynamic robot silhouettes and the infinite PhysX plane use the original query.
Unsupported collider types fail closed; no simplified map is supplied to policy.
"""
import itertools
import time
import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics
import warp as wp


@wp.kernel
def _cast(mesh:wp.uint64,origin:wp.vec3,directions:wp.array(dtype=wp.vec3),
          bounds:wp.array(dtype=wp.vec3),box_count:int,planes:wp.array(dtype=wp.vec4),plane_count:int,
          ranges:wp.array(dtype=float),fallback:wp.array(dtype=int),diagnostic:wp.array(dtype=wp.vec4),faces:wp.array(dtype=int)):
    i=wp.tid();d=directions[i]
    faces[i]=-1
    hit=wp.mesh_query_ray(mesh,origin,d,10.0)
    distance=float(10.0)
    if hit.result: distance=hit.t
    ranges[i]=distance if hit.result else -1.0
    needs_cpu=int(0)
    diagnostic[i]=wp.vec4(0.0)
    if hit.result:
        faces[i]=hit.face
        va=wp.mesh_eval_position(mesh,hit.face,1.0,0.0)
        vb=wp.mesh_eval_position(mesh,hit.face,0.0,1.0)
        vc=wp.mesh_eval_position(mesh,hit.face,0.0,0.0)
        area=wp.length(wp.cross(vb-va,vc-va))*0.5
        diagnostic[i]=wp.vec4(hit.u,hit.v,wp.dot(hit.normal,d),area)
        if wp.min(hit.u,wp.min(hit.v,1.0-hit.u-hit.v))<0.02 or area<1.0e-6 or wp.abs(wp.dot(hit.normal,d))<0.02:needs_cpu=1
    # Warp queries both triangle sides; PhysX rejects back faces. Near mesh
    # silhouettes the two engines also have different watertight tolerances.
    # A six-ray 2 mm origin packet detects unstable edges, including triangles
    # missed by the center ray, without changing the actual sensor ray.
    if hit.result and wp.dot(hit.normal,d)>0.0:needs_cpu=1
    for axis in range(3):
        for sign in range(2):
            delta=wp.vec3(0.0)
            delta[axis]=0.002*float(2*sign-1)
            peer=wp.mesh_query_ray(mesh,origin+delta,d,10.0)
            if peer.result!=hit.result:
                needs_cpu=1
            elif peer.result:
                denom=wp.dot(peer.normal,d)
                if denom>=-0.02:
                    needs_cpu=1
                else:
                    projected=peer.t+wp.dot(peer.normal,delta)/denom
                    if wp.abs(projected-distance)>0.001:needs_cpu=1
    for b in range(box_count):
        lo=float(0.0);hi=distance+0.002
        for axis in range(3):
            v=d[axis]
            if wp.abs(v)<1.0e-8:
                if origin[axis]<bounds[2*b][axis] or origin[axis]>bounds[2*b+1][axis]:hi=-1.0
            else:
                a=(bounds[2*b][axis]-origin[axis])/v
                c=(bounds[2*b+1][axis]-origin[axis])/v
                lo=wp.max(lo,wp.min(a,c));hi=wp.min(hi,wp.max(a,c))
        if hi>=lo:needs_cpu=1
    for b in range(plane_count):
        p=planes[b];normal=wp.vec3(p[0],p[1],p[2]);denom=wp.dot(normal,d)
        if wp.abs(denom)>1.0e-8:
            t=-(wp.dot(normal,origin)+p[3])/denom
            if t>=0.0 and t<=distance+0.002:needs_cpu=1
    fallback[i]=needs_cpu


def _triangles(mesh,offset):
    counts=np.asarray(mesh.GetFaceVertexCountsAttr().Get(),dtype=np.int64)
    indices=np.asarray(mesh.GetFaceVertexIndicesAttr().Get(),dtype=np.int32)
    if np.all(counts==3) and not mesh.GetHoleIndicesAttr().Get():return indices+offset
    out=[];start=0
    holes=set(mesh.GetHoleIndicesAttr().Get() or [])
    for fi,n in enumerate(counts):
        face=indices[start:start+n];start+=n
        if fi in holes:continue
        for j in range(1,n-1):out.extend((face[0]+offset,face[j]+offset,face[j+1]+offset))
    return np.asarray(out,dtype=np.int32)


def cpu_ranges(query,origin,directions,exclude_robot=True):
    distances=np.full(len(directions),-1.,dtype=np.float32)
    o=tuple(map(float,origin))
    for i,d in enumerate(directions.tolist()):
        hit=query.raycast_closest(o,tuple(d),10.)
        if hit['hit'] and hit['distance']>.4:
            if exclude_robot and any(str(hit.get(k,'')).startswith('/World/Go2') for k in ('rigidBody','collision')):continue
            distances[i]=hit['distance']
    return distances


class CudaLidar:
    def __init__(self,stage,robot,beam_count,device='cuda:3'):
        started=time.perf_counter();wp.init()
        self.device=device;self.robot=robot;self.beam_count=beam_count
        xf=UsdGeom.XformCache();bb=UsdGeom.BBoxCache(Usd.TimeCode.Default(),['default','render','proxy','guide'],False,True)
        points=[];indices=[];self.mesh_paths=[];self.face_ends=[];offset=0;self.robot_bounds={};planes=[];meshes=0
        for prim in stage.Traverse(Usd.TraverseInstanceProxies()):
            if not prim.HasAPI(UsdPhysics.CollisionAPI) or not UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get():continue
            parent=prim;rigid=None
            while parent and not parent.IsPseudoRoot():
                if parent.HasAPI(UsdPhysics.RigidBodyAPI):rigid=parent;break
                parent=parent.GetParent()
            if str(prim.GetPath()).startswith('/World/Go2/'):
                if rigid is None:raise ValueError('Robot collider has no rigid parent: '+str(prim.GetPath()))
                bound=bb.ComputeRelativeBound(prim,rigid).ComputeAlignedRange()
                if bound.IsEmpty():raise ValueError('Empty robot bounds: '+str(prim.GetPath()))
                lo=np.asarray(bound.GetMin());hi=np.asarray(bound.GetMax());name=rigid.GetName()
                if name in self.robot_bounds:
                    lo=np.minimum(lo,self.robot_bounds[name][0]);hi=np.maximum(hi,self.robot_bounds[name][1])
                self.robot_bounds[name]=(lo,hi)
                continue
            if rigid is not None:raise ValueError('Dynamic environment collider unsupported: '+str(prim.GetPath()))
            transform=np.asarray(xf.GetLocalToWorldTransform(prim))
            if prim.GetTypeName()=='Plane':
                axis=str(prim.GetAttribute('axis').Get() or 'Z');normal=np.eye(3)['XYZ'.index(axis)]
                normal=np.linalg.inv(transform[:3,:3])@normal;normal/=np.linalg.norm(normal)
                planes.append([*normal,-float(normal@transform[3,:3])]);continue
            if not prim.IsA(UsdGeom.Mesh):raise ValueError('Unsupported collider: '+str(prim.GetPath()))
            approximation=UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()
            if approximation not in (None,'none'):raise ValueError('Non-triangle collider: '+str(prim.GetPath()))
            mesh=UsdGeom.Mesh(prim);p=np.asarray(mesh.GetPointsAttr().Get(),dtype=np.float64)
            if not len(p):continue
            if mesh.GetPointsAttr().ValueMightBeTimeVarying():raise ValueError('Animated mesh unsupported')
            points.append((np.c_[p,np.ones(len(p))]@transform)[:,:3].astype(np.float32))
            indices.append(_triangles(mesh,offset));offset+=len(p);meshes+=1
            self.mesh_paths.append(str(prim.GetPath()));self.face_ends.append(sum(len(i)//3 for i in indices))
        if not self.robot_bounds:raise ValueError('No robot collision shapes found')
        self.body_names=list(robot._articulation_view.body_names)
        unknown=set(self.robot_bounds)-set(self.body_names)
        if unknown:raise ValueError('Unmapped robot collision links: '+str(unknown))
        self.link_indices=[self.body_names.index(k) for k in self.robot_bounds]
        self.corners=np.asarray([list(itertools.product(*zip(lo-.002,hi+.002))) for lo,hi in self.robot_bounds.values()])
        self.mesh=wp.Mesh(points=wp.array(np.concatenate(points),dtype=wp.vec3,device=device),
                          indices=wp.array(np.concatenate(indices),dtype=int,device=device))
        self.directions=wp.empty(beam_count,dtype=wp.vec3,device=device)
        self.bounds=wp.empty(2*len(self.corners),dtype=wp.vec3,device=device)
        self.planes=wp.array(np.asarray(planes,dtype=np.float32).reshape(-1,4),dtype=wp.vec4,device=device)
        self.ranges=wp.empty(beam_count,dtype=float,device=device)
        self.diagnostic=wp.empty(beam_count,dtype=wp.vec4,device=device)
        self.faces=wp.empty(beam_count,dtype=int,device=device)
        self.fallback=wp.empty(beam_count,dtype=int,device=device)
        self.info={'backend':'warp-hybrid','device':device,'warp_version':wp.__version__,'meshes':meshes,
                   'vertices':offset,'triangles':sum(len(i)//3 for i in indices),'dynamic_links':list(self.robot_bounds),
                   'planes':len(planes),'build_seconds':time.perf_counter()-started}
        self.last={}

    def scan(self,query,origin,directions,exclude_robot=True):
        started=time.perf_counter()
        # PhysX tensor API returns current link poses, including XYZW quaternions.
        poses=np.asarray(self.robot._articulation_view._physics_view.get_link_transforms())[0,self.link_indices]
        q=poses[:,3:];v=self.corners;u=q[:,None,:3];w=q[:,None,3:4]
        rotated=v+2*np.cross(u,np.cross(u,v)+w*v)+poses[:,None,:3]
        boxes=np.stack((rotated.min(axis=1),rotated.max(axis=1)),axis=1).reshape(-1,3)
        self.bounds.assign(boxes.astype(np.float32));self.directions.assign(np.asarray(directions,dtype=np.float32))
        uploaded=time.perf_counter()
        wp.launch(_cast,dim=self.beam_count,inputs=[self.mesh.id,wp.vec3(*origin),self.directions,self.bounds,
                  len(self.corners),self.planes,len(self.planes),self.ranges,self.fallback,self.diagnostic,self.faces],device=self.device)
        values=self.ranges.numpy();fallback=self.fallback.numpy().astype(bool)
        queried=time.perf_counter()
        if fallback.any():values[fallback]=cpu_ranges(query,origin,directions[fallback],exclude_robot)
        values[values<=.4]=-1.
        ended=time.perf_counter()
        self.last={'upload_ms':1000*(uploaded-started),'cuda_query_download_ms':1000*(queried-uploaded),
                   'physx_fallback_ms':1000*(ended-queried),'total_ms':1000*(ended-started),
                   'fallback_rays':int(fallback.sum()),'rays':self.beam_count}
        return values


def compare_ranges(reference,actual,tolerance=.002):
    a=reference>0;b=actual>0;common=a&b
    differences=np.abs(reference[common]-actual[common])
    return {'rays':len(reference),'hit_mismatches':int(np.sum(a!=b)),
            'distance_mismatches':int(np.sum(differences>tolerance)),
            'max_error_m':float(differences.max()) if len(differences) else 0.,
            'p99_error_m':float(np.quantile(differences,.99)) if len(differences) else 0.,
            'tolerance_m':tolerance,'passed':bool(np.array_equal(a,b) and np.all(differences<=tolerance))}


class VerifiedCudaLidar(CudaLidar):
    """Runtime reference audit; any mismatch disables acceleration for this run.

    The edge packet is conservative, not a proof of PhysX cooking equivalence.
    Check the full first frame and every 50th frame, plus a rotating 1024-ray
    sample on intermediate frames. No evaluator/map information crosses the
    policy boundary. New scenes retain the original backend by default.
    """
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.scans=0;self.disabled=False
        self.info.update(audit_rays=0,audit_failures=0,disabled=False,
                         audit_policy='full first/every 50; rotating 1024 other scans')

    def scan(self,query,origin,directions,exclude_robot=True):
        if self.disabled:return cpu_ranges(query,origin,directions,exclude_robot)
        values=super().scan(query,origin,directions,exclude_robot)
        if self.scans%50==0:
            indices=np.arange(self.beam_count)
        else:
            indices=(np.arange(min(1024,self.beam_count))+self.scans*1024)%self.beam_count
        reference=cpu_ranges(query,origin,directions[indices],exclude_robot)
        check=compare_ranges(reference,values[indices]);self.scans+=1
        self.info['audit_rays']+=len(indices)
        if not check['passed']:
            self.disabled=True;self.info.update(disabled=True,audit_failures=self.info['audit_failures']+1,
                                               first_failure={'scan':self.scans,**check})
            return cpu_ranges(query,origin,directions,exclude_robot)
        return values
