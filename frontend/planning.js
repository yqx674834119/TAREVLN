'use strict';
// All geometry comes from recorded ROS messages in a single odometry frame.
const plannerView = (() => {
  let key='',snapshot=null,errorText='';
  const checked=id=>document.getElementById(id)?.checked;
  function draw(data,frame){
    const note=document.getElementById('planningState');
    if(!data?.planning_available){note.textContent='此记录没有原生 TARE 图层；仅显示实际轨迹。';return false;}
    const next=data.id+'@'+(frame?.t??0);
    if(key!==next){
      key=next;snapshot=null;errorText='';
      api('/api/planning?id='+encodeURIComponent(data.id)+'&time='+(frame?.t??0)).then(value=>{
        if(key!==next)return;snapshot=value;draw(data,frame);
      }).catch(error=>{if(key===next){errorText=error.message;draw(data,frame);}});
    }
    const canvas=document.getElementById('map'),ctx=canvas.getContext('2d');
    ctx.clearRect(0,0,canvas.width,canvas.height);ctx.fillStyle='#f8faff';ctx.fillRect(0,0,canvas.width,canvas.height);
    if(!snapshot?.available){
      note.textContent=errorText||snapshot?.reason||'读取当前时刻的原生规划…';
      ctx.fillStyle='#8191a6';ctx.font='16px sans-serif';ctx.fillText(note.textContent,18,40);return true;
    }
    const layers=snapshot.layers,robot=snapshot.robot.position;
    const fresh=layer=>layer&&snapshot.time-layer.time<=3;
    const layerPoints=k=>fresh(layers[k])?layers[k].points:[];
    const free=layerPoints('free'),candidates=layerPoints('candidates'),obstacles=layerPoints('obstacles');
    const room=snapshot.room_mask;
    const roomBounds=room?[room.origin[0],room.origin[0]+room.width*room.resolution[0],
                            room.origin[1],room.origin[1]+room.height*room.resolution[1]]:null;
    const extent=checked('planFollow')?[robot[0]-6,robot[0]+6,robot[1]-6,robot[1]+6]:(()=>{
      const points=[robot,...snapshot.trail,...free,...candidates];
      const xs=points.map(p=>p[0]),ys=points.map(p=>p[1]);
      if(roomBounds){xs.push(roomBounds[0],roomBounds[1]);ys.push(roomBounds[2],roomBounds[3]);}
      return [Math.min(...xs)-1,Math.max(...xs)+1,Math.min(...ys)-1,Math.max(...ys)+1];
    })();
    const scale=Math.min((canvas.width-48)/Math.max(extent[1]-extent[0],2),(canvas.height-45)/Math.max(extent[3]-extent[2],2));
    const cx=(extent[0]+extent[1])/2,cy=(extent[2]+extent[3])/2;
    const xy=p=>[canvas.width/2+(p[0]-cx)*scale,canvas.height/2-(p[1]-cy)*scale];
    ctx.strokeStyle='#e0e7ef';ctx.lineWidth=1;
    for(let x=Math.floor(extent[0]);x<=extent[1];x++){const u=xy([x,cy])[0];ctx.beginPath();ctx.moveTo(u,0);ctx.lineTo(u,canvas.height);ctx.stroke();}
    for(let y=Math.floor(extent[2]);y<=extent[3];y++){const v=xy([cx,y])[1];ctx.beginPath();ctx.moveTo(0,v);ctx.lineTo(canvas.width,v);ctx.stroke();}
    function cloud(points,color,size,groundOnly=true){ctx.fillStyle=color;for(const p of points){if(groundOnly&&Math.abs(p[2]-robot[2])>.8)continue;const [x,y]=xy(p);ctx.fillRect(x-size/2,y-size/2,size,size);}}
    function line(points,color,width,dash=[]){if(!points.length)return;ctx.beginPath();ctx.strokeStyle=color;ctx.lineWidth=width;ctx.setLineDash(dash);points.forEach((p,i)=>{const [x,y]=xy(p);i?ctx.lineTo(x,y):ctx.moveTo(x,y);});ctx.stroke();ctx.setLineDash([]);}
    function roomColor(id){return `hsl(${(Math.abs(Number(id))*53)%360} 58% 66%)`;}
    function drawRooms(){
      if(!checked('planRooms')||!room?.ids?.length)return;
      const rx=room.resolution[0],ry=room.resolution[1];
      for(let i=0;i<room.height;i++)for(let j=0;j<room.width;j++){
        const id=Number(room.ids[i]?.[j]||0);if(id<=0)continue;
        const a=xy([room.origin[0]+i*rx,room.origin[1]+j*ry]);
        const b=xy([room.origin[0]+(i+1)*rx,room.origin[1]+(j+1)*ry]);
        ctx.fillStyle=roomColor(id);ctx.globalAlpha=.28;
        ctx.fillRect(Math.min(a[0],b[0]),Math.min(a[1],b[1]),Math.max(1,Math.abs(b[0]-a[0])),Math.max(1,Math.abs(b[1]-a[1])));
      }
      ctx.globalAlpha=1;
      for(const item of (snapshot.rooms||[])){
        if(!Array.isArray(item.centroid)||item.centroid.length<2)continue;
        const [x,y]=xy(item.centroid);const current=Number(item.id)===Number(snapshot.current_room_id);
        ctx.font=current?'bold 13px sans-serif':'12px sans-serif';ctx.fillStyle=current?'#9d174d':'#7a245c';
        ctx.strokeStyle='#fff';ctx.lineWidth=3;const text=`R${item.id}${item.label?' · '+item.label:''}${current?' · 当前':''}`;
        ctx.strokeText(text,x+5,y-5);ctx.fillText(text,x+5,y-5);
      }
    }
    drawRooms();
    if(checked('planArea')){cloud(free,'#79d6b3',Math.max(2,.19*scale));cloud(candidates,'#1fa877',Math.max(2,.12*scale));cloud(layerPoints('free_paths'),'#bce6f1',2);}
    if(checked('planObstacles'))cloud(obstacles,'#ed8186',Math.max(2,.12*scale));
    line(snapshot.trail,'#3978d8',2);
    if(checked('planPath')){line(layerPoints('coverage_path'),'#94a3b8',1,[4,3]);line(layerPoints('path'),'#7b49d5',3);}
    if(checked('planWaypoints')){
      cloud(snapshot.waypoint_history,'#f9c45d',3,false);
      const wp=layerPoints('waypoint')[0];
      if(wp){const [x,y]=xy(wp);ctx.fillStyle='#e5a01b';ctx.strokeStyle='#815700';ctx.lineWidth=1.5;ctx.beginPath();
        for(let i=0;i<10;i++){const a=i*Math.PI/5-Math.PI/2,r=i%2?4:10;i?ctx.lineTo(x+Math.cos(a)*r,y+Math.sin(a)*r):ctx.moveTo(x+Math.cos(a)*r,y+Math.sin(a)*r);}
        ctx.closePath();ctx.fill();ctx.stroke();ctx.font='13px sans-serif';ctx.fillStyle='#815700';ctx.fillText('waypoint',x+13,y-6);}
    }
    const [rx,ry]=xy(robot),yaw=snapshot.robot.yaw;
    ctx.save();ctx.translate(rx,ry);ctx.rotate(-yaw);ctx.fillStyle='#123c68';ctx.beginPath();ctx.moveTo(10,0);ctx.lineTo(-6,-5);ctx.lineTo(-6,5);ctx.closePath();ctx.fill();ctx.restore();
    ctx.fillStyle='#64748b';ctx.font='13px sans-serif';ctx.fillText('odom / map · 1 m 网格 · 当前层',12,canvas.height-10);
    note.textContent=`规划 ${snapshot.time.toFixed(1)} s · 可通行采样 ${fresh(layers.free)?free.length:'无数据'} · 房间 Mask ${room?.nonzero_ids?.length?room.nonzero_ids.map(id=>'R'+id).join('、'):'无数据'} · 候选视点 ${fresh(layers.candidates)?candidates.length:'无数据'} · 障碍点 ${fresh(layers.obstacles)?obstacles.length:'无数据'} · waypoint ${layerPoints('waypoint').length?'已发布':'等待'}。房间来源 /room_mask；未知区域不填色。`;
    return true;
  }
  document.addEventListener('DOMContentLoaded',()=>{for(const id of ['planArea','planRooms','planObstacles','planPath','planWaypoints','planFollow'])document.getElementById(id).addEventListener('change',()=>drawMap(state.data,state.data?.frames[state.index]));});
  return {draw};
})();
