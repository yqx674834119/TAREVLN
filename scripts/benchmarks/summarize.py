"""Build an evidence report from completed, frozen runs; never drive policies."""
import gzip
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
PROJECT = ROOT/'cross-building-floor-nav'
RUNS = ROOT/'outputs/benchmarks'
ASSETS = ROOT/'assets/benchmarks'
DOCS = PROJECT/'docs/benchmarks'


def read(path):
    return json.loads(path.read_text())


def mean(rows, key):
    return float(np.mean([r[key] for r in rows]))


def main():
    control = read(RUNS/'mp3d-scene-102/results.json')
    vlm = read(RUNS/'mp3d-vlm-001/results.json') + read(RUNS/'mp3d-vlm-002/results.json')
    assert len(vlm) == 10 and len({r['episode_id'] for r in vlm}) == 10
    selected = {r['episode_id'] for r in vlm}
    summary = {'controls_102': read(RUNS/'mp3d-scene-102/summary.json'), 'matched_pilot_10': {}}
    for name, rows in [('stop',[r for r in control if r['policy']=='stop' and r['episode_id'] in selected]),
                       ('random',[r for r in control if r['policy']=='random' and r['episode_id'] in selected]),
                       ('handcrafted',[r for r in control if r['policy']=='handcrafted' and r['episode_id'] in selected]),
                       ('oracle_privileged',[r for r in control if r['policy']=='oracle' and r['episode_id'] in selected]),
                       ('vlm',vlm)]:
        summary['matched_pilot_10'][name] = dict(n=len(rows), **{k:mean(rows,k) for k in ['sr','osr','spl','ne_m','ndtw_fast','sdtw_fast','steps','collisions']})
    summary['vlm_episodes'] = vlm
    summary['scope'] = 'One scene, 10 instruction episodes, four unique routes; development pilot, not full val_unseen or paper reproduction'
    summary['geodesic_max_delta_m'] = max(abs(r['preflight']['geodesic_delta_m']) for r in control)
    summary['policy_call_cap'] = 130
    summary['download_snapshot'] = {}
    for name in ['mp3d','etpnav']:
        d = read(ASSETS/(name+'-download-status.json'))
        summary['download_snapshot'][name] = dict(completed=d['completed'], total=d['total'],
            bytes=sum(r.get('bytes',0) for r in d['files']), failures=[r for r in d['files'] if r['status']!='verified'])
    summary['environment'] = dict(python=platform.python_version())
    for package in ['habitat-sim','numpy','fastdtw','Pillow']:
        try:
            summary['environment'][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            summary['environment'][package] = 'available via inherited base; metadata unavailable'
    summary['report_generation_source_sha256'] = {str(p.relative_to(PROJECT)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (PROJECT/'scripts/benchmarks').glob('*.py')}
    summary['project_base_commit'] = subprocess.check_output(['git','-C',str(PROJECT),'rev-parse','HEAD'],text=True).strip()
    summary['source_capture_note'] = 'Pilot retained protocol manifests, policy requests and results; script hashes here describe report-generation code after telemetry additions. Future console runs archive runner/metrics/worker/API source before simulation.'
    summary['vlm_usage'] = dict(calls=0, prompt_tokens=0, completion_tokens=0, api_seconds=0)
    for run in ['mp3d-vlm-001','mp3d-vlm-002']:
        for path in (RUNS/run/'vlm').glob('*/policy/call-*/response.json'):
            record = read(path)
            summary['vlm_usage']['calls'] += 1
            for key in ['prompt_tokens','completion_tokens']:
                summary['vlm_usage'][key] += record.get('usage',{}).get(key,0)
            summary['vlm_usage']['api_seconds'] += record.get('latency_seconds',0)
    (DOCS/'first-scene-results.json').write_text(json.dumps(summary, indent=2))

    md = ['# 第一场景实际评测结果', '',
          '场景 `x8F5xyUWy9e`，R2R-CE v1-3 val_unseen。Habitat-Sim 0.3.3 移植评测器；未运行 Isaac 或机器狗。', '',
          '## 场景与评测器检查', '',
          '102 条任务覆盖 34 条独立路线。四组诊断共 408 次 episode；全部起点/终点在 navmesh 上，全部初始 geodesic distance 与任务文件一致（最大差 0m）。', '',
          '| 策略 | N | SR % | OSR % | SPL % | NE m | nDTW % |',
          '|---|---:|---:|---:|---:|---:|---:|']
    for name, r in summary['controls_102'].items():
        md.append(f"| {name}{'（特权诊断）' if name=='oracle' else ''} | {r['n']} | {100*r['sr']:.2f} | {100*r['osr']:.2f} | {100*r['spl']:.2f} | {r['ne_m']:.3f} | {100*r['ndtw_fast']:.2f} |")
    md += ['', 'oracle 知道目标坐标和 navmesh，只用于验证任务可执行；其 100% 不是自主 VLN 性能。由于成功区域为 3m 内，路径短于到目标中心的最短距离时 SPL 仍可为 1。', '',
           '## 固定前 10 条指令的配对 pilot', '',
           '选取 episode 7、8、9、31、32、33、100、101、102、187，4 条独立路线。未依据结果剔除任务，预算耗尽计失败。每条最多 500 原子动作、130 次 VLM 请求。', '',
           '| 策略 | N | SR % | OSR % | SPL % | NE m | nDTW % |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for name, r in summary['matched_pilot_10'].items():
        md.append(f"| {name} | {r['n']} | {100*r['sr']:.1f} | {100*r['osr']:.1f} | {100*r['spl']:.1f} | {r['ne_m']:.3f} | {100*r['ndtw_fast']:.1f} |")
    md += ['', '单场景、少量路线的 pilot 不支持泛化排名。模型服务未固定不可变版本，temperature=0 也不保证请求结果完全确定。', '',
           '| Episode | SR | NE m | 原子动作 | VLM 决策 | 碰撞动作 | 结束原因 |',
           '|---|---:|---:|---:|---:|---:|---|']
    for r in vlm:
        reason = r['termination']
        if reason=='runtime_error' and r['decisions']==130:
            reason += '（130 次策略请求预算耗尽）'
        md.append(f"| {r['episode_id']} | {int(r['sr'])} | {r['ne_m']:.3f} | {r['steps']} | {r['decisions']} | {r['collisions']} | {reason} |")
    md += ['', '## 可观察的问题', '',
           '- 第 7 条在错误的吧台位置 STOP，NE=4.179m，失败；nDTW=0.655，不能用“走得像”代替终点成功。',
           '- 第 8 条在吧台/柜子附近反复左右转向，耗尽 130 次请求，按失败保留。',
           '- 第 32 条成功，NE=2.914m，但花费 198 个动作；到达成功区域不等于路径高效。',
           '- 当前策略没有可达候选图和明确的反复转向恢复，说明需要把 TARE 的几何可达性与任务进度记忆接入，而不是仅扩大 VLM 提示词。',
           '- 当前渲染来自扫描纹理，仍有重建瑕疵；物理接触和真实传感器误差未被本轮实验覆盖。', '',
           '## 原始证据与复现', '',
           '`~/VLN/outputs/benchmarks/mp3d-scene-102/`：全部诊断任务。', '',
           '`~/VLN/outputs/benchmarks/mp3d-vlm-001/`：第 7 条 VLM。', '',
           '`~/VLN/outputs/benchmarks/mp3d-vlm-002/`：其余 9 条 VLM。', '',
           '每条含 start/end RGB、深度、完整动作/轨迹、预检和结果；VLM 另含逐次图像、请求提示、响应、时延与 token 用量。凭据不进入日志。四项指标单元测试通过，覆盖 STOP/3m 边界、路过目标、绕行惩罚与静止位置去重。', '',
           '图：`first-scene-evidence.png`；机器可读汇总与下载快照：`first-scene-results.json`。完整调研与复现命令见 `vln-survey.md`。', '']
    (DOCS/'first-scene-results.md').write_text('\n'.join(md))

    tasks = json.load(gzip.open(ASSETS/'tasks/R2R_VLNCE_v1-3/val_unseen/val_unseen.json.gz','rt'))['episodes']
    episodes = {e['episode_id']:e for e in tasks}
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), gridspec_kw={'width_ratios':[1,1,1.25]})
    for row, (eid, run) in enumerate([(7,'mp3d-vlm-001'), (32,'mp3d-vlm-002')]):
        base = RUNS/run/'vlm'/str(eid)
        result = read(base/'result.json')
        for col, key in enumerate(['start','end']):
            axes[row,col].imshow(Image.open(base/(key+'.jpg')))
            axes[row,col].axis('off')
            axes[row,col].set_title(f'Episode {eid}: {key} RGB', fontsize=11)
        trajectory = np.asarray(read(base/'trajectory.json')['positions'])
        reference = np.asarray(episodes[eid]['reference_path'])
        ax = axes[row,2]
        ax.plot(reference[:,0],reference[:,2],'--',color='#8291a5',label='Sparse reference (display only)')
        ax.plot(trajectory[:,0],trajectory[:,2],color='#3167bd',label='Executed trajectory')
        ax.scatter(*trajectory[0,[0,2]],color='#00a681',s=55,label='Start',zorder=4)
        ax.scatter(*trajectory[-1,[0,2]],color='#e46e42',s=70,marker='s',label='Stop',zorder=4)
        goal = np.asarray(episodes[eid]['goals'][0]['position'])
        ax.scatter(*goal[[0,2]],color='#9c359e',marker='*',s=140,label='Evaluator goal',zorder=4)
        ax.set_aspect('equal',adjustable='datalim')
        ax.grid(alpha=.2)
        ax.set_xlabel('World x (m)'); ax.set_ylabel('World z (m)')
        ax.set_title(f"{'SUCCESS' if result['sr'] else 'FAILURE'} | NE {result['ne_m']:.2f} m | {result['steps']} actions",fontsize=11)
        if row==0:
            ax.legend(fontsize=7,loc='best')
    fig.suptitle('MP3D / R2R-CE: RGB policy evidence and evaluator-only trajectories',fontsize=15)
    fig.tight_layout(rect=[0,0,1,.95])
    fig.savefig(DOCS/'first-scene-evidence.png',dpi=160)
    plt.close(fig)
    print(json.dumps(summary['matched_pilot_10'],indent=2))


if __name__ == '__main__':
    main()
