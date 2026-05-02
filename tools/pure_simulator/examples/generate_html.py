#!/usr/bin/env python3
import json

with open('/tmp/sim_comparison_data.json', 'r') as f:
    data = json.load(f)

config = data['config']
baseline_log = data['baseline']['log']
ttl_log = data['ttl']['log']

max_time = max(max(e['finish_time'] for e in baseline_log), max(e['finish_time'] for e in ttl_log))
pixels_per_second = 100

def hit_class(hit_type):
    if 'Miss' in hit_type:
        return 'hit-miss'
    elif hit_type == 'TTL':
        return 'hit-ttl'
    elif hit_type == 'L1':
        return 'hit-l1'
    else:
        return 'hit-l0'

html = f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Baseline vs TTL Timeline Comparison</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #0f1419; color: #e7e9ea; padding: 20px; }}
        h1 {{ color: #1d9bf0; margin-bottom: 10px; font-size: 1.5em; }}
        h2 {{ color: #71767b; margin: 20px 0 10px; font-size: 1.1em; }}
        .info {{ color: #71767b; margin-bottom: 20px; font-size: 0.9em; }}
        
        .comparison-header {{
            display: flex;
            gap: 40px;
            margin: 20px 0;
            align-items: stretch;
        }}
        .strategy-box {{
            background: #16181c;
            border-radius: 12px;
            padding: 20px;
            flex: 1;
            text-align: center;
        }}
        .strategy-box.baseline {{ border-left: 4px solid #f4212e; }}
        .strategy-box.ttl {{ border-left: 4px solid #00ba7c; }}
        .strategy-name {{ font-size: 1.2em; font-weight: 700; margin-bottom: 10px; }}
        .strategy-box.baseline .strategy-name {{ color: #f4212e; }}
        .strategy-box.ttl .strategy-name {{ color: #00ba7c; }}
        .strategy-jct {{ font-size: 2em; font-weight: 700; color: #e7e9ea; }}
        .strategy-label {{ color: #71767b; font-size: 0.8em; margin-top: 4px; }}
        .speedup-box {{
            background: #1d9bf0;
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }}
        .speedup-value {{ font-size: 2.5em; font-weight: 700; color: #fff; }}
        .speedup-label {{ color: rgba(255,255,255,0.8); font-size: 0.85em; }}
        
        .program-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.85em;
            background: #16181c;
            border-radius: 8px;
            overflow: hidden;
            margin: 20px 0;
        }}
        .program-table th {{
            background: #2f3336;
            color: #71767b;
            text-align: left;
            padding: 12px;
            font-weight: 600;
        }}
        .program-table td {{
            padding: 10px 12px;
            border-bottom: 1px solid #2f3336;
        }}
        .program-table tr:last-child td {{ border-bottom: none; }}
        .speedup-cell {{ font-weight: 700; }}
        .speedup-cell.fast {{ color: #00ba7c; }}
        .speedup-cell.slow {{ color: #f4212e; }}
        
        .timeline-container {{
            background: #16181c;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 30px;
            overflow-x: auto;
        }}
        .strategy-label-row {{
            display: flex;
            align-items: center;
            margin-bottom: 10px;
        }}
        .strategy-label-text {{
            font-weight: 700;
            font-size: 1em;
            width: 80px;
            flex-shrink: 0;
        }}
        .strategy-label-text.baseline {{ color: #f4212e; }}
        .strategy-label-text.ttl {{ color: #00ba7c; }}
        
        .time-scale {{
            display: flex;
            font-size: 0.7em;
            color: #71767b;
            min-width: {int(max_time) * pixels_per_second}px;
            border-bottom: 1px solid #3f4346;
            padding-bottom: 5px;
            margin-left: 80px;
        }}
        .time-mark {{
            flex: 1;
            text-align: center;
            border-left: 1px solid #3f4346;
            padding-left: 2px;
        }}
        
        .timeline-rows-wrapper {{ margin-left: 80px; }}
        .timeline-rows {{ position: relative; min-width: {int(max_time) * pixels_per_second}px; }}
        
        .timeline-row {{
            position: relative;
            height: 36px;
            margin-bottom: 2px;
            background: #2f3336;
            border-radius: 4px;
        }}
        .timeline-row:nth-child(odd) {{ background: #282d32; }}
        
        .request-block {{
            position: absolute;
            height: 28px;
            top: 4px;
            border-radius: 4px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.55em;
            font-weight: 700;
            color: #fff;
            cursor: pointer;
            transition: transform 0.15s, box-shadow 0.15s;
            text-shadow: 0 1px 2px rgba(0,0,0,0.3);
            overflow: hidden;
            white-space: nowrap;
        }}
        .request-block:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.4);
            z-index: 100;
        }}
        
        .hit-miss {{ background: linear-gradient(135deg, #f4212e, #d91a28); }}
        .hit-ttl {{ background: linear-gradient(135deg, #00ba7c, #00a06d); }}
        .hit-l1 {{ background: linear-gradient(135deg, #794bc4, #6840b0); }}
        .hit-l0 {{ background: linear-gradient(135deg, #f7931a, #e08815); }}
        
        .tooltip {{
            position: fixed;
            background: #2f3336;
            border: 1px solid #3f4346;
            border-radius: 8px;
            padding: 12px;
            font-size: 0.85em;
            max-width: 350px;
            z-index: 1000;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            display: none;
        }}
        .tooltip.show {{ display: block; }}
        .tooltip-title {{ color: #1d9bf0; font-weight: 600; margin-bottom: 8px; }}
        .tooltip-row {{ display: flex; justify-content: space-between; margin: 4px 0; }}
        .tooltip-label {{ color: #71767b; }}
        .tooltip-value {{ color: #e7e9ea; }}
        
        .legend {{ display: flex; gap: 20px; margin: 15px 0; font-size: 0.85em; }}
        .legend-item {{ display: flex; align-items: center; gap: 6px; }}
        .legend-color {{ width: 16px; height: 16px; border-radius: 3px; }}
    </style>
</head>
<body>
    <h1>Baseline vs TTL Strategy Comparison</h1>
    <div class="info">
        <strong>Config:</strong> cache={config['cache']}, progs={config['programs']}, turns={config['turns']}, TTL={config['ttl_min']}-{config['ttl_max']}s<br>
        <span style="color: #71767b; font-size: 0.85em;">← Scroll horizontally to see all requests →</span>
    </div>
    
    <div class="comparison-header">
        <div class="strategy-box baseline">
            <div class="strategy-name">Baseline</div>
            <div class="strategy-jct">{data['baseline']['avg_jct']:.0f}ms</div>
            <div class="strategy-label">Avg JCT</div>
        </div>
        <div class="speedup-box">
            <div class="speedup-value">{data['speedup']:.2f}x</div>
            <div class="speedup-label">TTL Speedup</div>
        </div>
        <div class="strategy-box ttl">
            <div class="strategy-name">TTL</div>
            <div class="strategy-jct">{data['ttl']['avg_jct']:.0f}ms</div>
            <div class="strategy-label">Avg JCT</div>
        </div>
    </div>
    
    <h2>Per-Program JCT Comparison</h2>
    <table class="program-table">
        <thead>
            <tr>
                <th>Program</th>
                <th>Baseline JCT (ms)</th>
                <th>TTL JCT (ms)</th>
                <th>Speedup</th>
            </tr>
        </thead>
        <tbody>
'''

for prog in sorted(data['baseline']['program_jcts'].keys()):
    b_jct = data['baseline']['program_jcts'][prog]
    t_jct = data['ttl']['program_jcts'][prog]
    speedup = b_jct / t_jct if t_jct > 0 else 0
    cell_class = 'fast' if speedup > 1.0 else 'slow' if speedup < 1.0 else 'neutral'
    html += f'''            <tr>
                <td><strong>{prog}</strong></td>
                <td>{b_jct:.1f}</td>
                <td>{t_jct:.1f}</td>
                <td class="speedup-cell {cell_class}">{speedup:.2f}x</td>
            </tr>
'''

html += '''        </tbody>
    </table>
    
    <div class="legend">
        <div class="legend-item">
            <div class="legend-color hit-miss"></div>
            <span>Miss</span>
        </div>
        <div class="legend-item">
            <div class="legend-color hit-ttl"></div>
            <span>TTL Hit</span>
        </div>
        <div class="legend-item">
            <div class="legend-color hit-l1"></div>
            <span>L1 Hit</span>
        </div>
        <div class="legend-item">
            <div class="legend-color hit-l0"></div>
            <span>L0 Hit</span>
        </div>
    </div>
    
    <h2>Baseline Timeline</h2>
    <div class="timeline-container">
        <div class="strategy-label-row">
            <div class="strategy-label-text baseline">Baseline</div>
            <div class="time-scale">
'''

num_marks = int(max_time) + 1
for i in range(num_marks):
    html += f'<div class="time-mark">{i}s</div>'

html += '''            </div>
        </div>
        <div class="timeline-rows-wrapper">
            <div class="timeline-rows">
                <div class="timeline-row">
'''

for e in baseline_log:
    left = e['start_time'] * pixels_per_second
    width = (e['finish_time'] - e['start_time']) * pixels_per_second
    rid_short = e['rid'].replace('_T', '_')
    cls = hit_class(e['hit_type'])
    html += f'                    <div class="request-block {cls}" style="left: {left:.1f}px; width: {width:.1f}px;" title="{e["rid"]}: {e["hit_type"]}">{rid_short}</div>\n'

html += '''                </div>
            </div>
        </div>
    </div>
    
    <h2>TTL Timeline</h2>
    <div class="timeline-container">
        <div class="strategy-label-row">
            <div class="strategy-label-text ttl">TTL</div>
            <div class="time-scale">
'''

for i in range(num_marks):
    html += f'<div class="time-mark">{i}s</div>'

html += '''            </div>
        </div>
        <div class="timeline-rows-wrapper">
            <div class="timeline-rows">
                <div class="timeline-row">
'''

for e in ttl_log:
    left = e['start_time'] * pixels_per_second
    width = (e['finish_time'] - e['start_time']) * pixels_per_second
    rid_short = e['rid'].replace('_T', '_')
    cls = hit_class(e['hit_type'])
    pinned = e.get('is_pinned', False)
    html += f'                    <div class="request-block {cls}" style="left: {left:.1f}px; width: {width:.1f}px;" title="{e["rid"]}: {e["hit_type"]}, pinned={pinned}">{rid_short}</div>\n'

html += '''                </div>
            </div>
        </div>
    </div>
    
    <div class="tooltip" id="tooltip">
        <div class="tooltip-title" id="tt-title">Request</div>
        <div class="tooltip-row"><span class="tooltip-label">Hit:</span><span class="tooltip-value" id="tt-hit"></span></div>
    </div>
    
    <script>
        const tooltip = document.getElementById('tooltip');
        const blocks = document.querySelectorAll('.request-block');
        blocks.forEach(block => {
            block.addEventListener('mouseenter', (e) => {
                document.getElementById('tt-title').textContent = block.getAttribute('title').split(':')[0];
                const title = block.getAttribute('title');
                let hitType = 'Miss';
                if (title.includes('TTL')) hitType = 'TTL Hit';
                else if (title.includes('L1')) hitType = 'L1 Hit';
                else if (title.includes('L0')) hitType = 'L0 Hit';
                document.getElementById('tt-hit').textContent = hitType;
                tooltip.classList.add('show');
            });
            block.addEventListener('mousemove', (e) => {
                tooltip.style.left = (e.clientX + 15) + 'px';
                tooltip.style.top = (e.clientY + 15) + 'px';
            });
            block.addEventListener('mouseleave', () => {
                tooltip.classList.remove('show');
            });
        });
    </script>
</body>
</html>'''

output_path = '/home/comp/csgfyu/multi-agents/KVBlocking/sglang_continuum/tools/pure_simulator/examples/new_comparison.html'
with open(output_path, 'w') as f:
    f.write(html)

print(f'Generated: {output_path}')
print(f'Max time: {max_time:.1f}s')
print(f'Baseline: {len(baseline_log)} requests')
print(f'TTL: {len(ttl_log)} requests')
