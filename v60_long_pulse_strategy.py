import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import engine_model_2 as engine_model

# TVS XL 100 Calibration Parameters
TRANSMISSION_EFFICIENCY = 0.95
V_AVG_TARGET = 25.05       # km/h (Minimum average speed target)
MAX_RACE_TIME = 1800.0    # 30 minutes
TARGET_TIME_BUFFER = 60.0  # 1 minute (Target 29 mins) — gives 2.1 km/h headroom below governor
IDLE_RPM = 1500.0         # RPM — centrifugal clutch engagement speed (glide stop)
EFF_RPM_LIMIT = 7000.0    # RPM — governor limit / pulse stop (most efficient RPM)
MIN_PULSE_TIME = 3.0      # s — min clutch engagement time; prevents sub-second micro-pulses
MIN_GLIDE_TIME_ADAPTIVE = 5.0  # s — min glide before adaptive pace trigger can fire; prevents chattering

def smooth(x, window_len=11, window='hanning'):
    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")
    if x.size < window_len:
        if x.size == 0:
            return x
        window_len = x.size
    if window_len<3:
        return x
    s = np.r_[x[window_len-1:0:-1], x, x[-2:-window_len-1:-1]]
    if window == 'flat': #moving average
        w = np.ones(window_len,'d')
    else:
        w = eval('np.'+window+'(window_len)')
    y = np.convolve(w/w.sum(), s, mode='valid')
    return y[(window_len//2):-(window_len//2) + (len(x) - len(y) + 2 * (window_len//2))]

def load_and_preprocess_track(filename):
    """
    Loads track coordinates and calculates curvature and slope.
    """
    df = pd.read_csv(filename, sep='\t')
    
    # Distance in meters
    distance_m = df['distance (km)'].values * 1000
    altitude = df['altitude (m)'].values
    lat = df['latitude'].values
    lon = df['longitude'].values
    
    # Compute heading and curvature
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    
    # Simple equirectangular projection for small area
    R = 6371000 # Earth radius in meters
    x = R * lon_rad * np.cos(np.mean(lat_rad))
    y = R * lat_rad
    
    dx = np.gradient(x, distance_m, edge_order=2)
    dy = np.gradient(y, distance_m, edge_order=2)
    
    heading = np.unwrap(np.arctan2(dy, dx))
    
    # Curvature (rate of change of heading per meter)
    d_heading_ds = np.gradient(heading, distance_m, edge_order=2)
    
    # Smooth curvature to avoid noise classifying small wiggles as curves
    # Instead of complicated smooth, let's use pandas rolling mean
    curvature_smoothed = pd.Series(np.abs(d_heading_ds)).rolling(window=15, center=True, min_periods=1).mean().values
    
    # Slope (radians)
    dz_ds = np.gradient(altitude, distance_m, edge_order=2)
    dz_ds_smoothed = pd.Series(dz_ds).rolling(window=15, center=True, min_periods=1).mean().values

    slope = np.arctan(dz_ds_smoothed)
    
    # Determine V_PULSE_STOP based on Efficient RPM
    # v = (RPM * 2 * pi * R) / (G * 60)
    v_pulse_stop_kmh = 60.0
    is_curve = np.zeros_like(distance_m, dtype=bool)

    # Define is_curve as neutral (False) to satisfy return structure while disabling curve capping
    is_curve = np.zeros_like(distance_m, dtype=bool)
    
    return distance_m, altitude, slope, is_curve, curvature_smoothed, v_pulse_stop_kmh

def evaluate_strategy(distance_m, slope, is_curve, v_avg_target=25.05, v_pulse_stop=35.0, num_laps=4):
    """
    Simulates the pulse and glide strategy physics using an instantaneous engine model.
    Supports either single values or per-lap list of parameters.
    """
    # Physics parameters
    mass = 140.0              # Assumed kg
    efficiency = 0.30         # 30%
    Cd = 0.5              
    A = 0.371                  # m^2
    Crr = 0.007
    

    # Constants
    rho = 1.225               # kg/m^3
    g = 9.81                  # m/s^2
    LHV = engine_model.LHV_PETROL
    R = engine_model.WHEEL_RADIUS
    G = engine_model.GEAR_RATIO
    # RPM-based velocity boundaries (physics of centrifugal clutch and governor)
    # v = (RPM * 2 * pi * R) / (G * 60)
    v_pulse_stop_mps = 60.0 / 3.6 # Hard cap at 60 km/h requested by user
    v_glide_stop_mps = (IDLE_RPM * 2 * np.pi * R) / (G * 60)       # ~11.0 km/h at 1500 RPM
    
    # Simulation settings
    dt = 0.05                 # seconds
    s_current = 0.0
    v_current = 0.0           # Starts at 0 km/hr
    t_current = 0.0
    t_last_state_change = 0.0
    
    total_length_single = distance_m[-1]
    total_length_race = total_length_single * num_laps
    
    state = "PULSE"           # Start with pulse (from rest)
    
    results = {
        'time': [],
        'distance': [],
        'velocity': [],
        'state': [],
        'slope_deg': [],
        'acceleration': [],
        'fuel_ml': [],
        'v_max_lim': [],
        'v_min_lim': [],
        'rpm': [],
        'force': [],
        'lap': [],
        'trans_lost_joules': 0.0,
        'v_avg_so_far': [],
        'fuel_joules': 0.0,
        'mechanical_joules': 0.0,
        'tractive_work_joules': 0.0
    }
    
    # Use global MAX_RACE_TIME
    
    fuel_energy_joules = 0.0
    mech_energy_joules = 0.0
    
    results['v_target_adaptive'] = [] # New tracking key

    while s_current < total_length_race and t_current < MAX_RACE_TIME:
        # Loop track data safely using modulo
        s_modulo = s_current % total_length_single
        idx = np.searchsorted(distance_m, s_modulo)
        if idx >= len(distance_m):
            idx = len(distance_m) - 1
            
        current_slope = slope[idx]
        current_in_curve = is_curve[idx]
        current_lap = int(s_current / total_length_single) + 1
        
        # --- PER-LAP STRATEGY PARAMETERS ---
        current_v_avg_target = v_avg_target
        
        # --- ADAPTIVE OPTIMIZATION (RACE MANAGER) ---
        # Recalculate required speed to finish 4 laps exactly at 2100s
        dist_remaining = total_length_race - s_current
        time_remaining = MAX_RACE_TIME - t_current
        
        if time_remaining > 5.0 and dist_remaining > 0:
            v_req_mps = dist_remaining / time_remaining
            # Clamp required speed (Never go below 25 km/h target for safety, max 45)
            # Use the current lap's optimized target as the floor
            v_req_kmh = max(current_v_avg_target, min(45.0, v_req_mps * 3.6))
            # Adaptively increase speed if we are falling behind the 30-min window
            v_req_kmh = max(current_v_avg_target, min(45.0, v_req_mps * 3.6))
        else:
            v_req_kmh = current_v_avg_target
            
        v_avg_trigger = v_req_kmh / 3.6
        # --- END ADAPTIVE OPTIMIZATION ---

        v_avg_mps = (s_current / t_current) if t_current > 0 else 0.0
        t_in_state = t_current - t_last_state_change
        
        if state == "PULSE":
            # Exit to GLIDE only when governor speed reached AND minimum engagement time met.
            # Governor force-cutoff (below) prevents overshoot during the wait.
            if v_current >= v_pulse_stop_mps and t_in_state >= MIN_PULSE_TIME:
                state = "GLIDE"
                t_last_state_change = t_current
        elif state == "GLIDE":
            # PHYSICS trigger: re-engage the moment wheel drops to idle RPM (instant, no slip)
            at_idle_rpm = v_current <= v_glide_stop_mps
            # ADAPTIVE trigger: re-engage early if behind on race pace —
            # Guard: only meaningful if vehicle is BELOW governor speed (pulse would help)
            # and the glide has lasted long enough to prevent chattering
            behind_on_pace = (
                (v_avg_mps < v_avg_trigger)
                and (v_glide_stop_mps < v_current < v_pulse_stop_mps)
                and (t_in_state >= MIN_GLIDE_TIME_ADAPTIVE)
            )
            if at_idle_rpm or behind_on_pace:
                state = "PULSE"
                t_last_state_change = t_current
        
        # Physics Forces
        f_drag = 0.5 * rho * Cd * A * (v_current ** 2)
        f_roll = Crr * mass * g * np.cos(current_slope)
        f_grade = mass * g * np.sin(current_slope)
        
        f_net = - f_drag - f_roll - f_grade
        
        # Engine RPM mirrors wheel RPM, bounded by physical limits:
        #   PULSE: engine is running → RPM >= IDLE_RPM (centrifugal clutch floor)
        #   GLIDE: engine is off    → RPM follows wheel freely (for recording only)
        wheel_rpm = (v_current * G * 60) / (2 * np.pi * R)
        if state == "PULSE":
            engine_rpm = max(IDLE_RPM, min(wheel_rpm, EFF_RPM_LIMIT))
        else:
            engine_rpm = wheel_rpm  # coasting, engine off
        
        # Get Engine Metrics at this instantaneous RPM
        m_dot, t_brake = engine_model.get_engine_metrics(engine_rpm)
        
        # Calculate Instantaneous Tractive Force (clamped >= 0: engine only pushes, never brakes)
        f_tractive_instant = max(0.0, (t_brake * G) / R)
        f_tractive_wheel = f_tractive_instant * TRANSMISSION_EFFICIENCY  # 5% gearbox loss
        
        if state == "PULSE":
            # GOVERNOR HARD-CUTOFF: no force or fuel above EFF_RPM_LIMIT
            # (engine cuts fuel when governor activates; wheel freewheels above this speed)
            if wheel_rpm < EFF_RPM_LIMIT:
                f_net += f_tractive_wheel
                fuel_energy_joules += m_dot * LHV * dt
                mech_energy_joules += f_tractive_wheel * v_current * dt
                results['tractive_work_joules'] += f_tractive_wheel * (v_current * dt)
                # Record transmission losses (5%)
                trans_loss = (f_tractive_instant - f_tractive_wheel) * v_current * dt
                results['trans_lost_joules'] += trans_loss
            
        # Kinematic update
        acceleration = f_net / mass
        v_next = v_current + acceleration * dt
        
        # Assume brakes apply if car is about to roll backwards significantly?
        # Actually in eco-marathon it might just stop and roll. We limit v >= 0
        if v_next < 0:
            v_next = 0
            
        s_next = s_current + v_current * dt
        
        # Save data every 0.1 seconds to dramatically improve phase boundary accuracy
        if int(t_current / dt) % 2 == 0:
            results['time'].append(t_current)
            results['distance'].append(s_current)
            results['velocity'].append(v_current)
            results['state'].append(1 if state == "PULSE" else 0)
            results['slope_deg'].append(np.degrees(current_slope))
            results['acceleration'].append(acceleration)
            results['fuel_ml'].append((fuel_energy_joules / 1e6 / 34.2) * 1000)
            results['v_max_lim'].append(v_pulse_stop_mps * 3.6)
            results['v_min_lim'].append(v_avg_trigger * 3.6)
            results['v_avg_so_far'].append(v_avg_mps * 3.6)
            results['rpm'].append(engine_rpm)
            results['force'].append(f_tractive_instant if state == "PULSE" else 0.0)
            results['lap'].append(current_lap)
            results['v_target_adaptive'].append(v_req_kmh)
            
        v_current = v_next
        s_current = s_next
        t_current += dt
        
        # Check for stalling completely to avoid infinite loop
        if v_current == 0 and acceleration <= 0 and state == "GLIDE":
             # We stalled in glide and won't transition to PULSE because of being in curve?
             # Check if we are in curve. If yes, car might actually get stuck.
             if current_in_curve:
                 # It's stuck in a corner!
                 # In real life, driver would have to break rule or push. Let's break early.
                 print(f"Warning: Vehicle stalled in a curve at distance {s_current:.1f} m.")
                 break
        
    # Calculate Average Tractive Force during Pulse
    pulse_forces = [f for f, s in zip(results['force'], results['state']) if s == 1]
    results['avg_tractive_force'] = np.mean(pulse_forces) if pulse_forces else 0.0
    
    results['fuel_joules'] = fuel_energy_joules
    results['mechanical_joules'] = mech_energy_joules
    results['final_time'] = t_current
    
    return results


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    track_file = os.path.join(script_dir, "sem_apme_2025-track_coordinates.csv")
    
    print("Loading track geometry...")
    distance_m, altitude, slope, is_curve, curvature, v_pulse_stop_limit = load_and_preprocess_track(track_file)
    
    # CALCULATE TARGETS BASED ON PHYSICAL CONSTRAINTS
    num_laps = 4
    total_race_dist_km = (distance_m[-1] * num_laps) / 1000.0
    
    # Completion Time Target: 30 minutes minus buffer
    target_time_s = MAX_RACE_TIME - TARGET_TIME_BUFFER
    v_avg_required_kmh = (total_race_dist_km / (target_time_s / 3600.0))
    
    print("\n" + "="*50)
    print("DERIVED PHYSICAL STRATEGY TARGETS")
    print("="*50)
    print(f"Total Race Distance:    {total_race_dist_km:.3f} km")
    print(f"Race Time Target:       {target_time_s/60:.2f} mins (incl. buffer)")
    print(f"Req. Average Speed:     {v_avg_required_kmh:.2f} km/h")
    print(f"Max Efficient RPM:      {EFF_RPM_LIMIT} RPM")
    print(f"RPM-Based Speed Cap:    {v_pulse_stop_limit:.2f} km/h")
    print("="*50)

    print("\nSimulating physical strategy...")
    res = evaluate_strategy(distance_m, slope, is_curve, v_avg_target=v_avg_required_kmh, v_pulse_stop=v_pulse_stop_limit, num_laps=num_laps)
    
    actual_distance_m = res['distance'][-1] if len(res['distance']) > 0 else 0.0
    total_race_dist_m = distance_m[-1] * num_laps
    avg_speed = (actual_distance_m / res['final_time']) * 3.6 if res['final_time'] > 0 else 0

    # Calculate volumetric fuel (assuming Petrol ~34.2 MJ / Liter)
    mj_total = res['fuel_joules'] / 1e6
    petrol_liters = mj_total / 34.2
    petrol_ml = petrol_liters * 1000
    
    # Calculate per-lap statistics
    lap_stats = []
    t_arr = np.array(res['time'])
    d_arr = np.array(res['distance'])
    f_arr = np.array(res['fuel_ml'])
    for lap_i in range(1, num_laps + 1):
        mask = np.array(res['lap']) == lap_i
        if not np.any(mask): continue
        l_time = t_arr[mask][-1] - t_arr[mask][0]
        l_fuel = f_arr[mask][-1] - f_arr[mask][0]
        l_dist = d_arr[mask][-1] - d_arr[mask][0]
        l_avg = (l_dist / l_time) * 3.6 if l_time > 0 else 0
        l_kml = (l_dist / 1000.0) / (l_fuel / 1000.0) if l_fuel > 0 else 0
        lap_stats.append((lap_i, l_time, l_fuel, l_avg, l_kml))
    
    baseline_fuel = 12.90 * num_laps # Based on first unoptimized run
    savings = baseline_fuel - petrol_ml
    savings_pct = (savings / baseline_fuel) * 100 if baseline_fuel > 0 else 0
    km_per_liter = (actual_distance_m / 1000.0) / (petrol_ml / 1000.0) if petrol_ml > 0 else 0
    
    # Export data for HTML animation
    # Subsample to keep JSON simple (~1 frame per 0.2s)
    import json
    # Map back to track x,y if we can, but simplest is to just use distances
    # Since we have lat/lon in the CSV, let's roughly get them back
    # Or just use the original x,y. Let's re-calculate x,y from long/lat:
    import pandas as pd
    df = pd.read_csv(track_file, sep='\t')
    lat = df['latitude'].values
    lon = df['longitude'].values
    R = 6371000
    lat_r = np.radians(lat)
    lon_r = np.radians(lon)
    track_x = R * lon_r * np.cos(np.mean(lat_r))
    track_y = R * lat_r
    
    # Standardize track coordinates to start roughly at 0,0
    track_x -= np.min(track_x)
    track_y -= np.min(track_y)
    
    anim_data = {
        'track': {
            'x': track_x.tolist(),
            'y': track_y.tolist(),
            's': distance_m.tolist()
        },
        'frames': []
    }
    
    # Fill frames at roughly 0.5s intervals (to keep 4-lap file size small)
    for i in range(0, len(res['time']), 10):
        s_abs = res['distance'][i]
        s_modulo = s_abs % distance_m[-1]
        
        # interpolate x,y from track_s
        idx = np.searchsorted(distance_m, s_modulo)
        if idx >= len(distance_m): idx = len(distance_m)-1
        x_val = track_x[idx]
        y_val = track_y[idx]
        
        anim_data['frames'].append({
            'x': float(x_val),
            'y': float(y_val),
            's': float(s_abs),
            'st': int(res['state'][i]),
            'lap': int(res['lap'][i]),
            'fml': float(res['fuel_ml'][i])
        })
        
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Pulse & Glide Animation</title>
    <style>
        body {{ font-family: Arial; text-align: center; background: #f0f0f0; }}
        canvas {{ background: white; border: 1px solid #ccc; margin-top: 20px; box-shadow: 0px 4px 10px rgba(0,0,0,0.1); }}
        #controls {{ margin-top: 10px; }}
        button {{ padding: 8px 16px; font-size: 14px; cursor: pointer; }}
    </style>
</head>
<body>
    <h2>Shell Eco Marathon - Pulse & Glide Strategy Track 2</h2>
    <div style="margin-bottom: 10px;">
        <span style="color:red; font-weight:bold;">● Pulse Mode</span> | 
        <span style="color:blue; font-weight:bold;">● Glide Mode</span>
    </div>
    <canvas id="trackCanvas" width="800" height="600"></canvas>
    <div id="controls">
        <button onclick="togglePlay()">Play/Pause</button>
        <button onclick="resetAnim()">Reset</button>
        <span style="margin-left:20px; font-weight:bold;" id="timeDisplay">Time: 0.0s | Dist: 0.0m</span>
    </div>
    
    <script>
        const animData = {json.dumps(anim_data)};
        
        const canvas = document.getElementById('trackCanvas');
        const ctx = canvas.getContext('2d');
        const timeDisplay = document.getElementById('timeDisplay');
        
        let trackX = animData.track.x;
        let trackY = animData.track.y;
        let frames = animData.frames;
        
        // Find bounds to fit the track
        let padding = 40;
        let minX = Math.min(...trackX), maxX = Math.max(...trackX);
        let minY = Math.min(...trackY), maxY = Math.max(...trackY);
        let rangeX = maxX - minX, rangeY = maxY - minY;
        
        // Compute scales
        let scaleX = (canvas.width - padding*2) / rangeX;
        let scaleY = (canvas.height - padding*2) / rangeY;
        let scale = Math.min(scaleX, scaleY);
        
        let offsetX = (canvas.width - (rangeX * scale)) / 2;
        let offsetY = (canvas.height - (rangeY * scale)) / 2;
        
        // Flip Y axis since canvas goes top to bottom
        function sx(rx) {{ return offsetX + (rx - minX) * scale; }}
        function sy(ry) {{ return canvas.height - (offsetY + (ry - minY) * scale); }}
        
        function drawTrackAll(limitFrame) {{
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            // Draw unvisited track as faint grey
            ctx.beginPath();
            ctx.strokeStyle = '#e0e0e0';
            ctx.lineWidth = 8;
            ctx.lineCap = 'round';
            ctx.lineJoin = 'round';
            if(frames.length > 0) {{
                ctx.moveTo(sx(frames[0].x), sy(frames[0].y));
                for(let i=1; i<frames.length; i++) {{
                    ctx.lineTo(sx(frames[i].x), sy(frames[i].y));
                }}
            }}
            ctx.stroke();
            
            // Draw colored trail based on state
            ctx.lineWidth = 8;
            for(let i=0; i<limitFrame; i++) {{
                let f1 = frames[i];
                let f2 = frames[Math.min(i+1, frames.length-1)];
                
                ctx.beginPath();
                ctx.strokeStyle = f1.st === 1 ? 'red' : 'blue';
                ctx.moveTo(sx(f1.x), sy(f1.y));
                ctx.lineTo(sx(f2.x), sy(f2.y));
                ctx.stroke();
            }}

            // Start/End dots
            if (trackX.length > 0) {{
                ctx.fillStyle = 'green';
                ctx.beginPath(); ctx.arc(sx(trackX[0]), sy(trackY[0]), 8, 0, Math.PI*2); ctx.fill();
                ctx.fillStyle = 'black';
                ctx.beginPath(); ctx.arc(sx(trackX[trackX.length-1]), sy(trackY[trackY.length-1]), 8, 0, Math.PI*2); ctx.fill();
            }}
        }}
        
        let currentFrame = 0;
        let playing = true;
        let animInterval;
        
        function drawCar() {{
            drawTrackAll(currentFrame);
            if (frames.length === 0) return;
            let f = frames[Math.min(currentFrame, frames.length - 1)];
            
            // Draw car dot - maybe simple black outline with white center showing current pos prominently
            ctx.beginPath();
            ctx.arc(sx(f.x), sy(f.y), 10, 0, Math.PI*2);
            ctx.fillStyle = 'yellow';
            ctx.fill();
            ctx.strokeStyle = 'black';
            ctx.lineWidth = 3;
            ctx.stroke();
            
            // Update HUD
            let simTime = (currentFrame * 0.5).toFixed(1);
            let economy = f.fml > 0.1 ? (f.s / 1000.0) / (f.fml / 1000.0) : 0;
            timeDisplay.textContent = `Lap: \\${{f.lap}} | Time: \\${{simTime}}s | Dist: \\${{f.s.toFixed(1)}}m | Economy: \\${{economy.toFixed(1)}} km/L`;
        }}
        
        function stepAnim() {{
            if (playing && currentFrame < frames.length) {{
                currentFrame++;
                drawCar();
                if(currentFrame >= frames.length) {{ playing = false; }}
            }}
        }}
        
        function togglePlay() {{
            playing = !playing;
        }}
        
        function resetAnim() {{
            currentFrame = 0;
            playing = true;
            drawCar();
        }}
        
        drawCar();
        animInterval = setInterval(stepAnim, 100); 
    </script>
</body>
</html>"""
    with open(os.path.join(script_dir, "animation.html"), "w", encoding='utf-8') as f:
        f.write(html_content)
        
    print("\nSaved animation data to animation.html")
    
    # Phase Segment Analysis
    t_arr = np.array(res['time'])
    d_arr = np.array(res['distance'])
    state_arr = np.array(res['state'])
    
    if len(state_arr) > 1:
        changes   = np.where(np.diff(state_arr) != 0)[0]
        boundaries = [0] + list(changes + 1) + [len(state_arr) - 1]
        v_arr_local = np.array(res['velocity']) * 3.6

        MIN_DV_REAL_PULSE = 3.0  # km/h
        total_pulse_time  = 0.0
        total_glide_time  = 0.0
        segments_data     = []

        # ── PASS 1: collect every raw segment ──────────────────────────────
        raw_segs = []
        for i in range(len(boundaries) - 1):
            si, ei = boundaries[i], boundaries[i+1]
            if si == ei:
                continue
            raw_segs.append({
                'phase':   "PULSE" if state_arr[si] == 1 else "GLIDE",
                'si': si, 'ei': ei,
                'dur':   t_arr[ei] - t_arr[si],
                'dist':  d_arr[ei] - d_arr[si],
                'v_s':   v_arr_local[si],
                'v_e':   v_arr_local[ei],
            })
            if state_arr[si] == 1:
                total_pulse_time += t_arr[ei] - t_arr[si]
            else:
                total_glide_time += t_arr[ei] - t_arr[si]

        # ── PASS 2: identify real pulses; combined glide = full gap between them ──
        real_pulses = [s for s in raw_segs
                       if s['phase'] == "PULSE" and (s['v_e'] - s['v_s']) >= MIN_DV_REAL_PULSE]
        real_engine_cycles = len(real_pulses)
        trivial = sum(1 for s in raw_segs
                      if s['phase'] == "PULSE" and (s['v_e'] - s['v_s']) < MIN_DV_REAL_PULSE)

        print("\n" + "="*40)
        print("PHASE SEGMENT ANALYSIS  (real cycles only)")
        print("="*40)
        print(f"    {'Phase':<12} | {'Time':>6} | {'Distance':>9} | Speed")
        print("    " + "-"*56)

        for n, pulse in enumerate(real_pulses):
            # ── Print this real pulse ──────────────────────────────────────
            label  = f"Pulse #{n+1}"
            strat  = f"Accelerate {pulse['v_s']:.1f} -> {pulse['v_e']:.1f} km/h"
            if pulse['v_s'] < 5.0:
                strat = f"STARTUP: 0 -> {pulse['v_e']:.1f} km/h"
            print(f"    {label:<12} | {pulse['dur']:5.1f} s | {pulse['dist']:8.1f} m | "
                  f"{pulse['v_s']:5.1f} -> {pulse['v_e']:5.1f} km/h")
            segments_data.append({
                'Phase': label, 'Instruction': strat,
                'Start Dist (m)':  d_arr[pulse['si']],
                'Dist Covered (m)': pulse['dist'],
                'Time (s)': f"{pulse['dur']:.1f}",
                'Start Speed': f"{pulse['v_s']:.1f}",
                'End Speed':   f"{pulse['v_e']:.1f}",
            })

            # ── Combined glide = from end of this pulse to start of next real pulse ──
            if n + 1 < len(real_pulses):
                next_p  = real_pulses[n + 1]
                g_dur   = t_arr[next_p['si']] - t_arr[pulse['ei']]
                g_dist  = d_arr[next_p['si']] - d_arr[pulse['ei']]
                g_v_s   = v_arr_local[pulse['ei']]   # where this pulse left off
                g_v_e   = v_arr_local[next_p['si']]  # where next pulse begins
                g_label = f"Glide #{n+1}"
                g_strat = f"Coast {g_v_s:.1f} -> {g_v_e:.1f} km/h"
                print(f"    {g_label:<12} | {g_dur:5.1f} s | {g_dist:8.1f} m | "
                      f"{g_v_s:5.1f} -> {g_v_e:5.1f} km/h")
                segments_data.append({
                    'Phase': g_label, 'Instruction': g_strat,
                    'Start Dist (m)':  d_arr[pulse['ei']],
                    'Dist Covered (m)': g_dist,
                    'Time (s)': f"{g_dur:.1f}",
                    'Start Speed': f"{g_v_s:.1f}",
                    'End Speed':   f"{g_v_e:.1f}",
                })

        print("    " + "-"*56)
        print(f"    Total Pulse Time : {total_pulse_time/60:5.2f} min  ({total_pulse_time:.1f} s)")
        print(f"    Total Glide Time : {total_glide_time/60:5.2f} min  ({total_glide_time:.1f} s)")
        print(f"    (Raw simulation  : {len([s for s in raw_segs if s['phase']=='PULSE'])} pulse segments, "
              f"{trivial} governor-bounces excluded)")
        print()
        print("=" * 58)
        print("  REALISTIC ENGINE ON/OFF CYCLES")
        print("=" * 58)
        avg_t = (total_pulse_time + total_glide_time) / real_engine_cycles if real_engine_cycles else 0
        print(f"  Total engine engagements : {real_engine_cycles}  (DeltaV >= {MIN_DV_REAL_PULSE:.0f} km/h)")
        print(f"  Per lap (~{num_laps} laps)      : ~{real_engine_cycles / num_laps:.0f} ON/OFF cycles/lap")
        print(f"  Avg cycle duration       : ~{avg_t:.0f} s  ({avg_t/60:.1f} min)")
        print("=" * 58)

        efficiency_pct = (res['tractive_work_joules'] / res['fuel_joules']) * 100 if res['fuel_joules'] > 0 else 0
        print(f"Mechanical Efficiency: {efficiency_pct:.2f}%")
        print("=" * 50)



    # Plotting 2D Track Map (X vs Y)
    try:
        from matplotlib.collections import LineCollection
        plt.figure(figsize=(10, 10))
        
        track_states = np.zeros(len(distance_m))
        for i in range(len(d_arr)-1):
            s1 = d_arr[i]
            s2 = d_arr[i+1]
            st = state_arr[i]
            
            idx1 = np.searchsorted(distance_m, s1 % distance_m[-1])
            idx2 = np.searchsorted(distance_m, s2 % distance_m[-1])
            idx1 = min(idx1, len(distance_m)-1)
            idx2 = min(idx2, len(distance_m)-1)
            
            track_states[idx1:idx2] = st
            
        points = np.array([track_x, track_y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        colors = ['red' if st == 1 else 'blue' for st in track_states[:-1]]
        
        lc = LineCollection(segments, colors=colors, linewidths=5, capstyle='round')
        ax = plt.gca()
        ax.add_collection(lc)
        ax.autoscale()
        
        plt.axis('equal')
        v_pulse_stop_kmh = 60.0
        v_glide_stop_kmh = (IDLE_RPM    * 2 * np.pi * engine_model.WHEEL_RADIUS) / (engine_model.GEAR_RATIO * 60) * 3.6
        plt.title(
            f'Pulse & Glide Track Map  |  Pulse (Red) = Engine ON  |  Glide (Blue) = Engine OFF\n'
            f'Governor: {v_pulse_stop_kmh:.1f} km/h ({EFF_RPM_LIMIT:.0f} RPM)  |  '
            f'Clutch re-engage: {v_glide_stop_kmh:.1f} km/h ({IDLE_RPM:.0f} RPM)',
            fontsize=11
        )
        plt.grid(True, linestyle=':', alpha=0.6)
        
        plt.scatter(track_x[0], track_y[0], color='green', s=120, zorder=5, label='Start')
        plt.scatter(track_x[-1], track_y[-1], color='black', s=120, zorder=5, label='Finish', marker='X')
        
        plt.plot([], [], color='red',  linewidth=5, label=f'PULSE  (engine on,  >{v_glide_stop_kmh:.1f} km/h)')
        plt.plot([], [], color='blue', linewidth=5, label=f'GLIDE  (engine off, coast down to {v_glide_stop_kmh:.1f} km/h)')
        plt.legend(loc='upper right', fontsize=9)
        
        map_path = os.path.join(script_dir, "pulse_glide_track_map.png")
        plt.savefig(map_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved track map to {map_path}")
    except Exception as e:
        print(f"Failed to plot track map: {e}")

    # Plotting telemetry
    try:
        v_arr   = np.array(res['velocity']) * 3.6
        rpm_arr = np.array(res['rpm'])
        v_max_arr = np.array(res['v_max_lim'])    # governor speed
        v_min_arr = np.array(res['v_min_lim'])    # adaptive pace target

        v_pulse_stop_kmh = 60.0
        v_glide_stop_kmh = (IDLE_RPM    * 2 * np.pi * engine_model.WHEEL_RADIUS) / (engine_model.GEAR_RATIO * 60) * 3.6

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                                        gridspec_kw={'height_ratios': [2, 1]})
        fig.suptitle(
            f'Pulse & Glide Strategy Telemetry  |  '
            f'Race: {res["final_time"]/60:.1f} min  |  '
            f'Avg: {avg_speed:.1f} km/h  |  '
            f'Economy: {km_per_liter:.0f} km/L',
            fontsize=13, fontweight='bold'
        )

        # --- Top panel: velocity ---
        ax1.fill_between(d_arr, 0, v_arr, where=(state_arr == 1),
                         color='#e74c3c', alpha=0.25, label='PULSE (engine ON)')
        ax1.fill_between(d_arr, 0, v_arr, where=(state_arr == 0),
                         color='#3498db', alpha=0.18, label='GLIDE (engine OFF)')
        ax1.plot(d_arr, v_arr, color='#2c3e50', linewidth=1.2, label='Vehicle speed')

        ax1.axhline(v_pulse_stop_kmh, color='#e74c3c', linestyle='--', linewidth=1.5,
                    label=f'Governor cutoff  {v_pulse_stop_kmh:.1f} km/h  ({EFF_RPM_LIMIT:.0f} RPM)')
        ax1.axhline(v_glide_stop_kmh, color='#27ae60', linestyle='--', linewidth=1.5,
                    label=f'Clutch re-engage {v_glide_stop_kmh:.1f} km/h  ({IDLE_RPM:.0f} RPM)')
        ax1.plot(d_arr, v_min_arr, color='#f39c12', linewidth=0.8, linestyle=':',
                 alpha=0.8, label='Adaptive pace target')

        # Lap boundary markers
        lap_arr = np.array(res['lap'])
        for lap_i in range(2, num_laps + 1):
            idx = np.where(np.diff((lap_arr == lap_i).astype(int)) > 0)[0]
            if len(idx) > 0:
                ax1.axvline(d_arr[idx[0]], color='gray', linestyle='-', linewidth=0.8, alpha=0.5)
                ax1.text(d_arr[idx[0]] + 20, v_pulse_stop_kmh * 0.95,
                         f'Lap {lap_i}', fontsize=8, color='gray')

        ax1.set_ylabel('Speed (km/h)', fontsize=11)
        ax1.set_ylim(0, max(v_arr.max() * 1.15, v_pulse_stop_kmh * 1.4))
        ax1.legend(loc='upper right', fontsize=8, ncol=2)
        ax1.grid(True, linestyle=':', alpha=0.5)

        # --- Bottom panel: engine RPM ---
        ax2.fill_between(d_arr, 0, rpm_arr, where=(state_arr == 1),
                         color='#e74c3c', alpha=0.3)
        ax2.fill_between(d_arr, 0, rpm_arr, where=(state_arr == 0),
                         color='#3498db', alpha=0.15)
        ax2.plot(d_arr, rpm_arr, color='#8e44ad', linewidth=1.0, label='Engine / Wheel RPM')
        ax2.axhline(EFF_RPM_LIMIT, color='#e74c3c', linestyle='--', linewidth=1.2,
                    label=f'Governor {EFF_RPM_LIMIT:.0f} RPM')
        ax2.axhline(IDLE_RPM, color='#27ae60', linestyle='--', linewidth=1.2,
                    label=f'Idle / Clutch {IDLE_RPM:.0f} RPM')
        ax2.set_ylabel('RPM', fontsize=11)
        ax2.set_xlabel('Distance (m)', fontsize=11)
        ax2.set_ylim(0, EFF_RPM_LIMIT * 1.3)
        ax2.legend(loc='upper right', fontsize=8)
        ax2.grid(True, linestyle=':', alpha=0.5)

        plt.tight_layout()
        plot_path = os.path.join(script_dir, "pulse_glide_telemetry.png")
        plt.savefig(plot_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"Saved telemetry plot to {plot_path}")
    except Exception as e:
        print(f"Failed to plot telemetry: {e}")

    print("\n" + "="*60)
    print("   FINAL PHYSICAL 4-LAP RACE PERFORMANCE SUMMARY")
    print("="*60)
    print(f"   Lap | Time (min) | Fuel (mL) | Avg Spd | Mileage")
    print("   " + "-" * 52)
    for row in lap_stats:
        print(f"   {row[0]:2}  | {row[1]/60:10.2f} | {row[2]:9.2f} | {row[3]:7.2f} | {row[4]:7.1f}")
    
    print("   " + "-" * 52)
    print(f"   Total Fuel Consumed:   {petrol_ml:.2f} mL")
    print(f"   Total Race Time:       {res['final_time']/60:5.2f} min ({res['final_time']:.1f} s)")
    print(f"   Race Average Speed:    {avg_speed:.2f} km/h")
    print(f"   Overall Efficiency:    {efficiency_pct:.2f}%")
    print(f"   TOTAL RACE ECONOMY:   {km_per_liter:.1f} km/L")
    print("="*60 + "\n")
    
    # Save statistics for driver guide
    try:
        excel_path = os.path.join(script_dir, "pulse_glide_driver_guide.xlsx")
        df_phases = pd.DataFrame(segments_data)
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            df_phases.to_excel(writer, sheet_name='Optimized_Strategy', index=False)
            df_metrics = pd.DataFrame({
                'Metric': ['Fuel (mL)', 'Time (min)', 'Avg Speed'],
                'Value': [petrol_ml, res['final_time']/60.0, avg_speed]
            })
            df_metrics.to_excel(writer, sheet_name='Summary', index=False)
    except: pass


if __name__ == "__main__":
    main()
