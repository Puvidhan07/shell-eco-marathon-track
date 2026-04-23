import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import engine_model

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
    
    # Identify curves threshold (e.g., > 0.003 rad/meter)
    curve_threshold = 0.003
    is_curve = curvature_smoothed > curve_threshold
    
    # Ensure start line is never a curve to allow take-off
    is_curve[:20] = False
    
    return distance_m, altitude, slope, is_curve, curvature_smoothed

def evaluate_strategy(distance_m, slope, is_curve):
    """
    Simulates the pulse and glide strategy physics using an instantaneous engine model.
    """
    # Physics parameters provided by user
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

    # Effective RPM Range (e.g. 2500 - 4000 RPM)
    # v = (RPM * 2 * pi * R) / (G * 60)
    v_eff_min = (2500 * 2 * np.pi * R) / (G * 60)
    v_eff_max = (4000 * 2 * np.pi * R) / (G * 60)
    
    # Simulation settings
    dt = 0.05                 # seconds
    s_current = 0.0
    v_current = 0.0           # Starts at 0 km/hr
    t_current = 0.0
    t_last_state_change = 0.0
    total_length = distance_m[-1]
    
    state = "PULSE"           # Start with pulse
    
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
        'fuel_joules': 0.0,
        'mechanical_joules': 0.0
    }
    
    # Prevent infinite loop if stuck
    max_time = 3600 * 2       # Maximum 2 hours
    
    fuel_energy_joules = 0.0
    mech_energy_joules = 0.0

    while s_current < total_length and t_current < max_time:
        # Interpolate track data safely
        idx = np.searchsorted(distance_m, s_current)
        if idx >= len(distance_m):
            idx = len(distance_m) - 1
            
        current_slope = slope[idx]
        current_in_curve = is_curve[idx]
        
        # Adaptive velocity limits based on slope and RPM efficiency band
        # Offset limits slightly based on slope to prevent pulse hunting or stall on hills
        slope_factor = np.sin(current_slope)
        v_max = v_eff_max + (5.0 * (-slope_factor)) / 3.6
        v_min = v_eff_min + (5.0 * (-slope_factor)) / 3.6
        
        if current_in_curve:
            v_max = min(v_max, 35.0 / 3.6)
        
        # State Machine Transitions with Hysteresis and Min Pulse Time
        t_in_state = t_current - t_last_state_change
        
        if state == "PULSE":
            # Only transition to GLIDE if we reached v_max AND have pulsed for at least 3 seconds
            if v_current >= v_max and t_in_state >= 3.0:
                state = "GLIDE"
                t_last_state_change = t_current
        elif state == "GLIDE":
            # Transition to PULSE if we dropped below v_min
            if v_current <= v_min:
                state = "PULSE"
                t_last_state_change = t_current
        
        # Physics Forces
        f_drag = 0.5 * rho * Cd * A * (v_current ** 2)
        f_roll = Crr * mass * g * np.cos(current_slope)
        f_grade = mass * g * np.sin(current_slope)
        
        f_net = - f_drag - f_roll - f_grade
        
        # Calculate Instantaneous RPM and Engine Metrics
        current_rpm = (v_current * G * 60) / (2 * np.pi * R)
        
        # Clutch Logic: Assume centrifugal clutch slips to maintain min drivable RPM
        effective_rpm = max(current_rpm, 1500.0)
        m_dot, t_brake = engine_model.get_engine_metrics(effective_rpm)
        
        # Calculate Instantaneous Tractive Force
        f_tractive_instant = (t_brake * G) / R
        
        if state == "PULSE":
            # Engine acts on vehicle
            f_net += f_tractive_instant
            fuel_energy_joules += m_dot * LHV * dt
            mech_energy_joules += f_tractive_instant * v_current * dt
            
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
            results['v_max_lim'].append(v_max * 3.6)
            results['v_min_lim'].append(v_min * 3.6)
            results['rpm'].append(current_rpm)
            results['force'].append(f_tractive_instant if state == "PULSE" else 0.0)
            
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
    
    print("Loading track and calculating curvature...")
    distance_m, altitude, slope, is_curve, curvature = load_and_preprocess_track(track_file)
    
    print("Simulating pulse and glide strategy using dynamic engine model...")
    
    res = evaluate_strategy(distance_m, slope, is_curve)
    
    actual_distance_m = res['distance'][-1] if len(res['distance']) > 0 else 0.0
    avg_speed = (actual_distance_m / res['final_time']) * 3.6 if res['final_time'] > 0 else 0

    
    # Calculate volumetric fuel (assuming Petrol ~34.2 MJ / Liter)
    # Convert Joules to mL
    mj_total = res['fuel_joules'] / 1e6
    petrol_liters = mj_total / 34.2
    petrol_ml = petrol_liters * 1000
    
    # Output metrics
    # Fix: use the actual traveled distance in case the car stalled before finishing the track
    actual_distance_m = res['distance'][-1] if len(res['distance']) > 0 else 0.0
    actual_distance_km = actual_distance_m / 1000
    
    avg_speed_kmh = (actual_distance_m / res['final_time']) * 3.6 if res['final_time'] > 0 else 0
    l_per_100km = (petrol_liters / actual_distance_km) * 100 if actual_distance_km > 0 else 0
    km_per_liter = 1 / (l_per_100km/100) if l_per_100km > 0 else 0
    
    print("\n" + "="*40)
    print("PULSE AND GLIDE FUEL CALCULATION")
    if actual_distance_m < distance_m[-1] - 10:
        print(">>> WARNING: Vehicle stalled before finishing the lap! <<<")
    print("="*40)
    
    print(f"Vehicle Mass:          140 kg (assumed)")
    print(f"Gear Ratio:            {engine_model.GEAR_RATIO}:1")
    print(f"Wheel Radius:          {engine_model.WHEEL_RADIUS:.3f} m")
    print(f"Total Tractive Force:  {res['avg_tractive_force']:.2f} N (Avg during Pulse)")
    print(f"Aero Drag (Cd*A):      {0.5 * 0.371:.3f} m^2")
    print(f"Rolling Resistance:    0.007")
    print("-" * 40)
    print(f"Run Time:              {res['final_time']/60:.2f} min ({res['final_time']:.1f} s)")
    print(f"Distance Traveled:     {actual_distance_m:.1f} m (out of {distance_m[-1]:.1f} m)")
    print(f"Avg Speed:             {avg_speed_kmh:.2f} km/h")
    print(f"Mechanical Energy:     {res['mechanical_joules']/1000:.2f} kJ")
    print(f"Fuel Energy Consumed:  {res['fuel_joules']/1000:.2f} kJ")
    print(f"Petrol Equivalent:     {petrol_ml:.2f} mL")
    print(f"Economy:               {km_per_liter:.1f} km/L  ({l_per_100km:.3f} L/100km)")
    print("="*40)
    
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
    
    # Fill frames at roughly 0.2s intervals
    for i in range(0, len(res['time']), 4):
        s_val = res['distance'][i]
        # interpolate x,y from track_s
        idx = np.searchsorted(distance_m, s_val)
        if idx >= len(distance_m): idx = len(distance_m)-1
        x_val = track_x[idx]
        y_val = track_y[idx]
        anim_data['frames'].append({
            'x': float(x_val),
            'y': float(y_val),
            's': float(s_val),
            'st': int(res['state'][i])
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
            let simTime = (currentFrame * 0.2).toFixed(1);
            timeDisplay.textContent = `Time: ${{simTime}}s | Dist: ${{f.s.toFixed(1)}}m`;
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
        
        drawTrack();
        drawCar();
        animInterval = setInterval(stepAnim, 100); // 100ms interval for fast playback
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
        changes = np.where(np.diff(state_arr) != 0)[0]
        boundaries = [0] + list(changes + 1) + [len(state_arr) - 1]
        
        print("\n" + "="*40)
        print("PHASE SEGMENT ANALYSIS")
        print("="*40)
        
        pulse_count = 0
        glide_count = 0
        total_pulse_time = 0.0
        total_glide_time = 0.0
        segments_data = [] # List to collect segment metrics for Excel
        
        for i in range(len(boundaries) - 1):
            start_idx = boundaries[i]
            end_idx = boundaries[i+1]
            if start_idx == end_idx:
                continue
                
            phase_type = "PULSE" if state_arr[start_idx] == 1 else "GLIDE"
            duration = t_arr[end_idx] - t_arr[start_idx]
            dist_covered = d_arr[end_idx] - d_arr[start_idx]
            
            if phase_type == "PULSE":
                pulse_count += 1
                total_pulse_time += duration
                label = f"Pulse #{pulse_count}"
            else:
                glide_count += 1
                total_glide_time += duration
                label = f"Glide #{glide_count}"
                
            # Get phase speeds (km/h)
            v_arr_local = np.array(res['velocity']) * 3.6
            v_start = v_arr_local[start_idx]
            v_end = v_arr_local[end_idx]
            
            # Construct human-readable strategy
            if phase_type == "PULSE":
                strat_line = f"RE-PULSE IMMEDIATELY: Accelerate from {v_start:.1f} to {v_end:.1f} km/h"
                if v_start < 5.0: # Startup
                    strat_line = f"STARTUP: Hard Pulse from 0 to {v_end:.1f} km/h"
            else:
                strat_line = f"GLIDE: Release throttle, coast from {v_start:.1f} down to {v_end:.1f} km/h"

            # Add to export list
            segments_data.append({
                'Phase': label,
                'Instruction': strat_line,
                'Start Dist (m)': d_arr[start_idx],
                'Dist Covered (m)': dist_covered,
                'Time (s)': f"{duration:.1f}",
                'Start Speed': f"{v_start:.1f}",
                'End Speed': f"{v_end:.1f}"
            })
                
            print(f"{label:10} | Time: {duration:6.1f} s | Distance: {dist_covered:6.1f} m")
            
        print("-" * 40)
        print(f"Total Pulse Phases: {pulse_count}")
        print(f"Total Glide Phases: {glide_count}")
        print(f"Total Pulse Time:   {total_pulse_time/60:.2f} min ({total_pulse_time:.1f} s)")
        print(f"Total Glide Time:   {total_glide_time/60:.2f} min ({total_glide_time:.1f} s)")
        print("="*40)

    # Plotting 2D Track Map (X vs Y)
    try:
        from matplotlib.collections import LineCollection
        plt.figure(figsize=(10, 10))
        
        # We need to compute states per track coordinate point
        track_states = np.zeros(len(distance_m))
        for i in range(len(d_arr)-1):
            s1 = d_arr[i]
            s2 = d_arr[i+1]
            st = state_arr[i]
            
            idx1 = np.searchsorted(distance_m, s1)
            idx2 = np.searchsorted(distance_m, s2)
            idx1 = min(idx1, len(distance_m)-1)
            idx2 = min(idx2, len(distance_m)-1)
            
            track_states[idx1:idx2] = st
            
        track_states[idx2:] = st
            
        points = np.array([track_x, track_y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        colors = ['red' if st == 1 else 'blue' for st in track_states[:-1]]
        
        lc = LineCollection(segments, colors=colors, linewidths=5, capstyle='round')
        ax = plt.gca()
        ax.add_collection(lc)
        ax.autoscale()
        
        plt.axis('equal')
        plt.title('2D Track Map: Pulse (Red) & Glide (Blue)')
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.grid(True, linestyle=':', alpha=0.6)
        
        plt.scatter(track_x[0], track_y[0], color='green', s=100, label='Start', zorder=5)
        plt.scatter(track_x[-1], track_y[-1], color='black', s=100, label='End', marker='X', zorder=5)
        
        # Dummy lines for legend
        plt.plot([], [], color='red', linewidth=5, label='Pulse Phase')
        plt.plot([], [], color='blue', linewidth=5, label='Glide Phase')
        plt.legend(loc='upper right')
        
        map_path = os.path.join(script_dir, "pulse_glide_track_map.png")
        plt.savefig(map_path, dpi=300)
        print(f"Saved 2D track map to {map_path}")
        plt.close()
    except Exception as e:
        print(f"Failed to plot track map: {e}")

    # Plotting Speed vs Curvature
    try:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        v_arr = np.array(res['velocity']) * 3.6 # km/h
        d_arr = np.array(res['distance'])
        
        # Speed Subplot
        ax1.plot(d_arr, v_arr, color='blue', linewidth=2, label='Velocity (km/h)')
        ax1.fill_between(d_arr, 0, 70, where=(np.array(res['state']) == 1), color='green', alpha=0.1, label='Pulse')
        ax1.set_ylabel('Speed (km/h)', fontsize=12)
        ax1.set_title('Speed and Track Curvature vs. Distance', fontsize=14)
        ax1.grid(True, linestyle=':')
        ax1.legend(loc='upper right')
        ax1.set_ylim(0, 80)
        
        # Curvature Subplot
        # Interpolate curvature to match telemetry distance
        curve_telemetry = np.interp(d_arr, distance_m, curvature)
        ax2.plot(d_arr, curve_telemetry, color='red', linewidth=1.5, label='Curvature (rad/m)')
        ax2.set_ylabel('Curvature (rad/m)', fontsize=12)
        ax2.set_xlabel('Distance (m)', fontsize=12)
        ax2.grid(True, linestyle=':')
        ax2.legend(loc='upper right')
        
        plt.tight_layout()
        analysis_path = os.path.join(script_dir, "track_analysis.png")
        plt.savefig(analysis_path, dpi=200)
        print(f"Saved Speed vs Curvature analysis to {analysis_path}")
        plt.close()
    except Exception as e:
        print(f"Failed to plot track analysis: {e}")

    # Plotting telemetry
    try:
        v_arr = np.array(res['velocity']) * 3.6 # km/h
        
        plt.figure(figsize=(12, 6))
        
        # Plot velocity
        plt.plot(d_arr, v_arr, label="Velocity (km/h)", color='blue', linewidth=2)
        
        # Plot states background
        # Create continuous block highlights for straight vs curve and pulse vs glide
        plt.fill_between(d_arr, 0, 35, where=(state_arr == 1), color='green', alpha=0.2, label='Pulse Phase')
        plt.fill_between(d_arr, 0, 35, where=(state_arr == 0), color='orange', alpha=0.2, label='Glide Phase')
        
        # Add curve overlays on the top axis 
        curve_d = np.interp(d_arr, distance_m, is_curve.astype(float))
        plt.plot(d_arr, curve_d * 50, color='red', alpha=0.8, label="In Curve", linestyle="--")

        plt.axhline(50, color='black', linestyle=':', label='Max Speed (50 km/h)')
        plt.axhline(25, color='gray', linestyle=':', label='Min Speed (25 km/h)')
        
        plt.title('Pulse and Glide Strategy Telemetry', fontsize=14)
        plt.xlabel('Distance (m)', fontsize=12)
        plt.ylabel('Speed (km/h)', fontsize=12)
        plt.ylim(0, 80)
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.tight_layout()
        
        plot_path = os.path.join(script_dir, "pulse_glide_telemetry.png")
        plt.savefig(plot_path, dpi=200)
        print(f"\nSaved telemetry plot to {plot_path}")
        # plt.show()
        
    except Exception as e:
        print(f"Failed to plot telemetry: {e}")

    # EXCEL EXPORT (Structured for Driver)
    try:
        excel_path = os.path.join(script_dir, "pulse_glide_driver_guide.xlsx")
        
        # Prepare Dash Statistics
        summary_rows = [
            ["SHELL ECO-MARATHON DRIVER STRATEGY GUIDE", ""],
            ["=========================================", ""],
            ["OVERALL PERFORMANCE SUMMARY", ""],
            ["Total Lap Distance (m)", actual_distance_m],
            ["Total Lap Time (min)", f"{res['final_time'] / 60:.2f}"],
            ["Predicted Economy (km/L)", f"{km_per_liter:.1f}"],
            ["Total Fuel Required (mL)", f"{petrol_ml:.2f}"],
            ["Average Speed (km/h)", f"{avg_speed_kmh:.2f}"],
            ["Engine Duty Cycle (P/G Ratio)", f"{(total_pulse_time / total_glide_time):.2f}" if total_glide_time > 0 else "MAX"],
            ["", ""],
            ["DETAILED SEGMENT-BY-SEGMENT STRATEGY", ""],
            ["-----------------------------------------", ""]
        ]
        
        df_summary_top = pd.DataFrame(summary_rows)
        df_phases = pd.DataFrame(segments_data)
        
        # Write to Excel with specific layout
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # Sheet 1: Driver Guide (Summary + Detailed Phases)
            df_summary_top.to_excel(writer, sheet_name='Driver_Guide', index=False, header=False)
            df_phases.to_excel(writer, sheet_name='Driver_Guide', index=False, startrow=len(summary_rows))
            
            # Sheet 2: Raw High-Res Telemetry (for technical analysis)
            df_telemetry = pd.DataFrame({
                'Time (s)': res['time'],
                'Distance (m)': res['distance'],
                'Velocity (km/h)': np.array(res['velocity']) * 3.6,
                'State': res['state'],
                'Slope_Deg': res['slope_deg'],
                'Accel_mps2': res['acceleration']
            })
            df_telemetry.to_excel(writer, sheet_name='Full_Physics_Log', index=False)
            
        print(f"\n[SUCCESS] Driver Strategy Guide exported to: {excel_path}")
        
    except Exception as e:
        print(f"\n[ERROR] Failed to export to Excel: {e}")

if __name__ == "__main__":
    main()