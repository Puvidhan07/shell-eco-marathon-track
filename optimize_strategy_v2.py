import numpy as np
import pandas as pd
import os
import pulse_glide_strategy

def optimize():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    track_file = os.path.join(script_dir, "sem_apme_2025-track_coordinates.csv")
    distance_m, altitude, slope, is_curve, curvature = pulse_glide_strategy.load_and_preprocess_track(track_file)
    
    # We want to find the best v_avg_target and v_pulse_stop
    v_avg_targets = [25.1, 25.3, 25.5, 26.0]
    # v_pulse_stops (km/h) 
    # Let's try every 2 km/h from 28 to 44
    v_pulse_stops = np.arange(28, 46.1, 2)
    
    results = []
    
    print(f"Grand Optimization Scan (Total tests: {len(v_avg_targets) * len(v_pulse_stops)})...")

    for v_avg in v_avg_targets:
        for v_stop in v_pulse_stops:
            res = pulse_glide_strategy.evaluate_strategy(distance_m, slope, is_curve, v_avg_target=v_avg, v_pulse_stop=v_stop)
            
            actual_dist = res['distance'][-1] if res['distance'] else 0
            if actual_dist < distance_m[-1] - 10:
                continue 
                
            avg_speed = (actual_dist / res['final_time']) * 3.6
            fuel_ml = (res['fuel_joules'] / 1e6 / 34.2) * 1000
            
            # Constraint: Must be >= 25.0
            if avg_speed < 24.99: # Allow tiny floating point margin
                continue 
                
            results.append({
                'V_AVG_TARGET': v_avg,
                'V_PULSE_STOP': v_stop,
                'Fuel_mL': fuel_ml,
                'Actual_Avg_Speed': avg_speed,
                'Lap_Time_S': res['final_time']
            })
            print(f"Tested: AvgTarget {v_avg:.1f}, PulseStop {v_stop:.1f} -> {fuel_ml:.2f} mL ({avg_speed:.2f} km/h)")

    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values(by='Fuel_mL')
        best = df.iloc[0]
        print("\n" + "="*40)
        print("OPTIMIZATION RESULTS")
        print("="*40)
        print(f"BEST STRATEGY FOUND:")
        print(f"V_AVG_TARGET: {best['V_AVG_TARGET']:.2f} km/h")
        print(f"V_PULSE_STOP: {best['V_PULSE_STOP']:.2f} km/h")
        print(f"Fuel Used:    {best['Fuel_mL']:.2f} mL")
        print(f"Actual Speed: {best['Actual_Avg_Speed']:.2f} km/h")
        
        df.to_csv("optimization_results_final.csv", index=False)
    else:
        print("No valid strategies found.")

if __name__ == "__main__":
    optimize()
