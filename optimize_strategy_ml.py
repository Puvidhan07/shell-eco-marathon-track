import numpy as np
import pandas as pd
import os
import sys
from pulse_glide_strategy import load_and_preprocess_track, evaluate_strategy

def optimize():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    track_file = os.path.join(script_dir, "sem_apme_2025-track_coordinates.csv")
    
    # Load track once
    distance_m, altitude, slope, is_curve, curvature = load_and_preprocess_track(track_file)
    
    # Define Search Space
    # Current Baseline: V_AVG_TARGET = 27.0, V_SAFETY_FLOOR = 15.0
    v_avg_targets = np.linspace(25.1, 26.5, 6) # Try to stay close to 25.0
    v_floors = [12.0, 15.0, 18.0]
    
    results = []
    
    print(f"Starting Grid Search Optimization (Total tests: {len(v_avg_targets) * len(v_floors)})...")
    
    import pulse_glide_strategy

    for v_avg in v_avg_targets:
        for v_floor in v_floors:
            # Overwrite global constants in the module for this run
            # Note: This is a bit hacky but works for a single-threaded simulation
            pulse_glide_strategy.V_AVG_TARGET = v_avg
            pulse_glide_strategy.V_SAFETY_FLOOR = v_floor
            
            res = evaluate_strategy(distance_m, slope, is_curve)
            
            # Check if car finished track (within 10m)
            actual_dist = res['distance'][-1] if res['distance'] else 0
            if actual_dist < distance_m[-1] - 10:
                continue # Stalled
                
            avg_speed = (actual_dist / res['final_time']) * 3.6
            fuel_ml = (res['fuel_joules'] / 1e6 / 34.2) * 1000
            
            # Penalize if it didn't meet the target (though average_strategy usually does)
            if avg_speed < 25.0:
                fuel_ml += 1000 # Massive penalty
                
            results.append({
                'V_AVG_TARGET': v_avg,
                'V_FLOOR': v_floor,
                'Fuel_mL': fuel_ml,
                'Real_Avg_Speed': avg_speed,
                'Total_Time_Min': res['final_time'] / 60.0
            })
            print(f"Tested: Target {v_avg:.1f} | Floor {v_floor:.1f} -> Result: {fuel_ml:.2f} mL ({avg_speed:.2f} km/h)")

    df = pd.DataFrame(results)
    df = df.sort_values(by='Fuel_mL')
    
    print("\n" + "="*40)
    print("OPTIMIZATION COMPLETE")
    print("="*40)
    if not df.empty:
        best = df.iloc[0]
        print(f"BEST STRATEGY FOUND:")
        print(f"V_AVG_TARGET: {best['V_AVG_TARGET']:.2f} km/h")
        print(f"V_FLOOR:      {best['V_FLOOR']:.2f} km/h")
        print(f"Fuel Used:    {best['Fuel_mL']:.2f} mL")
        print(f"Actual Speed: {best['Real_Avg_Speed']:.2f} km/h")
        print(f"Lap Time:     {best['Total_Time_Min']:.2f} min")
        
        # Save to file for main script to consume
        df.to_csv("optimization_results.csv", index=False)
    else:
        print("Error: No valid strategies found.")

if __name__ == "__main__":
    optimize()
