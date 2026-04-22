import numpy as np
import pandas as pd
import os
import pulse_glide_strategy

def final_optimization():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    track_file = os.path.join(script_dir, "sem_apme_2025-track_coordinates.csv")
    distance_m, altitude, slope, is_curve, curvature = pulse_glide_strategy.load_and_preprocess_track(track_file)
    
    # Grid search for the global minimum mL
    v_avg_targets = [25.05, 25.2, 25.5]
    v_pulse_stops = np.arange(25, 61, 2) # Test everything from 25 to 60 km/h
    
    results = []
    
    for v_avg in v_avg_targets:
        for v_stop in v_pulse_stops:
            if v_stop <= v_avg: continue # Invalid window
            
            res = pulse_glide_strategy.evaluate_strategy(distance_m, slope, is_curve, v_avg_target=v_avg, v_pulse_stop=v_stop)
            
            actual_dist = res['distance'][-1] if res['distance'] else 0
            if actual_dist < distance_m[-1] - 10:
                continue 
                
            avg_speed = (actual_dist / res['final_time']) * 3.6
            fuel_ml = (res['fuel_joules'] / 1e6 / 34.2) * 1000
            
            if avg_speed < 25.0:
                continue 
                
            results.append({
                'V_AVG_TARGET': v_avg,
                'V_PULSE_STOP': v_stop,
                'Fuel_mL': fuel_ml,
                'Actual_Avg_Speed': avg_speed,
                'Total_Time_Min': res['final_time'] / 60.0
            })

    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values(by='Fuel_mL')
        best = df.iloc[0]
        print(f"Optimal Strategy Found: AvgTarget {best['V_AVG_TARGET']:.2f}, PulseTo {best['V_PULSE_STOP']:.2f} -> {best['Fuel_mL']:.3f} mL")
        return best
    return None

if __name__ == "__main__":
    final_optimization()
