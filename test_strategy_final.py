import pulse_glide_strategy

def test():
    track_file = 'sem_apme_2025-track_coordinates.csv'
    d, a, s, ic, c = pulse_glide_strategy.load_and_preprocess_track(track_file)
    
    # PULSE UP TO THE PEAK EFFICIENCY RPM (4000 RPM = 60.3 km/h)
    # COAST UNTIL THE AVERAGE SPEED DROPS TO THE ABSOLUTE MINIMUM (25.1 km/h)
    v_target = 25.1
    v_stop = 60.3
    
    res = pulse_glide_strategy.evaluate_strategy(d, s, ic, v_avg_target=v_target, v_pulse_stop=v_stop)
    
    fuel_ml = (res['fuel_joules'] / 1e6 / 34.2) * 1000
    avg_speed = (d[-1] / res['final_time']) * 3.6
    
    print(f"========================================")
    print(f"STRATEGY: Pulse to 60.3 (4000 RPM) | Low Target 25.1")
    print(f"========================================")
    print(f"Fuel Consumption: {fuel_ml:.3f} mL")
    print(f"Average Speed:    {avg_speed:.2f} km/h")
    print(f"Time:             {res['final_time']:.1f} s")
    print(f"Pulse Count:      {res['state'].count(1) / (1 / 0.05)}") # rough
    print(f"Energy In (J):    {res['fuel_joules']:.1f}")
    
if __name__ == "__main__":
    test()
