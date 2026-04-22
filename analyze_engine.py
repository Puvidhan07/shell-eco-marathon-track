import numpy as np
import engine_model_2 as engine_model

def analyze_engine():
    rpms = np.linspace(1500, 5000, 100)
    bsfc_list = []
    
    for rpm in rpms:
        m_dot, t_brake = engine_model.get_engine_metrics(rpm)
        # Power = Torque * angular_velocity
        omega = rpm * (2 * np.pi / 60.0)
        p_brake = t_brake * omega  # Watts
        
        if p_brake > 0:
            # BSFC = kg/s / Watts = kg/J
            # Convert to g/kWh: (kg/s * 1000 * 3600) / (W / 1000) = (kg/s / W) * 3.6e9
            bsfc = (m_dot / p_brake) * 3.6e9
            bsfc_list.append((rpm, bsfc, p_brake))
            
    bsfc_list.sort(key=lambda x: x[1])
    
    print("Top 5 Most Efficient Engine Operating Points:")
    for i in range(min(5, len(bsfc_list))):
        r, b, p = bsfc_list[i]
        print(f"RPM: {r:.0f} | BSFC: {b:.2f} g/kWh | Power: {p/1000:.2f} kW")

if __name__ == "__main__":
    analyze_engine()
