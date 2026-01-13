######GERNERTAE REPORT######
import pandas as pd
from datetime import datetime
import os

def generate_report(detected_signals, team_id="XX"):
    # Define columns as per project spec 
    columns = [
        "run_id", "burst_id", "t_start_s", "t_end_s", "fc_hz", 
        "bw_hz", "modulation", "confidence", "snr_db", 
        "subband_start_hz", "subband_bw_hz", "notes"
    ]
    
    df = pd.DataFrame(detected_signals, columns=columns)
    
    # Generate filename: EE435_TeamXX_YYYY-MM-DDThh-mm-ssZ.csv 
    timestamp = datetime.utcnow().strftime('%Y-%m-%dT%H-%M-%SZ')
    filename = f"EE435_Team{team_id}_{timestamp}.csv"
    
    # Save as UTF-8 
    df.to_csv(filename, index=False, encoding='utf-8')
    return filename