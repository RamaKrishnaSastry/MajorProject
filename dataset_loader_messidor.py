"""Load MESSIDOR dataset with DME labels aligned to IDRiD format."""
import pandas as pd
import numpy as np
from pathlib import Path

def load_messidor_as_idrid_format(messidor_dir: str) -> pd.DataFrame:
    """
    Load MESSIDOR annotations and map to IDRiD DME label format.
    
    MESSIDOR DME labels:
        0 = No DME (no exudates)
        1 = Possible/Low risk (exudates outside macular zone)  
        2 = Present/High risk (exudates within macular zone)
    
    These map directly to IDRiD DME: 0=No, 1=Mild, 2=Moderate
    """
    records = []
    # MESSIDOR comes as Excel files per hospital
    # Download from https://www.adcis.net/en/third-party/messidor/
    for excel_file in Path(messidor_dir).glob("*.xls"):
        df = pd.read_excel(excel_file)
        # Column names vary — standard is 'Image name', 'Retinopathy grade', 'Risk of macular edema'
        df.columns = df.columns.str.strip()
        
        for _, row in df.iterrows():
            img_name = str(row.get('Image name', row.iloc[0])).strip()
            dme_label = int(row.get('Risk of macular edema', 
                           row.get('Macular edema risk', 0)))
            dr_label = int(row.get('Retinopathy grade', 0))
            
            # Find image file
            for ext in ['.tif', '.jpg', '.jpeg', '.png']:
                img_path = Path(messidor_dir) / (img_name + ext)
                if img_path.exists():
                    records.append({
                        'image_path': str(img_path),
                        'dme_label': dme_label,
                        'dr_label': min(dr_label, 4)  # cap at 4 for IDRiD compatibility
                    })
                    break
    
    df_out = pd.DataFrame(records)
    print(f"Loaded {len(df_out)} MESSIDOR samples")
    print(f"DME distribution: {df_out['dme_label'].value_counts().to_dict()}")
    return df_out