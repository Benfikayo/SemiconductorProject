import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from importlib.metadata import version
from pathlib import path

import pandas as pd
from dotenv import find_dotenv, load_dotenv
from mp_api.client import MPRester 

try:
    from src.exception import CustomException
    from src.logger import logging
except ImportError:
    import logging
    CustomException = Exception
PROJECT_ROOT = path(__file__).resolve().parents[2]


@dataclass
class DataIngestionConfig:
    data_dir: path = PROJECT_ROOT / "notebook" / "data"
    min_gap: float = 0.01  #eV
    max_gap: float = 20    #eV
    chunk_size: int = 1000


FIELDS = ["materials_id", "formula_pretty", "chemsys", "elements", "nelements", "nsites", "volume", "density", "symmetry", "formation_energy_per_atom", "energy_above_hull", "is_stable", "band_gap", "cbm", "is_theoretical",
         ]

def to_row(d)
    """Sumarry of Document of MP into a plan dict"""
    sym = gatattr(d, "symmetry", None)
    crystal_system = gatattr(d, "crystal_system", None) if sym else None
    elements = getattr(d, "crystal_system", None) if sym else None
    return{
        "material_id": str(d.material_id),
        "formula": getattr(d, "formula_pretty", None),
        "chemsys": getattr(d, "chemsys", None), 
        "elements": "_".join(sorted(str(e) for e in elements)) if elements else None,
        "nelements": getattr(d, "nelements", None),
        "nsites": getattr(d, "nsites", None),
        "volume": getattr(d, "volume", None),
        "density": getattr(d, "density", None),
        "crystal_system": getattr(d, "crystal_system", None),
        "space_group": getattr(d, "space_ group", None),
        "formation_energy_per_atom": getattr(d, "formation_energy_per_atom", None),
        "energy_above_hull": getattr(d, "energy_above_hull", None),
        "is_stable": getattr(d, "is_stable", None),
        "band_gap": getattr(d, "band_gap", None),
        "cbm": getattr(d, "cbm", None),
        "vbm": getattr(d, "vbm", None),
        "is_gap_direct": getattr(d, "is_gap_direct", None),
        "is_theoretical": getattr(d, "is_theoretical", None),
    }

class DataIngestion:
    def __init__(self, config: DataIngestionConfig = None):
        self.config = config or DataIngestionConfig()

    def initiate_data_ingestion(self, tesst_run: bool = False)  -> path:
        logging.info("Data ingestion process started (test_run=%s)", test_run)
        try:
            load_dotenv(find_dotenv(usecwd=True))
            api_key = os.getenv("MP_API_KEY")
            if not api_key:
                raise ValueError("MP_API_KEY not found in environment variables.")

            self.config.data_dir.mkdir(parents=True, exist_ok=True)

            with MPRester(api_key) as mpr:
                try:
                    db_version = mpr.get_database_version()
                except Exception as e:
                    db_version = "Unknown"
                docs = mpr.summary.search(
                    band_gap=(self.config.min_gap, self.config.max_gap),
                    fields=FIELD,
                    chunk_size=self.config.chunk_size,
                    num_chunks=1 if test_run else None,
                )

            logging.info("Downloaded %d materials from the Materials Project database (version: %s)", len(docs), db_version)

            df = pd.DataFrame([to_row(d) for d in docs])
            df = df.drop_duplicates(subset="material_id").sort_values(by="material_id").reset_index(drop=True)

            out_name = "RawData_test.csv" if test_run else "RawData.csv"
            out_path = self.config.data_dir / out_name
            df.to_csv(out_path, index=False)

            metadata = {

                "saved_at": datetime.now().isoformat(timespec="seconds"),
                "file_name": out_name,
                "rows": int(len(df))
                "rows_with_cbm_and_vbm": int(df[["cbm", "vbm"]].notna().all(axis=1).sum()),
                "band_gap_window_eV": [self.config.min_gap, self.config.max_gap],
                "test_run": test_run,
                "mp_database_version": db_version,
                "mp_api_version": version("mp-api"),
            }
            (self.config.data_dir / f"{out_path.stem}_metadata.json").write_text(json.dumps(metadata, indent=2))
            logging.info("Data ingestion process completed successfully. Data saved to %s", out_path, len(df))
            print(json.dumps(metadata, indent=2))
            return out_path
        except Exception as e:
            logging.error("Error occurred during data ingestion: %s", e)
            raise CustomException(e, sys)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest data from the Materials Project database.")
    parser.add_argument("--test_run", action="store_true", help="Run a test ingestion with a limited number of records.")
    args = parser.parse_args()

    data_ingestion = DataIngestion()
    data_ingestion.initiate_data_ingestion(test_run=args.test_run)
            
    
