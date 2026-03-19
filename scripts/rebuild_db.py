# scripts/rebuild_db.py
# One-click: parse all PDFs → chunk with section_semantic_chunking → ingest to Neo4j
from pathlib import Path
import os, shutil, subprocess

from healthcare_rag_llm.pipelines.ingest_parse import run_pipeline as parse_pipeline
from healthcare_rag_llm.chunking.section_semantic_chunking import section_semantic_chunking
from healthcare_rag_llm.graph_builder.neo4j_loader import Neo4jConnector


ROOT = Path(__file__).resolve().parents[1]

CHUNK_DIR = ROOT / "data" / "chunks" / "section_semantic_chunking_result"

RAW_MEDICAID = ROOT / "data" / "raw" / "medicaid_update"
OUT_MEDICAID = ROOT / "data" / "processed" / "medicaid_update"

RAW_WAIVER = ROOT / "data" / "raw" / "children_waiver"
OUT_WAIVER = ROOT / "data" / "processed" / "children_waiver"

RAW_PHARMACY = ROOT / "data" / "raw" / "pharmacy"
OUT_PHARMACY = ROOT / "data" / "processed" / "pharmacy"

RAW_PHARMACY_BILLING = ROOT / "data" / "raw" / "pharmacy_billing"
OUT_PHARMACY_BILLING = ROOT / "data" / "processed" / "pharmacy_billing"

INGEST_GRAPH_SCRIPT = ROOT / "scripts" / "ingest_graph.py"
METADATA_FILE = ROOT / "data" / "metadata" / "metadata_filled.csv"


def clear_folder(path):
    if not path.exists():
        path.mkdir(parents=True)
        return
    for name in os.listdir(path):
        full = path / name
        if full.is_file() or full.is_symlink():
            full.unlink()
        else:
            shutil.rmtree(full)


def reset_graph():
    connector = Neo4jConnector()
    with connector.driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
    connector.close()
    print("All nodes and relationships deleted from Neo4j.")


def main():
    clear_folder(CHUNK_DIR)

    # ── 1. Parse all document types ──
    print("\n=== Parsing Medicaid Updates ===")
    parse_pipeline(raw_dir=str(RAW_MEDICAID), out_dir=str(OUT_MEDICAID),
                   save_text=True, save_json=True)

    print("\n=== Parsing Children Waiver ===")
    parse_pipeline(raw_dir=str(RAW_WAIVER), out_dir=str(OUT_WAIVER),
                   save_text=True, save_json=True)

    print("\n=== Parsing Pharmacy ===")
    parse_pipeline(raw_dir=str(RAW_PHARMACY), out_dir=str(OUT_PHARMACY),
                   save_text=True, save_json=True)

    print("\n=== Parsing Pharmacy Billing ===")
    parse_pipeline(raw_dir=str(RAW_PHARMACY_BILLING), out_dir=str(OUT_PHARMACY_BILLING),
                   save_text=True, save_json=True)

    # ── 2. Chunk all documents with section_semantic_chunking ──

    # Medicaid Updates (JSON input)
    print("\n=== Chunking Medicaid Updates ===")
    for jp in sorted(OUT_MEDICAID.glob("*.json")):
        section_semantic_chunking(
            json_path=str(jp),
            output_dir=str(CHUNK_DIR),
            max_chunk_chars=1200,
            similarity_threshold=0.35,
            verbose=True,
        )

    # Children Waiver (JSON input)
    print("\n=== Chunking Children Waiver ===")
    for jp in sorted(OUT_WAIVER.glob("*.json")):
        section_semantic_chunking(
            json_path=str(jp),
            output_dir=str(CHUNK_DIR),
            max_chunk_chars=1200,
            similarity_threshold=0.35,
            verbose=True,
        )

    # Pharmacy Policy Guidelines (TXT input)
    print("\n=== Chunking Pharmacy Policy ===")
    pharmacy_txt = OUT_PHARMACY / "Pharmacy_Policy_Guidelines.txt"
    if pharmacy_txt.exists():
        section_semantic_chunking(
            txt_path=str(pharmacy_txt),
            output_dir=str(CHUNK_DIR),
            category="Pharmacy",
            max_chunk_chars=1200,
            similarity_threshold=0.35,
            preset="pharmacy_policy",
            verbose=True,
        )

    # Pharmacy Billing Guidelines (TXT input)
    print("\n=== Chunking Pharmacy Billing ===")
    billing_txt = OUT_PHARMACY_BILLING / "Pharmacy_Billing_Guidelines.txt"
    if billing_txt.exists():
        section_semantic_chunking(
            txt_path=str(billing_txt),
            output_dir=str(CHUNK_DIR),
            category="Pharmacy Billing",
            max_chunk_chars=1200,
            similarity_threshold=0.35,
            preset="pharmacy_billing",
            verbose=True,
        )

    # ── 3. Reset Neo4j and ingest ──
    print("\n=== Resetting Neo4j ===")
    reset_graph()

    print("\n=== Ingesting chunks ===")
    subprocess.run([
        "python",
        str(INGEST_GRAPH_SCRIPT),
        "--chunk_dir", str(CHUNK_DIR),
        "--meta_file", str(METADATA_FILE)
    ], check=True)

    print("\nDONE")


if __name__ == "__main__":
    main()
