import sqlite3, torch, argparse
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

T_TRAIN_END = 1479600000
MODEL_NAME  = "cardiffnlp/twitter-roberta-base-hate-latest"
MAX_TOKENS  = 512
THRESHOLD   = 0.5

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--table", default="train_lex_pool", help="Pool table name (default train_lex_pool)")
    args = ap.parse_args()

    DEVICE = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else
        "cpu"
    )
    print("Using device:", DEVICE)

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(DEVICE).eval()

    id2label = {int(k) if isinstance(k,str) and k.isdigit() else k: v for k,v in model.config.id2label.items()}
    hate_idx = next((i for i,lab in id2label.items() if "HATE" in str(lab).upper() and "NOT" not in str(lab).upper()),
                    1 if model.config.num_labels==2 else 0)
    print("Hate label index:", hate_idx)

    conn = sqlite3.connect(args.db, timeout=300)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute("PRAGMA cache_size=-200000")
    conn.commit()

    try:
        conn.execute("ALTER TABLE comments ADD COLUMN is_hate INTEGER")
        conn.commit()
    except sqlite3.OperationalError:
        pass

    cur = conn.cursor()
    cur.execute(f"""
        SELECT COUNT(*)
        FROM comments c
        JOIN {args.table} p ON p.id = c.id
        WHERE c.is_hate IS NULL AND c.created_utc < ?
    """, (T_TRAIN_END,))
    todo = cur.fetchone()[0]
    print(f"[INFO] Pool rows to label (TRAIN): {todo:,}")

    read_cur  = conn.cursor()
    write_cur = conn.cursor()
    read_cur.execute(f"""
        SELECT c.id, c.body
        FROM comments c
        JOIN {args.table} p ON p.id = c.id
        WHERE c.is_hate IS NULL AND c.created_utc < ?
    """, (T_TRAIN_END,))

    pbar = tqdm(total=todo, desc="Labeling pool", unit="rows")
    total_updated = 0

    while True:
        rows = read_cur.fetchmany(args.batch)
        if not rows:
            break

        ids   = [r[0] for r in rows]
        texts = [(r[1] or "") for r in rows]

        trivial_updates = [(0, cid) for cid, t in zip(ids, texts) if t in ("[deleted]","[removed]","")]
        model_inputs    = [t for t in texts if t not in ("[deleted]","[removed]","")]
        model_ids       = [cid for cid, t in zip(ids, texts) if t not in ("[deleted]","[removed]","")]

        updates = []
        if model_inputs:
            enc = tok(model_inputs, padding=True, truncation=True, max_length=MAX_TOKENS, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                logits = model(**enc).logits
                if logits.shape[-1] == 1:
                    probs = torch.sigmoid(logits.squeeze(-1)).tolist()
                else:
                    probs = torch.softmax(logits, dim=-1)[:, hate_idx].tolist()
            updates.extend((1 if p >= THRESHOLD else 0, cid) for cid, p in zip(model_ids, probs))

        updates.extend(trivial_updates)

        if updates:
            write_cur.executemany("UPDATE comments SET is_hate=? WHERE id=?", updates)
            conn.commit()
            total_updated += len(updates)

        pbar.update(len(rows))

    pbar.close()
    conn.close()
    print(f"[OK] Pool labeling complete. Rows updated: {total_updated:,}")

if __name__ == "__main__":
    main()
