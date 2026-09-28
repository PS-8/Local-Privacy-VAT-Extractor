#!/usr/bin/env python3
"""
Receipts -> finamt (local) -> SteuerLLM via Hugging Face

Single-file script that:
  1) Loads receipts (images/PDFs) from a directory and extracts structured data using a local finamt client if available,
     otherwise falls back to pytesseract OCR + heuristics.
  2) Summarizes VAT by rate and collects user inputs via a short Q&A.
  3) Sends a structured prompt to a SteuerLLM model hosted on Hugging Face Inference API and prints/saves the German draft
     Umsatzsteuererklärung (VAT declaration).

Usage:
  - Install dependencies: pip install finamt pytesseract pillow pandas requests huggingface-hub
  - Set environment variables:
      FINAMT_API_KEY (optional, only if your local finamt client needs it)
      HUGGINGFACE_API_TOKEN (required to call Hugging Face Inference API)
      STEUERLLM_MODEL (e.g., "username/steuerllm-model")
  - Place receipts in a folder (default: ./receipts). Run the script:
      python3 receipts_to_vat.py --receipts ./receipts --outdir ./output

Notes:
  - This script does NOT submit anything to tax authorities. Always review the generated draft with a tax advisor.
  - Replace model name and API token with your own. Adjust finamt client usage to match your local installation.
"""

import os
import sys
import argparse
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

try:
    from PIL import Image
except Exception:
    raise SystemExit("Please install pillow: pip install pillow")

import pandas as pd
import requests

# Optional local finamt client (if installed)
FINAMT_AVAILABLE = False
try:
    import finamt  # type: ignore
    FINAMT_AVAILABLE = True
except Exception:
    FINAMT_AVAILABLE = False

# Optional OCR fallback
try:
    import pytesseract  # type: ignore
    TESSERACT_AVAILABLE = True
except Exception:
    TESSERACT_AVAILABLE = False

# Hugging Face Inference API helper
HUGGINGFACE_API_URL = "https://api-inference.huggingface.co/models/{model}"


def ocr_image_text(path: Path) -> str:
    """Extract text from an image using pytesseract (fallback)."""
    if not TESSERACT_AVAILABLE:
        raise RuntimeError("pytesseract not available; install with `pip install pytesseract`")
    img = Image.open(path).convert("RGB")
    text = pytesseract.image_to_string(img, lang="deu+eng")
    return text


def parse_receipt_text(text: str) -> Dict[str, Any]:
    """
    Heuristic parsing for receipts:
      - vendor: first non-empty line
      - date: dd.mm.yyyy or dd.mm.yy
      - total: look for keywords like Gesamt, Summe, Total, Endbetrag; fallback to last currency-like number
      - vat_lines: find occurrences like '19%  X,XX' or '7% X,XX'
    """
    result: Dict[str, Any] = {"vendor": None, "date": None, "total": None, "vat_lines": [], "raw_text": text}
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if lines:
        result["vendor"] = lines[0]

    # date
    date_match = re.search(r'(\d{1,2}\.\d{1,2}\.\d{2,4})', text)
    if date_match:
        result["date"] = date_match.group(1)

    # total
    total_match = re.search(r'(Gesamt|Total|Summe|Endbetrag|Brutto)[^\d,.-]*([\d\.\s]+,\d{2})', text, re.IGNORECASE)
    if total_match:
        amt = total_match.group(2)
        result["total"] = _normalize_amount_str(amt)
    else:
        nums = re.findall(r'([\d\.\s]+,\d{2})', text)
        if nums:
            result["total"] = _normalize_amount_str(nums[-1])

    # VAT lines
    for m in re.finditer(r'(\d{1,2})\s?%[^\d,.-]*([\d\.\s]+,\d{2})', text):
        pct = int(m.group(1))
        amt = _normalize_amount_str(m.group(2))
        result["vat_lines"].append({"rate": pct, "amount": amt})

    return result


def _normalize_amount_str(s: str) -> str:
    """Convert German-style amount '1.234,56' or '1 234,56' to '1234.56' string."""
    s_clean = s.replace(" ", "").replace(".", "").replace(",", ".")
    return s_clean


def parse_with_finamt_local(path: Path, api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Example wrapper for a local finamt client. Adapt to your local finamt API.
    If finamt is not installed or fails, raise an Exception to let caller fallback to OCR.
    """
    if not FINAMT_AVAILABLE:
        raise RuntimeError("finamt package not available locally")

    # The following is illustrative pseudocode. Replace with actual finamt client usage.
    try:
        # Example: client = finamt.Client(api_key=api_key)
        # parsed = client.parse_document(str(path))
        # return parsed.to_dict()
        # If finamt provides a CLI or local binary, call it here and parse JSON output.
        # For now, raise to fallback.
        raise NotImplementedError("Adapt parse_with_finamt_local to your local finamt client API")
    except Exception as e:
        raise


def process_receipts_folder(folder: Path, finamt_api_key: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Process all supported files in the folder and return a list of parsed receipt dicts.
    Supported extensions: png, jpg, jpeg, tiff, pdf
    """
    records: List[Dict[str, Any]] = []
    supported = {".png", ".jpg", ".jpeg", ".tiff", ".pdf"}
    for p in sorted(folder.iterdir()):
        if p.suffix.lower() not in supported:
            continue
        print(f"Processing {p.name} ...")
        record: Dict[str, Any] = {"filename": p.name}
        # Try finamt local first if available
        if FINAMT_AVAILABLE:
            try:
                parsed = parse_with_finamt_local(p, api_key=finamt_api_key)
                # Expect parsed to be a dict with vendor, date, total, vat_lines etc.
                record.update(parsed)
                records.append(record)
                continue
            except Exception as e:
                print(f"finamt local parse failed for {p.name}: {e}. Falling back to OCR heuristics.")

        # Fallback OCR
        try:
            text = ocr_image_text(p)
        except Exception as e:
            print(f"OCR failed for {p.name}: {e}. Skipping file.")
            continue
        parsed = parse_receipt_text(text)
        record.update(parsed)
        records.append(record)
    return records


def summarize_vat(records: List[Dict[str, Any]]) -> pd.DataFrame:
    """Summarize VAT amounts by rate across all receipts."""
    rows = []
    for r in records:
        for v in r.get("vat_lines", []):
            try:
                amt = float(v["amount"])
            except Exception:
                try:
                    amt = float(str(v["amount"]).replace(",", "."))
                except Exception:
                    amt = 0.0
            rows.append({"filename": r.get("filename"), "rate": int(v["rate"]), "amount": amt})
    if not rows:
        return pd.DataFrame(columns=["rate", "total_amount"])
    df = pd.DataFrame(rows)
    summary = df.groupby("rate", as_index=False).sum().rename(columns={"amount": "total_amount"})
    return summary


def prompt_user_for_declaration(vat_summary: pd.DataFrame) -> Dict[str, Any]:
    """Collect required fields for a German VAT declaration via CLI Q&A."""
    print("\n--- VAT Declaration Q&A ---")
    company_name = input("Firmenname / Steuerpflichtiger: ").strip()
    tax_number = input("Steuernummer / USt-IdNr (optional): ").strip()
    period = input("Steuerjahr (z.B. 2025): ").strip()
    period_type = input("Zeitraum type (Jahreserklärung/Monat/Quartal) [Jahreserklärung]: ").strip() or "Jahreserklärung"

    def ask_float(prompt_text: str) -> Optional[float]:
        s = input(prompt_text + " (optional, leer lassen wenn unbekannt): ").strip()
        if not s:
            return None
        try:
            return float(s.replace(",", "."))
        except Exception:
            print("Ungültige Zahl, bitte erneut eingeben.")
            return ask_float(prompt_text)

    total_sales = ask_float("Gesamtumsatz (netto) for period")
    total_vat_collected = ask_float("Summe Umsatzsteuer (collected)")
    total_input_vat = ask_float("Summe Vorsteuer (input VAT to reclaim)")

    print("\nDetected VAT summary by rate:")
    if vat_summary.empty:
        print("  Keine VAT-Zeilen automatisch erkannt.")
    else:
        print(vat_summary.to_string(index=False))

    notes = input("\nZusätzliche Hinweise (optional): ").strip()

    return {
        "company_name": company_name,
        "tax_number": tax_number,
        "period": period,
        "period_type": period_type,
        "vat_summary_by_rate": vat_summary.to_dict(orient="records"),
        "total_sales_netto": total_sales,
        "total_vat_collected": total_vat_collected,
        "total_input_vat": total_input_vat,
        "notes": notes,
    }


def build_steuerllm_prompt(context: Dict[str, Any]) -> str:
    """Create a German-language prompt for SteuerLLM based on the collected context."""
    lines = []
    lines.append("Aufgabe: Erstelle einen Entwurf einer deutschen Umsatzsteuererklärung (Umsatzsteuer-Voranmeldung / Jahreserklärung) zur Überprüfung.")
    lines.append("Sprache: Deutsch.")
    lines.append("")
    lines.append("Kontextdaten:")
    lines.append(f"Steuerpflichtiger / Firma: {context.get('company_name')}")
    if context.get("tax_number"):
        lines.append(f"Steuernummer / USt-IdNr: {context.get('tax_number')}")
    lines.append(f"Zeitraum: {context.get('period')} ({context.get('period_type')})")
    lines.append("")
    lines.append("Erkannte Umsatzsteuer nach Steuersatz (aus Belegen):")
    if context.get("vat_summary_by_rate"):
        for v in context["vat_summary_by_rate"]:
            lines.append(f"  - {v.get('rate')}%: {v.get('total_amount'):.2f} EUR")
    else:
        lines.append("  - Keine automatischen Erkennungen vorhanden.")
    lines.append("")
    if context.get("total_sales_netto") is not None:
        lines.append(f"Gesamtumsatz (netto): {context.get('total_sales_netto'):.2f} EUR")
    if context.get("total_vat_collected") is not None:
        lines.append(f"Summe Umsatzsteuer (vereinnahmt): {context.get('total_vat_collected'):.2f} EUR")
    if context.get("total_input_vat") is not None:
        lines.append(f"Summe Vorsteuer (abzugsfähig): {context.get('total_input_vat'):.2f} EUR")
    if context.get("notes"):
        lines.append(f"Anmerkungen: {context.get('notes')}")
    lines.append("")
    lines.append("Anweisungen an das Modell:")
    lines.append("  1) Erstelle einen klaren, strukturierten Entwurf der Umsatzsteuererklärung in deutscher Sprache.")
    lines.append("  2) Füge eine kurze Tabelle der Schlüsselfiguren hinzu (Umsatz netto, Umsatzsteuer, Vorsteuer, Zahllast/Erstattung).")
    lines.append("  3) Liste Annahmen und Punkte auf, die man manuell prüfen muss (fehlende Rechnungen, Rundungen, steuerfreie Umsätze).")
    lines.append("  4) Gib konkrete nächste Schritte an (z. B. Belege prüfen, Steuerberater konsultieren).")
    lines.append("  5) Hinweis: Nicht an Finanzamt übermitteln; dies ist nur ein Entwurf zur Überprüfung.")
    lines.append("")
    lines.append("Bitte liefere die Antwort als gut lesbaren deutschen Text, mit klaren Abschnitten und einer kurzen Zusammenfassung am Anfang.")
    return "\n".join(lines)


def call_huggingface_inference(prompt: str, model: str, hf_token: str, max_length: int = 2000) -> Dict[str, Any]:
    """
    Call Hugging Face Inference API for text generation.
    Uses the standard text-generation endpoint. The exact behavior depends on the model.
    """
    url = HUGGINGFACE_API_URL.format(model=model)
    headers = {"Authorization": f"Bearer {hf_token}"}
    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": max_length, "temperature": 0.2, "return_full_text": False},
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=120)
    if resp.status_code == 503:
        raise RuntimeError("Model is loading on Hugging Face side; try again in a moment.")
    resp.raise_for_status()
    # Response format varies by model and HF; many models return a list of dicts with 'generated_text'
    try:
        data = resp.json()
    except Exception:
        raise RuntimeError("Invalid JSON response from Hugging Face API")
    # Normalize common shapes
    if isinstance(data, dict) and "error" in data:
        raise RuntimeError(f"Hugging Face API error: {data['error']}")
    if isinstance(data, list):
        # e.g., [{"generated_text": "..."}]
        if data and isinstance(data[0], dict) and "generated_text" in data[0]:
            return {"text": data[0]["generated_text"], "raw": data}
    # Some models return {"generated_text": "..."}
    if isinstance(data, dict) and "generated_text" in data:
        return {"text": data["generated_text"], "raw": data}
    # Fallback: return raw JSON as text
    return {"text": json.dumps(data, ensure_ascii=False, indent=2), "raw": data}


def save_output(outdir: Path, filename: str, content: str) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / filename
    path.write_text(content, encoding="utf-8")
    return path


def main(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(description="Process receipts and generate a VAT declaration draft via SteuerLLM on Hugging Face.")
    parser.add_argument("--receipts", "-r", type=str, default="./receipts", help="Folder with receipt images/PDFs")
    parser.add_argument("--outdir", "-o", type=str, default="./output", help="Output folder for summaries and drafts")
    parser.add_argument("--finamt-key", type=str, default=None, help="Optional local finamt API key (if required)")
    parser.add_argument("--hf-model", type=str, default=os.environ.get("STEERLLM_MODEL", ""), help="Hugging Face model name for SteuerLLM")
    parser.add_argument("--hf-token", type=str, default=os.environ.get("HUGGINGFACE_API_TOKEN", ""), help="Hugging Face API token")
    args = parser.parse_args(argv)

    receipts_dir = Path(args.receipts)
    outdir = Path(args.outdir)
    if not receipts_dir.exists() or not receipts_dir.is_dir():
        print(f"Receipts folder not found: {receipts_dir}")
        sys.exit(1)

    if not args.hf_model:
        print("Hugging Face model not specified. Set --hf-model or STEERLLM_MODEL env var.")
        sys.exit(1)
    if not args.hf_token:
        print("Hugging Face API token not specified. Set --hf-token or HUGGINGFACE_API_TOKEN env var.")
        sys.exit(1)

    print("Starting receipt processing...")
    records = process_receipts_folder(receipts_dir, finamt_api_key=args.finamt_key)
    print(f"Processed {len(records)} receipts.")

    # Save raw records
    save_output(outdir, "receipt_records.json", json.dumps(records, ensure_ascii=False, indent=2))
    print(f"Saved parsed receipt records to {outdir / 'receipt_records.json'}")

    vat_summary = summarize_vat(records)
    vat_csv_path = outdir / "vat_summary.csv"
    vat_summary.to_csv(vat_csv_path, index=False)
    print(f"Saved VAT summary to {vat_csv_path}")

    # Interactive Q&A
    context = prompt_user_for_declaration(vat_summary)

    # Build prompt and call SteuerLLM on Hugging Face
    prompt_text = build_steuerllm_prompt(context)
    print("\n--- Prompt preview (first 800 chars) ---")
    print(prompt_text[:800] + ("\n...[truncated]" if len(prompt_text) > 800 else ""))

    print("\nCalling SteuerLLM model on Hugging Face Inference API...")
    try:
        hf_resp = call_huggingface_inference(prompt_text, args.hf_model, args.hf_token)
        draft_text = hf_resp.get("text", "")
        if not draft_text:
            print("No text returned from model. Raw response saved.")
            save_output(outdir, "steuerllm_raw_response.json", json.dumps(hf_resp.get("raw", {}), ensure_ascii=False, indent=2))
        else:
            out_path = save_output(outdir, "umsatzsteuer_draft_de.txt", draft_text)
            print(f"Saved SteuerLLM draft to {out_path}")
            print("\n--- SteuerLLM Draft (preview) ---\n")
            print(draft_text[:2000] + ("\n...[truncated]" if len(draft_text) > 2000 else ""))
    except Exception as e:
        print("Error calling SteuerLLM on Hugging Face:", e)
        # Save prompt for debugging
        save_output(outdir, "steuerllm_prompt.txt", prompt_text)
        print(f"Saved prompt to {outdir / 'steuerllm_prompt.txt'} for inspection.")

    print("\nDone. Review the generated draft and the saved files in the output folder.")


if __name__ == "__main__":
    main()
