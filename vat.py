#!/usr/bin/env python3
"""
vat.py

Improved VAT extraction (German/English synonyms and abbreviations).
Uses PyMuPDF + EasyOCR and groups OCR tokens into visual lines,
then finds VAT keywords and nearest currency/percent tokens.

Run:
    python vat.py
"""

import sys
import os
import subprocess
import threading
import re
import shutil
from pathlib import Path

# ---------------------------
# Best-effort pip install of Python packages
# ---------------------------
PIP_PACKAGES = {
    "flask": "flask",
    "pandas": "pandas",
    "openpyxl": "openpyxl",
    "PIL": "Pillow",
    "tkinterdnd2": "tkinterdnd2",
    "easyocr": "easyocr",
    "torch": "torch",
    "fitz": "PyMuPDF",
    "numpy": "numpy"
}

def pip_install(packages):
    for import_name, pip_name in packages.items():
        try:
            if import_name == "PIL":
                import PIL  # type: ignore
            else:
                __import__(import_name)
        except Exception:
            print(f"[installer] Installing {pip_name} ...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])
            except subprocess.CalledProcessError as e:
                print(f"[installer] Failed to install {pip_name}: {e}")

try:
    pip_install(PIP_PACKAGES)
except Exception as e:
    print("Package install attempt failed:", e)

# ---------------------------
# Imports
# ---------------------------
try:
    from flask import Flask, send_file, jsonify
    import pandas as pd
    from PIL import Image
    import tkinter as tk
    from tkinter import filedialog, messagebox
    try:
        from tkinterdnd2 import DND_FILES, TkinterDnD
        TKDND_AVAILABLE = True
    except Exception:
        TKDND_AVAILABLE = False
    import numpy as np
    import easyocr
    import fitz  # PyMuPDF
except Exception as e:
    print("Missing Python modules after install attempt:", e)
    print("If easyocr/torch/fitz failed to install automatically, please install them manually.")
    sys.exit(1)

# ---------------------------
# Initialize EasyOCR reader
# ---------------------------
def init_easyocr_reader(langs=("en", "de"), gpu=False):
    try:
        reader = easyocr.Reader(list(langs), gpu=gpu)
        return reader
    except Exception as e:
        print("[easyocr] Failed to initialize reader:", e)
        return None

READER = init_easyocr_reader(("en", "de"), gpu=False)
if READER is None:
    print("[easyocr] Reader could not be initialized. Ensure torch and easyocr are installed.")
    sys.exit(1)

# ---------------------------
# Flask app (background)
# ---------------------------
app = Flask(__name__)
RESULT_FILE = Path.cwd() / "vat_results.xlsx"
last_status = {"status": "idle", "files": []}

@app.route("/status")
def status():
    return jsonify(last_status)

@app.route("/download")
def download():
    if RESULT_FILE.exists():
        return send_file(str(RESULT_FILE), as_attachment=True)
    return jsonify({"error": "no result file"}), 404

def run_flask():
    try:
        app.run(port=5001, debug=False, use_reloader=False)
    except Exception as e:
        print("[flask] Could not start Flask server:", e)

# ---------------------------
# VAT synonyms, negatives, regexes
# ---------------------------
VAT_KEYWORDS = [
    # German
    r"Umsatzsteuer", r"Mehrwertsteuer", r"Mehrwertsteuerbetrag",
    r"MwSt", r"MwSt\.", r"USt", r"USt\.",
    r"zzgl", r"zzgl\.", r"inkl", r"inkl\.", r"inklusive", r"exkl", r"exkl\.",
    # English
    r"Value\s*Added\s*Tax", r"Sales\s*Tax", r"VAT\s*Amount", r"VAT\s*Total",
    r"Tax\s*Amount", r"\bVAT\b", r"\bTax\b",
    # phrases
    r"inkl(?:\.|usive)?\s+MwSt", r"exkl(?:\.|usive)?\s+MwSt",
    r"inkl(?:\.|usive)?\s+VAT", r"exkl(?:\.|usive)?\s+VAT",
]

VAT_NEGATIVE = [
    r"Tax\s*ID", r"Tax\s*No", r"Steuernummer", r"USt-IdNr", r"USt-IdNr\.", r"USt-Id"
]

AMOUNT_RE = re.compile(r"(?:(?:€|\$|USD|EUR)\s?)?([0-9]{1,3}(?:[.,][0-9]{3})*(?:[.,][0-9]{2})?)\s?(?:€|\$|USD|EUR)?")
PERCENT_RE = re.compile(r"([0-9]{1,2}(?:[.,][0-9]+)?)\s?%")

def normalize_number(s):
    if s is None:
        return None
    s = str(s).strip()
    if s.count(",") > 0 and s.count(".") > 0:
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    else:
        s = s.replace(",", ".")
    try:
        return float(s)
    except:
        return None

def contains_negative(text):
    for neg in VAT_NEGATIVE:
        if re.search(neg, text, re.IGNORECASE):
            return True
    return False

# ---------------------------
# PDF rendering (PyMuPDF)
# ---------------------------
def pil_from_fitz_pixmap(pix):
    if pix.n < 4:
        return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    else:
        pix2 = fitz.Pixmap(fitz.csRGB, pix)
        img = Image.frombytes("RGB", [pix2.width, pix2.height], pix2.samples)
        pix2 = None
        return img

def render_pdf_to_images(pdf_path, zoom=2):
    images = []
    try:
        doc = fitz.open(str(pdf_path))
        mat = fitz.Matrix(zoom, zoom)
        for page in doc:
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = pil_from_fitz_pixmap(pix)
            images.append(img)
        doc.close()
    except Exception as e:
        print(f"[pdf] Failed to render PDF {pdf_path}: {e}")
    return images

# ---------------------------
# OCR token grouping and nearest-token extraction
# ---------------------------
def ocr_tokens_from_image(pil_img, reader=READER):
    """
    Returns list of tokens: each token is dict with keys:
    text, conf, bbox (x0,y0,x1,y1), x_center, y_center
    Uses easyocr.readtext(detail=1).
    """
    try:
        arr = np.array(pil_img.convert("RGB"))
        raw = reader.readtext(arr, detail=1)  # list of [bbox, text, conf]
    except Exception as e:
        print("[ocr] EasyOCR error:", e)
        return []

    tokens = []
    for item in raw:
        bbox, text, conf = item
        # bbox is [[x0,y0],[x1,y1],[x2,y2],[x3,y3]]
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
        x_center = (x0 + x1) / 2.0
        y_center = (y0 + y1) / 2.0
        tokens.append({
            "text": text.strip(),
            "conf": float(conf) if conf is not None else 0.0,
            "bbox": (x0, y0, x1, y1),
            "x_center": x_center,
            "y_center": y_center
        })
    return tokens

def group_tokens_into_lines(tokens, y_tol=10):
    """
    Group tokens into visual lines by y_center proximity.
    Returns list of lines; each line is list of tokens sorted by x_center.
    """
    if not tokens:
        return []
    tokens_sorted = sorted(tokens, key=lambda t: t["y_center"])
    lines = []
    current_line = [tokens_sorted[0]]
    for tok in tokens_sorted[1:]:
        if abs(tok["y_center"] - current_line[-1]["y_center"]) <= y_tol:
            current_line.append(tok)
        else:
            # finalize current line
            lines.append(sorted(current_line, key=lambda t: t["x_center"]))
            current_line = [tok]
    if current_line:
        lines.append(sorted(current_line, key=lambda t: t["x_center"]))
    return lines

def line_text(line):
    return " ".join([t["text"] for t in line]).strip()

def find_amount_percent_in_line(line):
    """
    Search for amount and percent tokens in a line (string).
    Returns tuple (amount_raw, amount_val, percent_val)
    """
    txt = line_text(line)
    amt_m = AMOUNT_RE.search(txt)
    pct_m = PERCENT_RE.search(txt)
    amt_raw = amt_m.group(1) if amt_m else None
    amt_val = normalize_number(amt_raw) if amt_raw else None
    pct_val = normalize_number(pct_m.group(1)) if pct_m else None
    return amt_raw, amt_val, pct_val

def find_nearest_amount_percent(lines, line_idx, max_lines_search=2):
    """
    Search the same line first, then nearby lines up to max_lines_search above and below.
    Returns first found candidate with highest combined confidence.
    """
    candidates = []
    # check same line
    for offset in range(0, max_lines_search+1):
        for sign in (0, -1, 1) if offset==0 else (-1,1):
            idx = line_idx + sign*offset
            if idx < 0 or idx >= len(lines):
                continue
            line = lines[idx]
            if contains_negative(line_text(line)):
                continue
            amt_raw, amt_val, pct_val = find_amount_percent_in_line(line)
            if amt_raw or pct_val:
                # compute average confidence of tokens in this line
                avg_conf = sum(t["conf"] for t in line) / max(1, len(line))
                candidates.append({
                    "line_idx": idx,
                    "line_text": line_text(line),
                    "amount_raw": amt_raw,
                    "amount": amt_val,
                    "percent": pct_val,
                    "avg_conf": avg_conf,
                    "offset": abs(idx - line_idx)
                })
        if candidates:
            break
    # choose best candidate: prefer percent present, then higher avg_conf, then smaller offset
    if not candidates:
        return None
    candidates.sort(key=lambda c: ((0 if c["percent"] is not None else 1), -c["avg_conf"], c["offset"]))
    return candidates[0]

# ---------------------------
# High-level file processing
# ---------------------------
def process_file(path):
    path = Path(path)
    text_aggregate = ""
    tokens_by_page = []
    try:
        if path.suffix.lower() == ".pdf":
            pages = render_pdf_to_images(path, zoom=2)
            for p in pages:
                tokens = ocr_tokens_from_image(p)
                tokens_by_page.append(tokens)
                text_aggregate += "\n".join([t["text"] for t in tokens]) + "\n"
        else:
            img = Image.open(str(path)).convert("RGB")
            tokens = ocr_tokens_from_image(img)
            tokens_by_page.append(tokens)
            text_aggregate += "\n".join([t["text"] for t in tokens]) + "\n"
    except Exception as e:
        print(f"[process] Failed to process {path}: {e}")
    return tokens_by_page, text_aggregate

def extract_vat_from_tokens(tokens_by_page):
    """
    For each page, group tokens into lines and search for VAT keywords.
    Returns list of dicts with file-level VAT findings.
    """
    findings = []
    for page_idx, tokens in enumerate(tokens_by_page):
        lines = group_tokens_into_lines(tokens, y_tol=12)
        for i, line in enumerate(lines):
            ltxt = line_text(line)
            if contains_negative(ltxt):
                continue
            if any(re.search(k, ltxt, re.IGNORECASE) for k in VAT_KEYWORDS):
                # found VAT keyword in this line; find nearest amount/percent
                candidate = find_nearest_amount_percent(lines, i, max_lines_search=3)
                # if candidate is None, still record the line (no amount found)
                findings.append({
                    "page": page_idx + 1,
                    "keyword_line": ltxt,
                    "amount_raw": candidate["amount_raw"] if candidate else None,
                    "amount": candidate["amount"] if candidate else None,
                    "percent": candidate["percent"] if candidate else None,
                    "found_line_text": candidate["line_text"] if candidate else None,
                    "confidence": candidate["avg_conf"] if candidate else None
                })
    return findings

# ---------------------------
# Main processing and Excel output
# ---------------------------
def process_files_and_write_excel(file_paths):
    global last_status
    last_status = {"status": "processing", "files": file_paths}
    rows = []
    for fp in file_paths:
        tokens_by_page, raw_text = process_file(fp)
        findings = extract_vat_from_tokens(tokens_by_page)
        # fallback heuristics if no findings
        if not findings:
            # try regex on raw_text
            for m in re.finditer(r"(.{0,80}(?:MwSt|Umsatzsteuer|Mehrwertsteuer|USt|VAT|Sales Tax|Value Added Tax|inkl|zzgl|exkl).{0,80})", raw_text, re.IGNORECASE):
                snippet = m.group(0).strip()
                if contains_negative(snippet):
                    continue
                amt_m = AMOUNT_RE.search(snippet)
                pct_m = PERCENT_RE.search(snippet)
                amt_raw = amt_m.group(1) if amt_m else None
                amt_val = normalize_number(amt_raw) if amt_raw else None
                pct_val = normalize_number(pct_m.group(1)) if pct_m else None
                findings.append({
                    "page": None,
                    "keyword_line": snippet,
                    "amount_raw": amt_raw,
                    "amount": amt_val,
                    "percent": pct_val,
                    "found_line_text": snippet,
                    "confidence": None
                })
        if not findings:
            rows.append({"file": os.path.basename(fp), "page": None, "keyword_line": "", "amount_raw": "", "amount": None, "percent": None, "confidence": None})
        else:
            for f in findings:
                rows.append({
                    "file": os.path.basename(fp),
                    "page": f.get("page"),
                    "keyword_line": f.get("keyword_line",""),
                    "amount_raw": f.get("amount_raw",""),
                    "amount": f.get("amount"),
                    "percent": f.get("percent"),
                    "confidence": f.get("confidence")
                })
    df = pd.DataFrame(rows, columns=["file", "page", "keyword_line", "amount_raw", "amount", "percent", "confidence"])
    try:
        df.to_excel(RESULT_FILE, index=False)
        last_status = {"status": "done", "files": file_paths, "result": str(RESULT_FILE)}
        try:
            if sys.platform.startswith("win"):
                os.startfile(str(RESULT_FILE))
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(RESULT_FILE)])
            else:
                subprocess.Popen(["xdg-open", str(RESULT_FILE)])
        except Exception as e:
            print("[open] Could not open result file automatically:", e)
    except Exception as e:
        last_status = {"status": "error", "error": str(e)}
        print("[excel] Failed to write Excel:", e)

# ---------------------------
# Tkinter GUI (drag-and-drop)
# ---------------------------
class App:
    def __init__(self, root):
        self.root = root
        root.title("VAT Extractor")
        root.geometry("820x520")
        self.frame = tk.Frame(root, padx=10, pady=10)
        self.frame.pack(fill="both", expand=True)

        lbl = tk.Label(self.frame, text="Drag and drop PDF or image files here\n(or click Select Files)", font=("Segoe UI", 14))
        lbl.pack(pady=8)

        self.drop_area = tk.Text(self.frame, height=18, width=100, bg="#f7f7f7")
        self.drop_area.insert("1.0", "Drop files here...")
        self.drop_area.config(state="disabled")
        self.drop_area.pack(pady=8)

        btn_frame = tk.Frame(self.frame)
        btn_frame.pack(pady=6)
        sel_btn = tk.Button(btn_frame, text="Select Files", command=self.select_files, width=18)
        sel_btn.pack(side="left", padx=6)
        proc_btn = tk.Button(btn_frame, text="Process Last Files", command=self.process_last_files, width=18)
        proc_btn.pack(side="left", padx=6)
        status_btn = tk.Button(btn_frame, text="Open Result Folder", command=self.open_result_folder, width=18)
        status_btn.pack(side="left", padx=6)

        self.last_files = []

        if TKDND_AVAILABLE:
            try:
                self.drop_area.drop_target_register(DND_FILES)
                self.drop_area.dnd_bind('<<Drop>>', self.handle_drop)
            except Exception as e:
                print("[dnd] tkinterdnd2 available but failed to bind:", e)
        else:
            hint = tk.Label(self.frame, text="Drag-and-drop support not available. Use Select Files.", fg="gray")
            hint.pack()

    def handle_drop(self, event):
        data = event.data
        files = self._parse_drop_data(data)
        if files:
            self._show_files(files)
            self.last_files = files
            threading.Thread(target=process_files_and_write_excel, args=(files,), daemon=True).start()

    def _parse_drop_data(self, data):
        files = []
        if data:
            parts = re.findall(r"\{([^}]+)\}|\"([^\"]+)\"|([^ ]+)", data)
            for p in parts:
                path = next((x for x in p if x), None)
                if path:
                    files.append(path)
        return files

    def _show_files(self, files):
        self.drop_area.config(state="normal")
        self.drop_area.delete("1.0", "end")
        for f in files:
            self.drop_area.insert("end", f + "\n")
        self.drop_area.config(state="disabled")

    def select_files(self):
        ftypes = [("PDF files", "*.pdf"), ("Image files", "*.png;*.jpg;*.jpeg;*.tiff;*.bmp"), ("All files", "*.*")]
        files = filedialog.askopenfilenames(title="Select files", filetypes=ftypes)
        if files:
            self.last_files = list(files)
            self._show_files(self.last_files)
            threading.Thread(target=process_files_and_write_excel, args=(self.last_files,), daemon=True).start()

    def process_last_files(self):
        if not self.last_files:
            messagebox.showinfo("No files", "No files selected yet.")
            return
        threading.Thread(target=process_files_and_write_excel, args=(self.last_files,), daemon=True).start()

    def open_result_folder(self):
        folder = str(Path.cwd())
        try:
            if sys.platform.startswith("win"):
                os.startfile(folder)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", folder])
            else:
                subprocess.Popen(["xdg-open", folder])
        except Exception as e:
            messagebox.showerror("Error", f"Could not open folder: {e}")

# ---------------------------
# Entry point
# ---------------------------
def main():
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()

    try:
        if TKDND_AVAILABLE:
            try:
                root = TkinterDnD.Tk()
            except Exception:
                root = tk.Tk()
        else:
            root = tk.Tk()
    except Exception as e:
        print("[tk] Could not start Tkinter:", e)
        sys.exit(1)

    app_gui = App(root)
    root.mainloop()

if __name__ == "__main__":
    main()
