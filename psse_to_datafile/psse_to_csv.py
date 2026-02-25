"""
PSSE RAW file (CSV format) -> bus.csv + branch.csv converter

Usage:
    python psse_to_csv.py <input_file.csv>

Output:
    bus.csv    - Bus data  (Node, NAME, BASKV, IDE, AREA, ZONE, OWNER, VM, VA,
                            NVHI, NVLO, EVHI, EVLO, NAME_MOD, BUSTYPE, Facility_name)
    branch.csv - Branch data (Node1, Node2, R, X, B, LEN)

BUSTYPE assignment:
    - Bus appears in Generator data  -> Power Plant
    - Bus appears in Transformer data -> Substation
    - Bus appears only in Load data   -> Substation
"""

import sys
import re
import csv
import os

ENCODING = "cp949"


# ---------- Name cleaning ----------

def get_name_mod(name):
    """Strip trailing alphanumeric characters from NAME -> NAME_MOD (first pass)."""
    return re.sub(r"[A-Za-z0-9]+$", "", name).strip()


def clean_facility_name(name_mod, bustype):
    """
    Remove unnecessary English suffixes and symbols from NAME_MOD
    to produce a clean Facility_name.

    Rules applied (in order):
      1. Leading single uppercase letter before Korean (e.g. D거제 -> 거제)
      2. Trailing S/ or G/ (e.g. 영월S/ -> 영월)
      3. Korean + optional digits + English plant-type suffix (TP/GT/ST/NP/PP/CC/EP/LNG/TIE)
         (e.g. 영흥TP -> 영흥, 당진1GT -> 당진)
      4. Underscore-separated English suffix after English base (e.g. POS_CC -> POS)
      5. Trailing E_ (e.g. 수완E_ -> 수완)
      6. # and everything after (e.g. 동서울# -> 동서울)
      7. Parenthesised numbers (e.g. 강화(70) -> 강화)
      8. Trailing digits after Korean (e.g. leftover digits after step 3)
      9. Trailing punctuation/whitespace (-, _, space)
    """
    s = name_mod

    # 1. Leading single uppercase letter + Korean
    s = re.sub(r"^[A-Z](?=[\uAC00-\uD7A3])", "", s)

    # 2. Trailing S/ or G/
    s = re.sub(r"[SG]/$", "", s)

    # 3. Korean + digits* + plant-type suffix + digits*
    s = re.sub(
        r"(?<=[\uAC00-\uD7A3])\d*(TP|GT|ST|NP|PP|CC|EP|LNG|TIE)\d*", "", s
    )

    # 4. Underscore-separated English plant-type suffix
    s = re.sub(r"_(GT|ST|CC|TP|NP|PP|EP)\d*", "", s)

    # 5. Trailing E_
    s = re.sub(r"E_$", "", s)

    # 6. # and everything after
    s = re.sub(r"#.*$", "", s)

    # 7. Parenthesised numbers
    s = re.sub(r"\(\d+\)", "", s)

    # 8. Trailing digits after Korean
    s = re.sub(r"(?<=[\uAC00-\uD7A3])\d+$", "", s)

    # 9. Trailing punctuation / whitespace
    s = re.sub(r"[-_\s]+$", "", s)

    return s.strip() + bustype


# ---------- Section parser ----------

def read_sections(filepath):
    """Read the PSSE raw file and split into named sections."""
    with open(filepath, "rb") as f:
        raw = f.read()
    text = raw.decode(ENCODING, errors="replace")
    lines = text.splitlines()

    sections = {
        "bus": [], "load": [], "generator": [],
        "branch": [], "transformer": []
    }
    current = "bus"   # first data section starts right after the 3-line header

    for line in lines[3:]:
        stripped = line.strip()
        if not stripped:
            continue

        # Section-change sentinel lines start with "0 /"
        if stripped.startswith("0 /") or stripped == "0":
            if "END OF BUS DATA" in stripped:
                current = "load"
            elif "END OF LOAD DATA" in stripped:
                current = None          # fixed shunt section (skip)
            elif "END OF FIXED SHUNT DATA" in stripped:
                current = "generator"
            elif "END OF GENERATOR DATA" in stripped:
                current = "branch"
            elif "END OF BRANCH DATA" in stripped:
                current = "transformer"
            continue

        if current and current in sections:
            sections[current].append(stripped)

    return sections


# ---------- Bus parser ----------

def parse_bus_line(line):
    """Parse one BUS data line and return a dict."""
    m = re.match(r"^\s*(\d+)\s*,\s*'([^']+)'\s*,\s*(.+)$", line)
    if not m:
        return None
    bus_id = m.group(1).strip()
    name   = m.group(2).strip()
    rest   = [v.strip() for v in m.group(3).strip().split(",")]
    if len(rest) < 11:
        return None
    return {
        "I":     bus_id,
        "NAME":  name,
        "BASKV": rest[0],
        "IDE":   rest[1],
        "AREA":  rest[2],
        "ZONE":  rest[3],
        "OWNER": rest[4],
        "VM":    rest[5],
        "VA":    rest[6],
        "NVHI":  rest[7],
        "NVLO":  rest[8],
        "EVHI":  rest[9],
        "EVLO":  rest[10],
    }


def collect_bus_ids(lines):
    """Return the set of bus IDs (first field) from any section's lines."""
    ids = set()
    for line in lines:
        parts = line.split(",")
        if parts:
            try:
                ids.add(int(parts[0].strip()))
            except ValueError:
                pass
    return ids


def collect_transformer_bus_ids(lines):
    """
    Transformer records span 4 lines each.
    The first line contains I, J, K bus IDs in fields 0-2.
    """
    ids = set()
    for i, line in enumerate(lines):
        if i % 4 == 0:
            parts = line.split(",")
            for p in parts[:3]:
                try:
                    v = int(p.strip())
                    if v != 0:
                        ids.add(v)
                except ValueError:
                    pass
    return ids


# ---------- Branch parser ----------

def parse_branch_line(line):
    """Parse one BRANCH data line and return a dict."""
    parts = [v.strip() for v in line.split(",")]
    if len(parts) < 16:
        return None
    keys = [
        "I", "J", "CKT", "R", "X", "B",
        "RATEA", "RATEB", "RATEC",
        "GI", "BI", "GJ", "BJ", "ST", "MET", "LEN"
    ]
    return {k: parts[i] for i, k in enumerate(keys)}


# ---------- Main ----------

def main(input_file):
    print(f"Reading: {input_file}")
    sections = read_sections(input_file)

    # --- Parse bus rows ---
    bus_rows = [r for r in (parse_bus_line(l) for l in sections["bus"]) if r]
    print(f"  BUS     : {len(bus_rows)} rows")

    # --- Collect bus IDs from other sections ---
    gen_ids   = collect_bus_ids(sections["generator"])
    trans_ids = collect_transformer_bus_ids(sections["transformer"])
    # (load_ids are implicitly everything else -> also Substation)

    print(f"  Generator buses   : {len(gen_ids)}")
    print(f"  Transformer buses : {len(trans_ids)}")

    # --- Assign BUSTYPE, NAME_MOD, Facility_name ---
    for row in bus_rows:
        bus_i = int(row["I"])
        if bus_i in gen_ids:
            row["BUSTYPE"] = "발전소"
        else:
            # Transformer buses AND load-only buses are both Substation
            row["BUSTYPE"] = "변전소"
        row["NAME_MOD"] = get_name_mod(row["NAME"])
        row["FULLNAME"] = clean_facility_name(row["NAME_MOD"], row["BUSTYPE"])

    # --- Write bus.csv ---
    out_cols = [
        "I", "NAME", "BASKV", "IDE", "AREA", "ZONE", "OWNER",
        "VM", "VA", "NVHI", "NVLO", "EVHI", "EVLO",
        "NAME_MOD", "BUSTYPE", "FULLNAME"
    ]
    rename = {"I": "Node", "FULLNAME": "Facility_name"}

    output_bus = os.path.join(os.getcwd(), "bus.csv")
    with open(output_bus, "w", newline="", encoding="utf-8-sig") as f:
        header = [rename.get(c, c) for c in out_cols]
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for row in bus_rows:
            writer.writerow({rename.get(k, k): row[k] for k in out_cols})
    print(f"  -> Saved: {output_bus}")

    # --- Parse branch rows ---
    branch_rows = [
        r for r in (parse_branch_line(l) for l in sections["branch"]) if r
    ]
    print(f"  BRANCH  : {len(branch_rows)} rows")

    # --- Write branch.csv ---
    branch_out_cols = ["I", "J", "R", "X", "B", "LEN"]
    branch_rename   = {"I": "Node1", "J": "Node2"}

    output_branch = os.path.join(os.getcwd(), "branch.csv")
    with open(output_branch, "w", newline="", encoding="utf-8-sig") as f:
        header = [branch_rename.get(c, c) for c in branch_out_cols]
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for row in branch_rows:
            writer.writerow({branch_rename.get(k, k): row[k] for k in branch_out_cols})
    print(f"  -> Saved: {output_branch}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python psse_to_csv.py <input_file.csv>")
        sys.exit(1)
    main(sys.argv[1])
