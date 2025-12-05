#!/usr/bin/env python3
import argparse, json, os, re, sys, xml.etree.ElementTree as ET
from pathlib import Path

# ---- Helpers ---------------------------------------------------------------

REG_REGISTER = re.compile(
    r'registerMatcher\(\s*"(?P<name>[^"]+)"\s*,\s*(?P<expr>[^)]+?)\s*\)\s*;'
)

def base_symbol(expr: str) -> str:
    """Extract a C++ identifier we can search in Doxygen (qualified name if possible)."""
    s = expr.strip()

    # static_cast<...>(IDENT)
    m = re.search(r'static_cast<[^>]+>\s*\(\s*([A-Za-z0-9_:]+)\s*\)', s)
    if m:
        s = m.group(1)

    # Strip call parens if any (we only register function *pointers* or variables)
    s = re.sub(r'\(.*\)$', '', s)

    # For templates: query::matcher::m_GetDefinitions<...>  -> keep qualified + base
    s = re.sub(r'<.*>', '', s)

    # Normalize spaces
    s = re.sub(r'\s+', '', s)

    return s  # e.g., m_Attr  or  query::matcher::m_GetDefinitions

def load_registry_entries(mlir_query_cpp: Path):
    text = mlir_query_cpp.read_text(encoding="utf-8")
    out = []
    for m in REG_REGISTER.finditer(text):
        name = m.group("name")
        expr = m.group("expr")
        sym = base_symbol(expr)
        out.append({"name": name, "symbol": sym})
    return out

def doxygen_index(xml_dir: Path):
    """Iterate (compound_file, memberdef) pairs with useful fields."""
    idx = ET.parse(xml_dir / "index.xml").getroot()
    for comp in idx.findall(".//compound"):
        refid = comp.get("refid")
        comp_xml = xml_dir / f"{refid}.xml"
        if not comp_xml.exists():
            continue
        croot = ET.parse(comp_xml).getroot()
        for m in croot.findall(".//memberdef"):
            yield (refid, m)

def qname_of(member):
    q = member.findtext("qualifiedname") or ""
    return q.strip()

def file_of(member):
    loc = member.find("location")
    return (loc.get("file") if loc is not None else "") or ""

def brief_of(member):
    def text(node): return "".join(node.itertext()).strip() if node is not None else ""
    brief = text(member.find("briefdescription"))
    if not brief:
        brief = text(member.find("detaileddescription"))
    # squish whitespace
    return " ".join(brief.split())

def match_member(symbol: str, cand_qname: str) -> bool:
    # Exact qualified match wins
    if symbol == cand_qname:
        return True
    # If symbol is unqualified (e.g., m_Attr), accept tail match
    if "::" not in symbol and cand_qname.endswith("::" + symbol):
        return True
    # If symbol is partially qualified (query::matcher::m_X)
    if "::" in symbol and cand_qname.endswith(symbol):
        return True
    return False

def build_docs_map(xml_dir: Path, registry):
    """Return name -> {desc, url, header, qualified} using Doxygen XML."""
    # Base URL (adjust if you publish elsewhere)
    base_url = "https://mlir.llvm.org/doxygen"

    # Build a cache of members keyed by name (fast path), but we also check qualifiedname.
    members = list(doxygen_index(xml_dir))

    result = {}
    for ent in registry:
        name, sym = ent["name"], ent["symbol"]
        best = None
        for comp_id, mem in members:
            qn = qname_of(mem)
            nm = mem.findtext("name") or ""
            if not (nm and qn): 
                continue
            if match_member(sym, qn):
                best = (comp_id, mem)
                # Prefer mlir::query::matcher namespace if multiple
                if "mlir::query::matcher" in qn:
                    break
        if best:
            comp_id, mem = best
            desc = brief_of(mem)
            hdr  = file_of(mem)
            member_id = mem.get("id") or ""
            url = f"{base_url}/{comp_id}.html#{member_id}" if member_id else f"{base_url}/{comp_id}.html"
            result[name] = {
                "symbol": sym,
                "qualified": qname_of(mem),
                "header": hdr,
                "desc": desc or "",
                "url": url
            }
        else:
            result[name] = {
                "symbol": sym,
                "qualified": "",
                "header": "",
                "desc": "",
                "url": ""
            }
    return result

def write_markdown(docs_map, out_md: Path):
    rows = []
    for k in sorted(docs_map.keys(), key=str.lower):
        d = docs_map[k]
        link = f"[`{k}`]({d['url']})" if d["url"] else f"`{k}`"
        desc = d["desc"] or "—"
        hdr  = Path(d["header"]).name if d["header"] else "—"
        rows.append((link, hdr, desc))
    with out_md.open("w", encoding="utf-8") as f:
        f.write("## Matcher Reference (auto-generated)\n\n")
        f.write("| Matcher | Declared In | Description |\n|---|---|---|\n")
        for link, hdr, desc in rows:
            f.write(f"| {link} | `{hdr}` | {desc} |\n")

# ---- Main -----------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Generate mlir-query matcher reference from registry + Doxygen XML")
    ap.add_argument("--mlir-query-cpp", required=True, help="Path to mlir/Tools/mlir-query/mlir-query.cpp (or the file that registers matchers)")
    ap.add_argument("--doxy-xml-dir", required=True, help="Path to Doxygen XML dir (Doxyfile: GENERATE_XML=YES, XML_OUTPUT=xml)")
    ap.add_argument("--out-md", required=True, help="Output Markdown file")
    ap.add_argument("--out-json", help="Optional: write a JSON dump alongside")
    args = ap.parse_args()

    reg = load_registry_entries(Path(args.mlir_query_cpp))
    if not reg:
        print("No registerMatcher(...) calls found.", file=sys.stderr)
        sys.exit(2)

    docs_map = build_docs_map(Path(args.doxy_xml_dir), reg)
    write_markdown(docs_map, Path(args.out_md))

    if args.out_json:
        Path(args.out_json).write_text(json.dumps(docs_map, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()
