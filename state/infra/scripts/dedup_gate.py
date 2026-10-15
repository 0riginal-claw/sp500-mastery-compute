"""dedup_gate.py — prevents idea drift by detecting duplicate proposals.

Reads SEEN_IDEAS.md, computes Jaccard similarity against new ideas,
rejects duplicates before they reach the planner/ideator.

Usage:
    ideator_output=$(...) 
    echo "$ideator_output" | python3 scripts/dedup_gate.py
    
    # Or pipe directly:
    generate_ideas | python3 scripts/dedup_gate.py
"""

import sys, re, json, os

SEEN_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'SEEN_IDEAS.md')

def load_seen_titles():
    if not os.path.exists(SEEN_FILE):
        return []
    seen = []
    for line in open(SEEN_FILE):
        if '|' in line and 'Timestamp' not in line and '---' not in line:
            parts = line.split('|')
            if len(parts) >= 3:
                seen.append(parts[2].strip().lower())
    return seen

def jaccard_similarity(a: str, b: str) -> float:
    a = re.sub(r'\s+', ' ', a.lower()).strip()
    b = re.sub(r'\s+', ' ', b.lower()).strip()
    s1, s2 = set(a.split()), set(b.split())
    if not s1 or not s2:
        return 0.0
    return len(s1 & s2) / len(s1 | s2)

def main():
    seen_titles = load_seen_titles()
    
    try:
        data = json.load(sys.stdin)
    except json.JSONDecodeError:
        # If stdin isn't JSON, pass through
        print(sys.stdin.read())
        return
    
    candidates = data.get('candidates', [])
    filtered = []
    
    for c in candidates:
        title = c.get('title', '').lower()
        if not title:
            filtered.append(c)
            continue
        
        # Check against all seen titles
        duplicate = False
        for seen in seen_titles:
            sim = jaccard_similarity(title, seen)
            if sim > 0.60:
                print(f"[dedup_gate] BLOCKED: \"{c['title']}\" (similarity={sim:.2f} to \"{seen[:60]}...\")", file=sys.stderr)
                duplicate = True
                break
        
        if not duplicate:
            # Also check against other new candidates
            for c2 in candidates:
                if c2 == c: continue
                if jaccard_similarity(title, c2.get('title', '').lower()) > 0.60:
                    # Keep the first one, skip the duplicate
                    if candidates.index(c) > candidates.index(c2):
                        duplicate = True
                        break
            
        if not duplicate:
            filtered.append(c)
    
    data['candidates'] = filtered
    print(json.dumps(data, indent=2))
    
    if len(candidates) != len(filtered):
        print(f"[dedup_gate] Filtered {len(candidates) - len(filtered)} duplicates. {len(filtered)} remain.", file=sys.stderr)

if __name__ == '__main__':
    main()
