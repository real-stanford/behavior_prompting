"""
Check task similarity for LIBEROGen splits.

Usage:
    python check_bddl_similarity_liberogen.py --split liberogen_chain [--simple]
    python check_bddl_similarity_liberogen.py --split liberogen_combination [--simple]

liberogen_chain
    Checks libero_goal_chain_selected_view against:
      libero_goal_chain_firststep_view
      libero_goal_chain_secondstep_view
      libero_goal_chain_selected_inverse_view

    Reports per unseen task:
      - Similar train chains (same 1st step, different 2nd step)
      - Similar train single-step tasks (2nd step only match)

    Example output:
      Unseen: [1st] Open the middle layer of the drawer and then [2nd] put the cream cheese on the stove
        Similar train chain (same 1st, different 2nd) [3]:
        -  Open the middle layer of the drawer and put the cream cheese inside
        -  Open the middle layer of the drawer and then put the cream cheese in front of the stove
        -  Open the middle layer of the drawer and then put the cream cheese on the top of the drawer
        Similar train single-step (2nd step match with wooden cabinet middle open): not present
        Similar train single-step (2nd step match with diff background object config): present

liberogen_combination
    Checks libero_spatial_selected_combinations_view against:
      libero_spatial_selected_combinations_inverse_view

    Each task involves grasping one of two identical bowls (pick location) and placing
    it at a target location (place target), both read from the BDDL goal predicate.
    Only source tasks with a corresponding _demo.hdf5 dataset are counted.
    Reports per unseen task:
      - Same pick location, different place target (training count)
      - Different pick location, same place target (training count)
    With --simple: shows counts only, deduplicates tasks by (pick, place).
    Without --simple: shows full task lists and exact matches.

    Example output (--simple):
      Unseen: Pick the bowl from table center and place it on the cookies box
        Same pick (bowl from table center), different place — Training: [9 tasks]
        Different pick, same place (bowl on the cookies box) — Training: [8 tasks]
"""

import argparse
import os
import re
import sys
import yaml


SPLITS = {
    "liberogen_chain": {
        "view": "libero_goal_chain_selected_view",
        "source_splits": [
            "libero_goal_chain_firststep_view",
            "libero_goal_chain_secondstep_view",
            "libero_goal_chain_selected_inverse_view",
        ],
    },
    "liberogen_combination": {
        "view": "libero_spatial_selected_combinations_view",
        "source_splits": [
            "libero_spatial_selected_combinations_inverse_view",
        ],
    },
}


# ---------------------------------------------------------------------------
# BDDL path resolution
# ---------------------------------------------------------------------------

def get_datasets_path():
    try:
        from libero.libero import get_libero_path
        return get_libero_path("datasets")
    except Exception:
        return None


def has_dataset(datasets_dir, split_name, task_name):
    if datasets_dir is None:
        return False
    path = os.path.join(datasets_dir, split_name, f"{task_name}_demo.hdf5")
    return os.path.exists(path)


def get_bddl_path():
    from libero.libero import get_libero_path
    return get_libero_path("bddl_files")


# ---------------------------------------------------------------------------
# Shared BDDL parsing helpers
# ---------------------------------------------------------------------------

def load_metadata(bddl_path, split_name):
    path = os.path.join(bddl_path, split_name, "task_metadata.yaml")
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return yaml.safe_load(f)


def read_language(bddl_path, split_name, task_name):
    path = os.path.join(bddl_path, split_name, f"{task_name}.bddl")
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        for line in f:
            m = re.match(r"\s*\(:language\s+(.+?)\s*\)\s*$", line)
            if m:
                return m.group(1)
    return None


def _parse_init(bddl_content):
    """Parse :init section into (locations dict, open set, turnon set)."""
    init_match = re.search(r"\(:init(.*?)(?=\)\s*\(:goal|\)\s*$)", bddl_content, re.S)
    if not init_match:
        return {}, set(), set()
    text = init_match.group(1)
    locations = {}
    for m in re.finditer(r"\(On\s+(\w+)\s+(\w+)\)", text):
        locations[m.group(1)] = m.group(2)
    for m in re.finditer(r"\(In\s+(\w+)\s+(\w+)\)", text):
        locations[m.group(1)] = m.group(2)
    open_set = {m.group(1) for m in re.finditer(r"\(Open\s+(\w+)\)", text)}
    turnon_set = {m.group(1) for m in re.finditer(r"\(Turnon\s+(\w+)\)", text)}
    return locations, open_set, turnon_set


def _strip_main_table(loc):
    return re.sub(r"^main_table_", "", loc) if loc else loc


# ---------------------------------------------------------------------------
# Chain similarity logic
# ---------------------------------------------------------------------------

def parse_chain_steps(execution_steps):
    """Split execution_steps into (first_steps, second_steps) at the last Grasp boundary.

    Returns ([], execution_steps) when there is no preceding step (plain pick-and-place).
    """
    last_grasp_idx = None
    for i, step in enumerate(execution_steps):
        if step.startswith("Grasp "):
            last_grasp_idx = i
    if last_grasp_idx is None or last_grasp_idx == 0:
        return [], list(execution_steps)
    return list(execution_steps[:last_grasp_idx]), list(execution_steps[last_grasp_idx:])


def tasks_are_similar(target_first, target_second, candidate_steps, cand_meta=None, all_task_steps=None):
    """Return True if candidate_steps is similar to a chained task with target_first/target_second.

    Case 1 - candidate is also chained: first steps must match exactly AND both second steps
             pick the same object (destination may differ).
    Case 2 - candidate is not chained: steps exactly match the target's second step AND its
             init state reflects the post-first-step environment (derived_from_chain=True).
    """
    cand_first, cand_second = parse_chain_steps(candidate_steps)
    if not cand_second or not cand_second[0].startswith("Grasp "):
        return False
    target_obj = target_second[0].split(" ")[1]
    cand_obj = cand_second[0].split(" ")[1]
    if len(cand_first) == 0:
        if cand_second != target_second:
            return False
        if cand_meta is None or all_task_steps is None:
            return True
        if not cand_meta.get("derived_from_chain", False):
            return False
        full_chain_task = cand_meta.get("derived_from_full_chain_task")
        if full_chain_task is None:
            return False
        chain_steps = all_task_steps.get(full_chain_task)
        if chain_steps is None:
            return False
        chain_first, chain_second = parse_chain_steps(chain_steps)
        return chain_first == target_first and chain_second == target_second
    else:
        return cand_first == target_first and cand_obj == target_obj


def compute_first_step_effects(target_first):
    moved, open_set, turnon_set = {}, set(), set()
    i = 0
    while i < len(target_first):
        step = target_first[i]
        if step.startswith("Open "):
            open_set.add(step.split(" ", 1)[1])
        elif step.startswith("Turnon "):
            turnon_set.add(step.split(" ", 1)[1])
        elif step.startswith(("Grasp ", "Touch ")):
            obj = step.split(" ", 1)[1]
            if i + 1 < len(target_first) and target_first[i + 1].startswith("Place "):
                parts = target_first[i + 1].split()
                moved[obj] = parts[2]
                i += 1
        i += 1
    return moved, open_set, turnon_set


def init_matches_post_first_step(effects, cand_bddl_content):
    moved, open_set, turnon_set = effects
    locations, cand_open, cand_turnon = _parse_init(cand_bddl_content)
    if not open_set.issubset(cand_open):
        return False
    if not turnon_set.issubset(cand_turnon):
        return False
    for obj, expected_loc in moved.items():
        if locations.get(obj) != expected_loc:
            return False
    return True


def compute_chain_similarity(view_name, source_split_names, bddl_path, datasets_dir):
    view_meta = load_metadata(bddl_path, view_name)
    if view_meta is None:
        raise FileNotFoundError(f"Metadata not found: {os.path.join(bddl_path, view_name)}")
    view_tasks = view_meta.get("tasks", {})

    view_splits = [s for s in source_split_names if s.endswith("_view")]

    all_task_steps = {}
    for split_name in source_split_names:
        meta = load_metadata(bddl_path, split_name)
        if meta is None:
            print(f"Warning: metadata not found for split '{split_name}', skipping.", file=sys.stderr)
            continue
        for task_name, task_meta in meta.get("tasks", {}).items():
            if task_name not in all_task_steps:
                all_task_steps[task_name] = task_meta.get("execution_steps", [])

    source_tasks = {}
    for split_name in source_split_names:
        meta = load_metadata(bddl_path, split_name)
        if meta is None:
            continue
        for task_name, task_meta in meta.get("tasks", {}).items():
            if not has_dataset(datasets_dir, split_name, task_name):
                continue
            source_tasks[(split_name, task_name)] = task_meta

    task_in_views = {}
    for vs in view_splits:
        meta = load_metadata(bddl_path, vs)
        if meta is None:
            continue
        for task_name in meta.get("tasks", {}).keys():
            task_in_views.setdefault(task_name, []).append(vs)

    records = []
    for target_task in sorted(view_tasks.keys()):
        target_steps = view_tasks[target_task].get("execution_steps", [])
        target_first, target_second = parse_chain_steps(target_steps)
        if len(target_first) == 0:
            continue

        first_step_effects = compute_first_step_effects(target_first)

        grasped_obj = None
        grasped_obj_expected_loc = None
        if target_second and target_second[0].startswith("Grasp "):
            grasped_obj = target_second[0].split(" ", 1)[1]
            chain_bddl_file = os.path.join(bddl_path, view_name, f"{target_task}.bddl")
            if os.path.exists(chain_bddl_file):
                with open(chain_bddl_file) as f:
                    chain_bddl = f.read()
                chain_locs, _, _ = _parse_init(chain_bddl)
                grasped_obj_expected_loc = chain_locs.get(grasped_obj)

        similar, similar_sso = [], []
        for (split_name, task_name), cand_meta in sorted(source_tasks.items()):
            cand_steps = cand_meta.get("execution_steps", [])
            if list(cand_steps) == list(target_steps):
                continue
            is_full = tasks_are_similar(
                target_first, target_second, cand_steps,
                cand_meta=cand_meta, all_task_steps=all_task_steps,
            )
            cand_first_tmp, cand_second_tmp = parse_chain_steps(cand_steps)
            is_sso = (
                not is_full
                and len(cand_first_tmp) == 0
                and bool(cand_second_tmp)
                and cand_second_tmp[0].startswith("Grasp ")
                and cand_second_tmp == target_second
            )
            if not is_full and not is_sso:
                continue
            views_list = sorted(task_in_views.get(task_name, []))
            lang = read_language(bddl_path, split_name, task_name)
            if is_full:
                kind = "chained" if len(cand_first_tmp) > 0 else "non-chained"
                similar.append((kind, split_name, task_name, views_list, lang))
            else:
                cand_bddl_file = os.path.join(bddl_path, split_name, f"{task_name}.bddl")
                init_match = False
                grasped_obj_loc_match = False
                if os.path.exists(cand_bddl_file):
                    with open(cand_bddl_file) as f:
                        cand_bddl = f.read()
                    init_match = init_matches_post_first_step(first_step_effects, cand_bddl)
                    if grasped_obj and grasped_obj_expected_loc is not None:
                        cand_locs, _, _ = _parse_init(cand_bddl)
                        grasped_obj_loc_match = (cand_locs.get(grasped_obj) == grasped_obj_expected_loc)
                similar_sso.append((split_name, task_name, views_list, lang, init_match, grasped_obj_loc_match))

        language = read_language(bddl_path, view_name, target_task)
        records.append(dict(
            task_name=target_task,
            language=language,
            target_first=target_first,
            target_second=target_second,
            grasped_obj=grasped_obj,
            similar=similar,
            similar_sso=similar_sso,
        ))
    return records


def _drop_region_suffix(s):
    return re.sub(r"\s+region$", "", s).strip()


_NAME_ALIASES = {
    "akita black bowl": "bowl",
}


def _clean_tok(tok):
    tok = re.sub(r"_\d+", "", tok)
    tok = re.sub(r"^main_table_", "", tok)
    tok = tok.replace("_", " ")
    tok = re.sub(r"\bTurnon\b", "Turn on", tok)
    return _NAME_ALIASES.get(tok, tok)


def fmt_steps(steps):
    cleaned = []
    for step in steps:
        parts = step.split()
        verb = _clean_tok(parts[0])
        args = [_clean_tok(t) for t in parts[1:]]
        if verb == "Place" and len(args) >= 2:
            cleaned.append(f"{verb} {args[0]} on {' '.join(args[1:])}")
        else:
            cleaned.append(" ".join([verb] + args))
    return " → ".join(cleaned)


def describe_first_step_result(target_first):
    if not target_first:
        return "step 1 done"
    step = target_first[0]
    if step.startswith("Turnon "):
        obj = _drop_region_suffix(_clean_tok(step.split(" ", 1)[1]))
        return f"{obj} turned on"
    if step.startswith("Open "):
        region = _drop_region_suffix(_clean_tok(step.split(" ", 1)[1]))
        return f"{region} open"
    for s in target_first:
        if s.startswith("Place "):
            parts = s.split()
            obj = _clean_tok(parts[1])
            loc = _drop_region_suffix(_clean_tok(parts[2]))
            return f"{obj} on {loc}"
    return fmt_steps(target_first)


def print_chain_similarity(records, view_name, source_split_names, bddl_path):
    print("\n" + "=" * 80)
    print(f"Chain Similarity Report: {view_name}")
    print(f"BDDL path : {bddl_path}")
    print(f"Checking against: {', '.join(sorted(source_split_names))}")
    print("=" * 80)

    for r in records:
        if r["language"] and " and then " in r["language"]:
            parts = r["language"].split(" and then ", 1)
            title = f"Unseen: [1st] {parts[0]} and then [2nd] {parts[1]}"
        elif r["language"]:
            title = f"Unseen: {r['language']}"
        else:
            title = r["task_name"]
        print(f"\n{title}")

        print(f"  Training: similar chain (same 1st, different 2nd) [{len(r['similar'])} tasks]:")
        if not r["similar"]:
            print("    (none)")
        else:
            for kind in ("chained", "non-chained"):
                for entry in sorted(
                    (e for e in r["similar"] if e[0] == kind), key=lambda e: e[2]
                ):
                    _, _, task_name, _, lang = entry
                    label = lang if lang else task_name
                    print(f"  -  {label}")

        sso_matched = [(s, t, v, l, m, g) for s, t, v, l, m, g in r["similar_sso"] if m]
        sso_fresh   = [(s, t, v, l, m, g) for s, t, v, l, m, g in r["similar_sso"] if not m and g]

        first_ctx = describe_first_step_result(r["target_first"])
        print(f"  Training: similar single-step (2nd step match with {first_ctx}): {'present' if sso_matched else 'not present'}")
        print(f"  Training: similar single-step (2nd step match with diff background object config): {'present' if sso_fresh else 'not present'}")


# ---------------------------------------------------------------------------
# Combination similarity logic
# ---------------------------------------------------------------------------

_OBJ_CLEAN = {
    "glazed_rim_porcelain_ramekin": "ramekin",
    "flat_stove": "stove",
    "akita_black_bowl": "bowl",
    "wooden_cabinet": "wooden cabinet",
    "cookies": "cookies",
    "plate": "plate",
}


def _clean_location(loc):
    """Return a human-readable location/object name."""
    if loc is None:
        return loc
    s = re.sub(r"^main_table_", "", loc)
    s = re.sub(r"_resized$", "", s)
    s = re.sub(r"_region$", "", s)
    m = re.match(r"^(.+?)_(\d+)(?:_(.+))?$", s)
    if m:
        obj, sub = m.group(1), m.group(3)
        obj_clean = _OBJ_CLEAN.get(obj, obj.replace("_", " "))
        if sub:
            if obj_clean == "stove":
                return obj_clean
            return f"{obj_clean} {sub.replace('_', ' ')}"
        return obj_clean
    s = s.replace("_", " ")
    s = re.sub(r"^between (\S+) (\S+)$", r"between \1 and \2", s)
    return s


def _parse_combination_goal(bddl_content):
    """Return (picked_bowl_instance, place_target) from the BDDL goal predicate."""
    m = re.search(r"\(:goal\s*\(And\s*\(On\s+(akita_black_bowl_\d+)\s+(\S+?)\)\s*\)\s*\)", bddl_content)
    if m:
        return m.group(1), m.group(2).rstrip(")")
    return None, None


def _load_combination_task(bddl_path, split_name, task_name):
    """Return (pick_location, place_target, language) for a combination task."""
    path = os.path.join(bddl_path, split_name, f"{task_name}.bddl")
    if not os.path.exists(path):
        return None, None, None
    with open(path) as f:
        content = f.read()
    picked_bowl, place_target = _parse_combination_goal(content)
    if picked_bowl is None:
        return None, None, None
    locations, _, _ = _parse_init(content)
    pick_location = locations.get(picked_bowl)
    lang = None
    for line in content.splitlines():
        m = re.match(r"\s*\(:language\s+(.+?)\s*\)\s*$", line)
        if m:
            lang = m.group(1)
            break
    return pick_location, place_target, lang


def compute_combination_similarity(view_name, source_split_names, bddl_path, datasets_dir, simple=False):
    view_meta = load_metadata(bddl_path, view_name)
    if view_meta is None:
        raise FileNotFoundError(f"Metadata not found: {os.path.join(bddl_path, view_name)}")

    source_tasks = []
    seen_combos = set()
    for split_name in source_split_names:
        meta = load_metadata(bddl_path, split_name)
        if meta is None:
            print(f"Warning: metadata not found for split '{split_name}', skipping.", file=sys.stderr)
            continue
        for task_name in sorted(meta.get("tasks", {}).keys()):
            if not has_dataset(datasets_dir, split_name, task_name):
                continue
            pick_loc, place_tgt, lang = _load_combination_task(bddl_path, split_name, task_name)
            if pick_loc is None:
                continue
            key = (pick_loc, place_tgt)
            if simple and key in seen_combos:
                continue
            seen_combos.add(key)
            source_tasks.append((split_name, task_name, pick_loc, place_tgt, lang))

    records = []
    for target_task in sorted(view_meta.get("tasks", {}).keys()):
        pick_loc, place_tgt, language = _load_combination_task(bddl_path, view_name, target_task)
        if pick_loc is None:
            continue

        same_pick_diff_place = []
        same_place_diff_pick = []
        exact_match = []

        for (split_name, task_name, cand_pick, cand_place, lang) in source_tasks:
            pick_matches = (cand_pick == pick_loc)
            place_matches = (cand_place == place_tgt)
            if pick_matches and place_matches:
                exact_match.append((split_name, task_name, lang))
            elif pick_matches:
                same_pick_diff_place.append((split_name, task_name, cand_place, lang))
            elif place_matches:
                same_place_diff_pick.append((split_name, task_name, cand_pick, lang))

        records.append(dict(
            task_name=target_task,
            language=language,
            pick_location=pick_loc,
            place_target=place_tgt,
            same_pick_diff_place=same_pick_diff_place,
            same_place_diff_pick=same_place_diff_pick,
            exact_match=exact_match,
        ))
    return records


def _extract_place_desc(lang):
    """Extract 'on the plate' from 'Pick the akita black bowl ... and place it on the plate'."""
    if lang is None:
        return None
    m = re.search(r" and place it (.+)$", lang, re.IGNORECASE)
    return m.group(1) if m else None


def _extract_pick_desc(lang):
    """Extract 'on the stove' from 'Pick the akita black bowl on the stove and place it ...'."""
    if lang is None:
        return None
    m = re.search(r"Pick the akita black bowl (.+?) and place it", lang, re.IGNORECASE)
    return m.group(1) if m else None


def print_combination_similarity(records, view_name, source_split_names, bddl_path, simple=False):
    print("\n" + "=" * 80)
    print(f"Combination Similarity Report: {view_name}")
    print(f"BDDL path : {bddl_path}")
    print(f"Checking against: {', '.join(sorted(source_split_names))}")
    print("=" * 80)

    for r in records:
        lang_clean = r["language"].replace("akita black bowl", "bowl") if r["language"] else None
        title = f"Unseen: {lang_clean}" if lang_clean else r["task_name"]
        lang = r["language"].replace("akita black bowl", "bowl") if r["language"] else None
        pick_display = f"bowl {_extract_pick_desc(lang) or _clean_location(r['pick_location'])}"
        place_display = f"bowl {_extract_place_desc(lang) or _clean_location(r['place_target'])}"
        print(f"\n{title}")

        places = [_extract_place_desc(lang.replace("akita black bowl", "bowl") if lang else None) or _clean_location(cand_place)
                  for _, _, cand_place, lang in sorted(r["same_pick_diff_place"], key=lambda x: x[1])]
        if simple:
            print(f"  Training: Same pick ({pick_display}), different place [{len(places)} tasks]")
        else:
            print(f"  Same pick ({pick_display}), different place [{len(places)} tasks]:")
            print(f"    different place: {', '.join(places) if places else '(none)'}")

        picks = [_extract_pick_desc(lang.replace("akita black bowl", "bowl") if lang else None) or _clean_location(cand_pick)
                 for _, _, cand_pick, lang in sorted(r["same_place_diff_pick"], key=lambda x: x[1])]
        if simple:
            print(f"  Training: Different pick, same place ({place_display}) [{len(picks)} tasks]")
        else:
            print(f"  Different pick, same place ({place_display}) [{len(picks)} tasks]:")
            print(f"    different pick: {', '.join(picks) if picks else '(none)'}")

        if not simple:
            print(f"  Exact match (same pick and place) [{len(r['exact_match'])}]:")
            if not r["exact_match"]:
                print("    (none)")
            else:
                for _, _, lang in sorted(r["exact_match"], key=lambda x: x[1]):
                    label = lang if lang else _
                    print(f"  -  {label}")



# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Check task similarity for LIBEROGen splits.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--split", required=True, choices=["liberogen_chain", "liberogen_combination"],
        help="Which LIBEROGen split to analyze.",
    )
    parser.add_argument(
        "--simple", action="store_true",
        help="Compact output: for combination, omits the exact match section.",
    )
    args = parser.parse_args()

    bddl_path = get_bddl_path()
    if not os.path.isdir(bddl_path):
        print(f"Error: bddl_files directory not found: {bddl_path}", file=sys.stderr)
        sys.exit(1)

    datasets_dir = get_datasets_path()
    if datasets_dir is None:
        print("Error: datasets directory not found (install libero to auto-resolve).", file=sys.stderr)
        sys.exit(1)

    cfg = SPLITS[args.split]
    view_name = cfg["view"]
    source_splits = cfg["source_splits"]

    if args.split == "liberogen_chain":
        records = compute_chain_similarity(view_name, source_splits, bddl_path, datasets_dir)
        print_chain_similarity(records, view_name, source_splits, bddl_path)
    else:
        records = compute_combination_similarity(view_name, source_splits, bddl_path, datasets_dir, simple=args.simple)
        print_combination_similarity(records, view_name, source_splits, bddl_path, simple=args.simple)
