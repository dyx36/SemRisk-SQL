"""One-sided rule discovery following LearnRisk, SIGMOD 2020, section 5.2.

Independently implemented from the published algorithm and an audit of the
author-linked RiskER source. No third-party source or binary is imported.
Class 1 means matching/correct; class 0 means unmatching/wrong. The feature
matrix contains *basic metrics*, not predictions, SQL identities or labels.

The reference implementation uses weighted mass in Eq. (7), unweighted Gini
for accepting leaves, exhaustive per-metric/two-weight expansion, depth 4,
and finally drops risk features observed fewer than 5 times. It does not use
question weighting, a beam, threshold subsampling, a rule cap, a seed search,
or training-activation equivalence as logical redundancy.

Unknown/NA values have no invented numeric route here: nonfinite metrics are
rejected. Defining a faithful semantic encoding is an upstream prerequisite,
not permission to silently impute zero or change the tree algorithm.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Mapping, Sequence

import numpy as np


ARTIFACT_TYPE = "learnrisk_sigmod2020_one_sided_forest"
FLOAT_EPSILON = float(np.finfo(np.float64).eps)
# Dense sklearn 0.21 splitter tolerance; inputs to the author's stump are f32.
FEATURE_THRESHOLD = np.float32(1e-7)


@dataclass(frozen=True)
class OriginalTreeConfig:
    lambda_weight: float = 0.2
    max_depth: int = 4
    class_weight: float = 1000.0
    match_split_gini: float = 0.2
    unmatch_split_gini: float = 0.1
    match_rule_gini: float = 0.2
    unmatch_rule_gini: float = 0.1
    # Post-discovery activation support, NOT min_samples_leaf or Q support.
    min_support: int = 5


def _config(config=None) -> OriginalTreeConfig:
    if config is None:
        cfg = OriginalTreeConfig()
    elif isinstance(config, OriginalTreeConfig):
        cfg = config
    elif isinstance(config, Mapping):
        cfg = OriginalTreeConfig(**config)
    else:
        raise ValueError("config must be OriginalTreeConfig or a mapping")
    for name in ("lambda_weight", "match_split_gini", "unmatch_split_gini",
                 "match_rule_gini", "unmatch_rule_gini", "class_weight"):
        value = getattr(cfg, name)
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("invalid finite configuration number: " + name)
    if not 0 <= cfg.lambda_weight <= 1 or cfg.class_weight < 1:
        raise ValueError("invalid lambda or class weight")
    if any(not 0 <= getattr(cfg, name) <= .5 for name in
           ("match_split_gini", "unmatch_split_gini", "match_rule_gini", "unmatch_rule_gini")):
        raise ValueError("Gini thresholds must be between 0 and 0.5")
    if type(cfg.max_depth) is not int or cfg.max_depth < 1:
        raise ValueError("max_depth must be a positive integer")
    if type(cfg.min_support) is not int or cfg.min_support < 1:
        raise ValueError("min_support must be a positive integer")
    return cfg


def _matrix(x, *, allow_empty=False):
    source = np.asarray(x)
    if source.dtype.kind not in "biuf":
        raise ValueError("basic metrics must be numeric, not strings or objects")
    if source.ndim != 2 or source.shape[1] == 0 or (not allow_empty and source.shape[0] == 0):
        raise ValueError("basic metrics must be a nonempty 2D matrix")
    if not np.isfinite(source).all():
        raise ValueError("nonfinite/unknown metrics require an upstream encoding; no imputation is allowed")
    with np.errstate(over="ignore", invalid="ignore"):
        result = source.astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("basic metrics overflow the reference float32 input representation")
    return result


def _labels(correct, rows):
    result = np.asarray(correct)
    if result.dtype.kind not in "biuf" or result.shape != (rows,) or not np.isin(result, [0, 1]).all():
        raise ValueError("correct must be an aligned binary vector: U=0, M=1")
    return result.astype(np.int8)


def _names(names, columns):
    names = list(names)
    if len(names) != columns or len(set(names)) != columns or any(type(n) is not str or not n for n in names):
        raise ValueError("unique nonempty feature names must match the columns")
    forbidden = {"question_id", "question_index", "candidate_index", "candidate_id", "db_id",
                 "sql_hash", "sql_sha256", "wrong", "correct", "label", "gold", "gold_sql"}
    if forbidden.intersection(names):
        raise ValueError("identities and labels cannot be basic metrics")
    return names


def _gini(count0, count1):
    count0 = np.asarray(count0, dtype=np.float64)
    count1 = np.asarray(count1, dtype=np.float64)
    total = count0 + count1
    return 1.0 - (count0 * count0 + count1 * count1) / (total * total)


def one_sided_gini(left_counts, right_counts, class_weights=(1., 1.), lambda_weight=.2):
    """Reference-code Eq. (7): both Gini and size use weighted class counts."""
    weights = np.asarray(class_weights, dtype=np.float64)
    counts = np.asarray([left_counts, right_counts], dtype=np.float64)
    if weights.shape != (2,) or not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("two positive finite class weights are required")
    if counts.shape != (2, 2) or not np.isfinite(counts).all() or np.any(counts < 0):
        raise ValueError("two nonnegative class counts per child are required")
    if isinstance(lambda_weight, bool) or not isinstance(lambda_weight, (float, int)) or not 0 <= lambda_weight <= 1:
        raise ValueError("invalid lambda")
    weighted = counts * weights
    scores = []
    for count0, count1 in weighted:
        total = count0 + count1
        scores.append(math.inf if total == 0 else lambda_weight / total + (1-lambda_weight) * float(_gini(count0, count1)))
    return min(scores)


def best_split(values, correct, class_weights=(1000., 1.), lambda_weight=.2):
    """Best one-dimensional reference stump, or None if it cannot split.

    All admissible threshold positions are tested, in increasing order. Equal
    scores keep the first position, as the one-feature reference BestSplitter
    does. No threshold quantile budget is imposed. A negative impurity decrease
    prevents a split, as in the author's patched sklearn stump.
    """
    values = np.asarray(values)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("stump values must be a nonempty numeric vector")
    x = _matrix(values.reshape(-1, 1))[:, 0]
    y = _labels(correct, len(x))
    # Also validates the scalar weights/lambda for an otherwise constant input.
    one_sided_gini([1, 0], [0, 1], class_weights, lambda_weight)
    weights = np.asarray(class_weights, dtype=np.float64)
    if len(x) < 2:
        return None
    parent_counts = np.bincount(y, minlength=2)
    weighted_parent = parent_counts * weights
    parent_impurity = float(_gini(*weighted_parent))
    if parent_impurity <= FLOAT_EPSILON:
        return None
    order = np.argsort(x, kind="stable")
    sorted_x, sorted_y = x[order], y[order]
    # sklearn skips successive values within its f32 FEATURE_THRESHOLD.
    if sorted_x[-1] <= np.float32(sorted_x[0] + FEATURE_THRESHOLD):
        return None
    positions = np.flatnonzero(sorted_x[1:] > (sorted_x[:-1] + FEATURE_THRESHOLD)) + 1
    if len(positions) == 0:
        return None
    left1 = np.cumsum(sorted_y, dtype=np.int64)[positions - 1]
    left0 = positions - left1
    right0, right1 = parent_counts[0] - left0, parent_counts[1] - left1
    weighted_left0, weighted_left1 = left0 * weights[0], left1 * weights[1]
    weighted_right0, weighted_right1 = right0 * weights[0], right1 * weights[1]
    left_mass = weighted_left0 + weighted_left1
    right_mass = weighted_right0 + weighted_right1
    score_left = lambda_weight / left_mass + (1-lambda_weight) * _gini(weighted_left0, weighted_left1)
    score_right = lambda_weight / right_mass + (1-lambda_weight) * _gini(weighted_right0, weighted_right1)
    scores = np.minimum(score_left, score_right)
    index = int(np.argmin(scores))
    score = float(scores[index])
    if parent_impurity - score + FLOAT_EPSILON < 0:
        return None
    position = int(positions[index])
    lower, upper = float(sorted_x[position - 1]), float(sorted_x[position])
    threshold = lower / 2.0 + upper / 2.0
    if threshold == upper or not math.isfinite(threshold):
        threshold = lower
    return {
        "threshold": threshold,
        "score": score,
        "parent_impurity": parent_impurity,
        "impurity_improvement": parent_impurity - score,
        "left_counts": [int(left0[index]), int(left1[index])],
        "right_counts": [int(right0[index]), int(right1[index])],
        "left_weighted_mass": float(left_mass[index]),
        "right_weighted_mass": float(right_mass[index]),
        "admissible_thresholds": len(positions),
    }


def _unweighted_stats(counts):
    count0, count1 = counts
    total = count0 + count1
    # The author's Python code re-evaluates this, after undoing class weights.
    impurity = 1.0 - (count0 / total)**2 - (count1 / total)**2
    return {
        "support": total,
        "class_counts": list(counts),
        "predicted_correct": int(count1 > count0),  # argmax tie selects class 0
        "empirical_p_wrong": count0 / total,
        "gini": impurity,
    }


def _canonical_path(path):
    bounds = {}
    for literal in path:
        key = (literal["feature"], literal["op"])
        prior = bounds.get(key)
        if prior is None or (literal["op"] == "gt" and literal["value"] > prior["value"]) or (
                literal["op"] == "le" and literal["value"] < prior["value"]):
            bounds[key] = dict(literal)
    return [bounds[key] for key in sorted(bounds)]


def _rule_key(rule):
    return (rule["predicted_correct"], tuple((p["feature"], p["op"], p["value"]) for p in rule["path"]))


def _dominates(wider, narrower):
    if wider["predicted_correct"] != narrower["predicted_correct"]:
        return False
    tight = {(p["feature"], p["op"]): p["value"] for p in narrower["path"]}
    for item in wider["path"]:
        key = (item["feature"], item["op"])
        if key not in tight:
            return False
        if item["op"] == "gt" and item["value"] > tight[key]:
            return False
        if item["op"] == "le" and item["value"] < tight[key]:
            return False
    return True


def clean_rules(rules):
    """Same-conclusion predicate equality/containment, never activation hashes.

    Numeric predicates are compared structurally, avoiding the author's legacy
    text-substring shortcut. The intended logical containment operation is
    unchanged; source-code string bugs and multiprocessing order are not part
    of the statistical algorithm being reproduced.
    """
    unique = {}
    for raw in rules:
        rule = dict(raw, path=_canonical_path(raw["path"]))
        key = _rule_key(rule)
        if key in unique:
            unique[key]["generations"] += rule.get("generations", 1)
        else:
            rule["generations"] = rule.get("generations", 1)
            unique[key] = rule
    ordered = sorted(unique.values(), key=lambda r: (len(r["path"]), _rule_key(r)))
    retained = []
    removed_candidates = []
    for i, candidate in enumerate(ordered):
        dominator = next((other for j, other in enumerate(ordered)
                          if j != i and _dominates(other, candidate) and
                          (not _dominates(candidate, other) or j < i)), None)
        if dominator is None:
            retained.append(candidate)
        else:
            removed_candidates.append(candidate)
    # Record the final retained dominator, not an intermediate rule that may
    # itself be removed later (e.g. a<=.2, a<=.3, a<=.4).
    removed = []
    for candidate in removed_candidates:
        final_wider = next(rule for rule in retained if _dominates(rule, candidate))
        removed.append({"removed": candidate["path"], "retained_wider": final_wider["path"],
                        "predicted_correct": candidate["predicted_correct"]})
    return retained, {"unique_before_containment": len(unique), "contained_rules_removed": removed}


def discover(x, correct, feature_names: Sequence[str], config=None, checkpoint_dir=None):
    """Fit only on the supplied discovery rows and return binary risk rules.

    This low-level function has no access to fit/validation data or SQL pools.
    Use the separate train entry point to enforce frozen FINER partitions.
    """
    cfg = _config(config)
    x = _matrix(x)
    y = _labels(correct, len(x))
    names = _names(feature_names, x.shape[1])
    generated = []
    splits = []
    attempts = 0
    regimes = ((cfg.class_weight, 1.), (1., cfg.class_weight))

    def construct(rows, path, available, depth):
        nonlocal attempts
        for feature in available:
            for regime_index, weights in enumerate(regimes):
                if depth == 1 and selected_root is not None and (feature, regime_index) != selected_root:
                    continue
                attempts += 1
                split = best_split(x[rows, feature], y[rows], weights, cfg.lambda_weight)
                if split is None:
                    continue
                threshold = split["threshold"]
                # Compare in f64: a Python float threshold must not be rounded
                # back to float32 by NumPy scalar promotion at the midpoint.
                left_mask = x[rows, feature].astype(np.float64) <= threshold
                row_sides = (rows[left_mask], rows[~left_mask])
                stats = [_unweighted_stats(split["left_counts"]), _unweighted_stats(split["right_counts"])]
                pure = [s["gini"] < (cfg.match_split_gini if s["predicted_correct"] else cfg.unmatch_split_gini)
                        for s in stats]
                splits.append({"depth": depth, "feature": feature, "name": names[feature],
                               "path_before_split": [dict(p) for p in path],
                               "class_weight_regime": regime_index, "class_weights": list(weights),
                               **split, "left_unweighted": stats[0], "right_unweighted": stats[1],
                               "accepted_sides": [side for side, ok in zip(("le", "gt"), pure) if ok]})
                for side, is_pure, statistics in zip(("le", "gt"), pure, stats):
                    if is_pure:
                        generated.append({"path": [*path, {"feature": feature, "name": names[feature],
                                                           "op": side, "value": threshold}],
                                          **statistics, "generations": 1})
                if sum(pure) == 1 and depth < cfg.max_depth:
                    remaining = 1 if pure[0] else 0
                    next_available = tuple(f for f in available if f != feature) if "diff" in names[feature] else available
                    literal = {"feature": feature, "name": names[feature],
                               "op": "gt" if remaining else "le", "value": threshold}
                    construct(row_sides[remaining], [*path, literal], next_available, depth + 1)

    selected_root = None
    if checkpoint_dir is None:
        construct(np.arange(len(x)), [], tuple(range(x.shape[1])), 1)
    else:
        from pathlib import Path
        import pickle, hashlib, json, os
        cp=Path(checkpoint_dir);cp.mkdir(parents=True,exist_ok=True)
        identity=hashlib.sha256(x.tobytes()+y.tobytes()+json.dumps([names,asdict(cfg)],sort_keys=True).encode()).hexdigest()
        contract=cp/'contract.txt'
        if contract.exists(): assert contract.read_text()==identity
        else: contract.write_text(identity)
        all_generated=[];all_splits=[];total_attempts=0
        for feature in range(x.shape[1]):
            for regime in range(2):
                selected_root=(feature,regime);p=cp/('%03d_%d.pkl'%selected_root)
                if p.exists():
                    with p.open('rb') as f:generated,splits,attempts=pickle.load(f)
                else:
                    generated=[];splits=[];attempts=0
                    construct(np.arange(len(x)), [], tuple(range(x.shape[1])), 1)
                    tmp=p.with_suffix('.tmp')
                    with tmp.open('wb') as f:pickle.dump((generated,splits,attempts),f)
                    os.replace(tmp,p)
                all_generated.extend(generated);all_splits.extend(splits);total_attempts+=attempts
                print(json.dumps(dict(stage='TREE_ROOT_SAVED',root=selected_root,roots_total=2*x.shape[1])),flush=True)
        generated=all_generated;splits=all_splits;attempts=total_attempts
    filtered = [r for r in generated if r["gini"] <= (
        cfg.match_rule_gini if r["predicted_correct"] else cfg.unmatch_rule_gini)]
    cleaned, cleaning_audit = clean_rules(filtered)
    supported = [r for r in cleaned if r["support"] >= cfg.min_support]
    for i, rule in enumerate(supported):
        rule["id"] = i
        rule["infer_class"] = "M" if rule["predicted_correct"] else "U"
    return {
        "schema_version": 1,
        "artifact_type": ARTIFACT_TYPE,
        "implementation": "independent_numpy_reimplementation_of_author_algorithm",
        "binary_compatibility_claim": False,
        "label_semantics": {"0": "U_wrong", "1": "M_correct"},
        "feature_names": names,
        "config": asdict(cfg),
        "rules": supported,
        "statistics": {"discovery_rows": len(x), "basic_metrics": x.shape[1],
                       "class_counts": np.bincount(y, minlength=2).tolist(),
                       "stump_attempts": attempts, "successful_stumps": len(splits),
                       "generated_leaves": len(generated), "after_final_gini_filter": len(filtered),
                       "after_redundancy_removal": len(cleaned), "effective_risk_features": len(supported),
                       "insufficient_observation_rules": len(cleaned) - len(supported),
                       "support_unit": "raw_candidate_rows_not_questions",
                       "question_balancing": False, "candidate_deduplication": False,
                       "beam_or_rule_cap": False, "threshold_subsampling": False},
        "audit": {"splits": splits, **cleaning_audit},
        "provenance": {
            "paper": "https://arxiv.org/abs/1912.02947",
            "paper_doi": "10.1145/3318464.3380572",
            "author_page": "https://chenbenben.org.cn/risk.html",
            "author_code": "https://chenbenben.org.cn/RiskER/RiskER.zip",
            "criterion_original_entry_sha256": "f806085bac1d36a3e9c0478cb38557fd7dbee6534454bdcd0eaf33ba0e8871b9",
            "tree_original_entry_sha256": "4670e697ec7bf75d1a7ca71ccbfc760884c449187395b1818a40f0f9831fa2df",
            "code_revision": "author ZIP, no published Git commit",
            "known_runtime_differences": [
                "Python/NumPy instead of legacy mysklearn Cython extension; no bitwise binary equivalence claim",
                "logical numeric containment instead of the legacy string-substring shortcut",
                "pure leaves collected immediately per Algorithm 1; does not reproduce a legacy empty-feature-loop leaf-loss bug",
                "finite metrics required; no new missing-value tree route",
            ],
        },
    }


def encode(x, forest):
    """Frozen-rule activation only: not activated never implies the opposite."""
    x = _matrix(x, allow_empty=True)
    if not isinstance(forest, Mapping) or forest.get("artifact_type") != ARTIFACT_TYPE or forest.get("schema_version") != 1:
        raise ValueError("not a LearnRisk original-method forest artifact")
    names = _names(forest.get("feature_names", []), x.shape[1])
    rules = forest.get("rules")
    if not isinstance(rules, list):
        raise ValueError("forest rules must be a list")
    result = np.zeros((len(x), len(rules)), dtype=np.uint8)
    for index, rule in enumerate(rules):
        if rule.get("id") != index or rule.get("predicted_correct") not in (0, 1):
            raise ValueError("invalid rule identity or conclusion")
        path = rule.get("path")
        if not isinstance(path, list) or not path:
            raise ValueError("risk rule must have a nonempty path")
        active = np.ones(len(x), dtype=bool)
        for item in path:
            feature = item.get("feature")
            threshold = item.get("value")
            if type(feature) is not int or not 0 <= feature < len(names) or item.get("name") != names[feature]:
                raise ValueError("rule feature identity drift")
            if type(threshold) not in (float, int) or not math.isfinite(threshold):
                raise ValueError("invalid rule threshold")
            value = x[:, feature].astype(np.float64)
            if item.get("op") == "le":
                active &= value <= threshold
            elif item.get("op") == "gt":
                active &= value > threshold
            else:
                raise ValueError("only reference <= and > predicates are permitted")
        result[:, index] = active
    return result
