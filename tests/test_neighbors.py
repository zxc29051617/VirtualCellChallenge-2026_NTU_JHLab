import json
from pathlib import Path

import numpy as np
import pytest

from vcc2026.transfer import apply_neighbor_effects, neighbor_fold_changes, source_effect_table

faiss = pytest.importorskip("faiss")

from vcc2026.neighbors import GeneIndex, neighbour_weights  # noqa: E402


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_search_matches_brute_force_cosine() -> None:
    rng = np.random.default_rng(0)
    vectors = rng.normal(size=(200, 16)).astype(np.float32)
    queries = rng.normal(size=(10, 16)).astype(np.float32)
    names = [f"G{i}" for i in range(200)]

    sims, ids = GeneIndex(names, vectors).search(
        [f"Q{i}" for i in range(10)], queries, 5, exclude_self=False
    )

    truth = _unit(queries) @ _unit(vectors).T
    expected = np.argsort(-truth, axis=1)[:, :5]
    assert ids.tolist() == expected.tolist()
    assert sims == pytest.approx(np.take_along_axis(truth, expected, axis=1), abs=1e-5)


def test_a_query_never_retrieves_itself_and_still_gets_k_neighbours() -> None:
    rng = np.random.default_rng(1)
    vectors = rng.normal(size=(50, 8)).astype(np.float32)
    names = [f"G{i}" for i in range(50)]
    index = GeneIndex(names, vectors)

    _, ids = index.search(names[:10], vectors[:10], 4)

    assert ids.shape == (10, 4)
    for row in range(10):
        assert row not in ids[row]


def test_k_is_capped_by_what_is_left_to_retrieve() -> None:
    vectors = np.eye(3, dtype=np.float32)
    index = GeneIndex(["A", "B", "C"], vectors)
    sims, ids = index.search(["A"], vectors[:1], 10)
    assert ids.shape == (1, 2)
    assert 0 not in ids[0]


def test_uniform_weights_are_a_plain_mean_and_low_temperature_picks_the_closest() -> None:
    sims = np.array([[0.9, 0.5, 0.1]], dtype=np.float32)
    assert neighbour_weights(sims, None)[0].tolist() == pytest.approx([1 / 3] * 3)
    sharp = neighbour_weights(sims, 0.01)[0]
    assert sharp[0] > 0.999
    assert neighbour_weights(sims, 0.3)[0].sum() == pytest.approx(1.0)
    with pytest.raises(ValueError):
        neighbour_weights(sims, 0.0)


def test_effects_come_from_the_query_s_own_cluster() -> None:
    cluster_a = np.array([[1.0, 0.05], [1.0, -0.05], [0.95, 0.0]], dtype=np.float32)
    cluster_b = np.array([[0.05, 1.0], [-0.05, 1.0], [0.0, 0.95]], dtype=np.float32)
    names = ["A1", "A2", "A3", "B1", "B2", "B3"]
    effects = np.array([[1.0, 0.0]] * 3 + [[0.0, -1.0]] * 3, dtype=np.float32)
    index = GeneIndex(names, np.vstack([cluster_a, cluster_b]))

    got = index.average_effects(
        effects, ["QA", "QB"], np.array([[1.0, 0.1], [0.1, 1.0]], dtype=np.float32),
        k=3, temperature=None,
    )

    assert got[0].tolist() == pytest.approx([1.0, 0.0])
    assert got[1].tolist() == pytest.approx([0.0, -1.0])


# --- merging neighbour effects into the target table -------------------------------------

MEASURED = np.array([True, True, False])


def test_fill_only_touches_targets_nothing_measured_and_only_measured_genes() -> None:
    primary = np.array([[2.0, -2.0, 0.0], [0.0, 0.0, 0.0]])
    neighbor = np.array([[9.0, 9.0, 9.0], [1.0, -1.0, 5.0]])
    out, covered, filled, blended = apply_neighbor_effects(
        primary, np.array([True, False]), neighbor, np.array([True, True]), MEASURED,
        weight=0.0, fill=True,
    )
    assert out[0].tolist() == [2.0, -2.0, 0.0]  # measured row untouched
    assert out[1, 2] == 0.0  # a gene the screen never saw gets no opinion
    assert covered.tolist() == [True, True]
    assert (filled, blended) == (1, 0)


def test_fill_matches_the_neighbours_effect_size_to_the_measured_targets() -> None:
    primary = np.array([[2.0, -2.0, 0.0], [0.0, 0.0, 0.0]])
    neighbor = np.array([[0.0, 0.0, 0.0], [0.2, -0.2, 0.0]])
    out, *_ = apply_neighbor_effects(
        primary, np.array([True, False]), neighbor, np.array([True, True]), MEASURED
    )
    assert out[1, :2].tolist() == pytest.approx([2.0, -2.0])


def test_a_target_scgpt_does_not_know_is_left_for_the_fallback() -> None:
    primary = np.zeros((1, 3))
    out, covered, filled, _ = apply_neighbor_effects(
        primary, np.array([False]), np.ones((1, 3)), np.array([False]), MEASURED
    )
    assert covered.tolist() == [False]
    assert filled == 0
    assert out.tolist() == primary.tolist()


def test_weight_mixes_into_measured_targets_on_measured_genes_only() -> None:
    primary = np.array([[2.0, 2.0, 7.0]])
    neighbor = np.array([[0.0, 4.0, 100.0]])
    out, _, filled, blended = apply_neighbor_effects(
        primary, np.array([True]), neighbor, np.array([True]), MEASURED, weight=0.25, fill=False
    )
    assert out[0].tolist() == pytest.approx([1.5, 2.5, 7.0])
    assert (filled, blended) == (0, 1)


def test_weight_zero_and_no_fill_change_nothing() -> None:
    primary = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    out, covered, filled, blended = apply_neighbor_effects(
        primary, np.array([True, False]), np.ones((2, 3)), np.array([True, True]), MEASURED,
        weight=0.0, fill=False,
    )
    assert out.tolist() == primary.tolist()
    assert covered.tolist() == [True, False]
    assert (filled, blended) == (0, 0)


def test_weight_outside_the_unit_interval_is_rejected() -> None:
    with pytest.raises(ValueError):
        apply_neighbor_effects(
            np.zeros((1, 3)), np.array([True]), np.zeros((1, 3)), np.array([True]), MEASURED,
            weight=1.5,
        )


# --- the retrieval end to end, on a screen small enough to check by hand -------------------

SOURCE_GENES = ["G1", "G2", "G3", "X1"]  # X1 is not on the Challenge axis
CHALLENGE_AXIS = ["G1", "G2", "G3", "G4"]  # G4 was never measured


def _write_screen(path: Path) -> None:
    control = np.full(4, 10.0)
    rows = {
        "control": control,
        "A": control * np.array([4.0, 1.0, 1.0, 1.0]),
        "B": control * np.array([3.0, 1.0, 1.0, 1.0]),
        "C": control * np.array([0.25, 1.0, 1.0, 1.0]),
        "D": control * np.array([0.3, 1.0, 1.0, 1.0]),
    }
    np.savez_compressed(
        path,
        means=np.stack(list(rows.values())).astype(np.float32),
        perturbations=np.asarray(list(rows)),
        counts=np.full(len(rows), 100),
        genes=np.asarray(SOURCE_GENES),
    )


def _write_embeddings(path: Path) -> None:
    vectors = {
        "A": [1.0, 0.05, 0.0],
        "B": [1.0, -0.05, 0.0],
        "C": [0.0, 1.0, 0.05],
        "D": [0.0, 1.0, -0.05],
        "T": [0.9, 0.1, 0.0],  # a Challenge target no screen measured, nearest A and B
    }
    np.savez_compressed(
        path,
        genes=np.asarray(list(vectors)),
        vectors=np.asarray(list(vectors.values()), dtype=np.float32),
    )


def test_neighbours_carry_their_effect_onto_the_challenge_axis(tmp_path: Path) -> None:
    _write_screen(tmp_path / "s.npz")
    _write_embeddings(tmp_path / "e.npz")
    labels, effects, _ = source_effect_table(tmp_path / "s.npz")
    row = {label: i for i, label in enumerate(labels)}

    out, retrieved, measured = neighbor_fold_changes(
        tmp_path / "s.npz", tmp_path / "e.npz", ["T", "UNKNOWN"], CHALLENGE_AXIS, k=2,
        temperature=None,
    )

    assert retrieved.tolist() == [True, False]
    assert measured.tolist() == [True, True, True, False]
    expected = (effects[row["A"]] + effects[row["B"]]) / 2
    assert out[0, :3].tolist() == pytest.approx(expected[:3].tolist(), abs=1e-5)
    assert out[0, 0] > 0  # A and B raise G1
    assert out[0, 3] == 0.0  # G4 was never measured
    assert out[1].tolist() == [0.0, 0.0, 0.0, 0.0]


def test_a_measured_target_does_not_retrieve_itself(tmp_path: Path) -> None:
    _write_screen(tmp_path / "s.npz")
    _write_embeddings(tmp_path / "e.npz")
    labels, effects, _ = source_effect_table(tmp_path / "s.npz")
    row = {label: i for i, label in enumerate(labels)}

    out, retrieved, _ = neighbor_fold_changes(
        tmp_path / "s.npz", tmp_path / "e.npz", ["A"], CHALLENGE_AXIS, k=1, temperature=None
    )

    assert retrieved.tolist() == [True]
    assert out[0, :3].tolist() == pytest.approx(effects[row["B"]][:3].tolist(), abs=1e-5)


# --- the scGPT checkpoint reader -----------------------------------------------------------


def test_special_tokens_are_dropped_and_ids_map_to_their_rows(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    from vcc2026.embeddings import cache_gene_embeddings, load_gene_embeddings

    table = torch.arange(20, dtype=torch.float32).reshape(5, 4)
    torch.save({"encoder.embedding.weight": table, "other": torch.zeros(1)}, tmp_path / "best_model.pt")
    vocab = {"<pad>": 0, "<cls>": 1, "<eoc>": 2, "GENEA": 3, "GENEB": 4}
    (tmp_path / "vocab.json").write_text(json.dumps(vocab), encoding="utf-8")

    out = cache_gene_embeddings(tmp_path, tmp_path / "cache.npz")
    genes, vectors = load_gene_embeddings(out)

    assert genes == ["GENEA", "GENEB"]
    assert vectors[0].tolist() == table[3].tolist()
    assert vectors[1].tolist() == table[4].tolist()


def test_a_checkpoint_without_a_gene_table_says_what_it_does_contain(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    from vcc2026.embeddings import load_scgpt_gene_embeddings

    torch.save({"decoder.embed_proj.weight": torch.zeros(2, 2)}, tmp_path / "best_model.pt")
    (tmp_path / "vocab.json").write_text(json.dumps({"G": 0}), encoding="utf-8")

    with pytest.raises(KeyError, match="decoder.embed_proj.weight"):
        load_scgpt_gene_embeddings(tmp_path)
