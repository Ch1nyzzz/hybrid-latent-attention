import json
import random

import pytest

from hla.latent.corpus_index import RecordIndex


@pytest.fixture
def corpus(tmp_path):
    path = tmp_path / "train.jsonl"
    rows = [dict(source=s, record_id=f"{s}:{i}", input_ids=list(range(n)))
            for s, count in (("openr1", 6), ("fineweb", 4))
            for i, n in enumerate([64] * count + [10])]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    index = RecordIndex(path)
    yield index
    index.close()


def test_source_epochs_cover_each_pool_before_repeating(corpus):
    rows = [corpus.sample_at(i, seed=42, stage=2) for i in range(20)]
    assert [r["source"] for r in rows[:5]] == ["openr1"] * 3 + ["fineweb"] * 2
    assert len({r["record_id"] for r in rows[:10]}) == 10
    assert {r["record_id"] for r in rows[:10]} == {r["record_id"] for r in rows[10:]}
    assert all(len(r["input_ids"]) >= 64 for r in rows)


def test_rank_partition_and_fresh_process_resume_select_same_records(corpus):
    def ids(index, positions, stage=2):
        return [index.sample_at(i, seed=42, stage=stage)["record_id"] for i in positions]
    expected = ids(corpus, range(37))
    actual = [None] * 37
    for rank in range(4):
        for position in range(rank, 37, 4):
            actual[position] = ids(corpus, [position])[0]
    assert actual == expected
    resumed = RecordIndex(corpus.path)
    try:
        assert ids(resumed, range(17, 37)) == expected[17:]
        assert ids(resumed, range(10), stage=3) != expected[:10]
    finally:
        resumed.close()


def test_legacy_replacement_draws_and_empty_eligible_pool(corpus):
    a, b = random.Random(17), random.Random(17)
    candidates = corpus.rows["openr1"]
    for _ in range(20):
        while True:
            offset, length = candidates[b.randrange(len(candidates))]
            if length >= 64:
                break
        expected = corpus._read(offset)
        assert corpus.sample("openr1", a, min_length=64) == expected
    with pytest.raises(ValueError, match="No eligible"):
        corpus.sample_at(0, seed=1, stage=1, min_length=100)
    with pytest.raises(ValueError, match="nonnegative"):
        corpus.sample_at(-1, seed=1, stage=1)
