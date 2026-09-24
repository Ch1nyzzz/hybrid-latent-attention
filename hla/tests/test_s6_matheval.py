import json
import sys
from types import SimpleNamespace

import pytest

from hla.vllm_latent import matheval


@pytest.mark.parametrize('chunk_size', [0, 2])
def test_batched_decode_matches_serial_with_padding_stops_and_context_limit(chunk_size):
    import torch
    from hla.tests.test_s6_engine import fixture
    from hla.latent.generate import LatentDecoder, BatchedLatentDecoder
    model, student, _, ids = fixture()
    prompts = [ids[:1, :1], ids[1:2, :3], ids[:1, 4:8]]
    serial = LatentDecoder(model, student, max_len=8, prompt_chunk_size=chunk_size)
    batch = BatchedLatentDecoder(model, student, max_len=8, prompt_chunk_size=chunk_size)
    expected = serial.generate(prompts, 6, set())
    assert batch.generate(prompts, 6, set()) == expected
    stop_ids = {expected[0][1]}
    assert batch.generate(prompts, 6, stop_ids) == serial.generate(prompts, 6, stop_ids)


def test_batched_decode_fixed_prefix_logits_and_stopped_history():
    import torch
    from hla.tests.test_s6_engine import fixture
    from hla.latent.generate import LatentDecoder, BatchedLatentDecoder
    from hla.latent.batched_engine import BatchedRollingEngine
    model, student, _, ids = fixture()
    prompts = [ids[:1, :2], ids[1:2, :5], ids[:1, 5:8]]
    batch, logits = BatchedLatentDecoder(model, student, 20, 0).prefill_batch(prompts)
    refs, expected = [], []
    for prompt in prompts:
        engine = BatchedRollingEngine(model, student, False)
        pred, _ = engine.prefill(prompt, last_logits_only=True)
        engine.detach_history()
        refs.append(engine); expected.append(pred[:, -1])
    with torch.no_grad():
        for step in range(4):
            for i in range(3):
                if i != 1 or step == 0:
                    torch.testing.assert_close(logits[i:i+1], expected[i], rtol=2e-5, atol=2e-7)
            tokens = torch.tensor([[3+step], [8+step], [16+step]])
            active = torch.tensor([[True], [False], [True]])
            pred, _ = batch.step(tokens, active); batch.detach_history()
            logits = pred[:, -1]
            assert batch.prefix_mask.sum(-1).tolist() == [3+step, 5, 4+step]
            assert batch.positions.tolist() == [3+step, 5, 4+step]
            for i in (0, 2):
                pred, _ = refs[i].step(tokens[i:i+1]); refs[i].detach_history()
                expected[i] = pred[:, -1]


@pytest.mark.parametrize('incomplete', [False, True])
def test_batched_evaluation_counts_and_rejects_missing_samples(tmp_path, monkeypatch, incomplete):
    data = tmp_path / 'data.jsonl'
    data.write_text('\n'.join(json.dumps(dict(id=i, problem=str(i), answer='1')) for i in range(5)))
    calls = []

    class Engine:
        def __init__(self, **kwargs):
            assert kwargs['max_num_seqs'] == 4
            assert not kwargs['enable_prefix_caching']
            assert not kwargs['enable_chunked_prefill']

        def get_tokenizer(self):
            return SimpleNamespace(eos_token_id=0, convert_tokens_to_ids=lambda _: 0,
                                   apply_chat_template=lambda messages, **kwargs: messages[0]['content'])

        def generate(self, prompts, params):
            calls.append(len(prompts))
            completion = SimpleNamespace(text='answer', finish_reason='stop', token_ids=[1, 2])
            return [SimpleNamespace(outputs=[completion] * (1 if incomplete else params.n)) for _ in prompts]

    monkeypatch.setitem(sys.modules, 'vllm', SimpleNamespace(LLM=Engine, SamplingParams=SimpleNamespace))
    monkeypatch.setattr(matheval, 'grade', lambda *_: True)
    monkeypatch.setattr(sys, 'argv', ['matheval', '--model', 'unused', '--student', 'student-600.pt',
                                    '--data', str(data), '--output', str(tmp_path), '--n', '2',
                                    '--request-batch', '2', '--max-num-seqs', '4'])
    if incomplete:
        with pytest.raises(RuntimeError, match='incomplete'):
            matheval.main()
        assert not (tmp_path / 'summary0.json').exists()
    else:
        matheval.main()
        assert calls == [2, 2, 1]
        rows = [json.loads(x) for x in (tmp_path / 'shard0.jsonl').read_text().splitlines()]
        assert {(r['id'], r['sample']) for r in rows} == {(i, k) for i in range(5) for k in range(2)}
        summary = json.loads((tmp_path / 'summary0.json').read_text())
        assert summary['total_samples'] == 10
        assert summary['n_problems'] == 5
        assert summary['avg_at_n'] == summary['pass_at_n'] == 1.


def test_resume_keeps_completed_pairs_and_rejects_mismatch(tmp_path):
    from hla.latent.eval_resume import load_completed
    samples = [({'id': str(i), 'answer': str(i)}, k) for i in range(3) for k in range(2)]
    protocol = dict(student='student-600.pt', max_new=8192, n=2, shard=0)
    (tmp_path/'resume-protocol.json').write_text(json.dumps(protocol))
    row = dict(id='1', sample=1, gold='1', correct=True, truncated=False, tokens=9, text='answer')
    target = tmp_path/'shard0.jsonl'
    target.write_text(json.dumps(row)+'\n')
    completed, metadata = load_completed(tmp_path, samples, protocol)
    assert completed == [row] and metadata == protocol
    assert len([(r,k) for r,k in samples if (r['id'],k) not in {(x['id'],x['sample']) for x in completed}]) == 5
    with pytest.raises(ValueError, match='protocol'):
        load_completed(tmp_path, samples, dict(protocol, n=4))
    target.write_text((json.dumps(row)+'\n')*2)
    with pytest.raises(ValueError, match='Duplicate'):
        load_completed(tmp_path, samples, protocol)
    target.write_text(json.dumps(dict(row, gold='wrong'))+'\n')
    with pytest.raises(ValueError, match='foreign'):
        load_completed(tmp_path, samples, protocol)


def test_generation_resume_imports_scores_and_only_generates_missing_samples(tmp_path, monkeypatch):
    import torch
    from hla.latent import generate
    from hla.tests.test_s6_engine import fixture
    _, student, _, _ = fixture()
    ckpt=tmp_path/'student-600.pt'
    torch.save(dict(cfg=student.cfg, student=student.state_dict()),ckpt)
    data=tmp_path/'data.jsonl'
    data.write_text('\n'.join(json.dumps(dict(id=str(i), problem=str(i), answer=str(i))) for i in range(3)))
    resume=tmp_path/'resume';resume.mkdir()
    protocol=dict(student=ckpt.name,checkpoint_asset='',base_asset='',loops=4,max_new=4,max_model_len=10,n=2,temperature=0.,top_p=1.,seed=0,prompt_chunk_size=0,shard=0,nshards=1)
    (resume/'resume-protocol.json').write_text(json.dumps(protocol))
    old=dict(id='1',sample=1,gold='1',correct=False,truncated=True,tokens=4,text='saved answer')
    (resume/'shard0.jsonl').write_text(json.dumps(old)+'\n')
    calls=[]
    class Decoder:
        def __init__(self,*args,**kwargs):self.max_len=kwargs['max_len']
        def generate(self,enc,*args):calls.append(len(enc));return [[6] for _ in enc]
    class Tokenizer:
        eos_token_id=0;pad_token_id=0
        def convert_tokens_to_ids(self,_):return 0
        def apply_chat_template(self,*args,**kwargs):return 'prompt'
        def __call__(self,*args,**kwargs):return SimpleNamespace(input_ids=torch.ones(1,1,dtype=torch.long))
        def decode(self,*args,**kwargs):return 'new answer'
    monkeypatch.setattr(generate,'load_teacher',lambda *args,**kwargs:None)
    monkeypatch.setattr(generate.AutoTokenizer,'from_pretrained',lambda *args,**kwargs:Tokenizer())
    monkeypatch.setattr(generate,'BatchedLatentDecoder',Decoder)
    monkeypatch.setattr(generate,'grade',lambda *args:True)
    out=tmp_path/'output'
    monkeypatch.setattr(sys,'argv',['generate','--reference-hf','--model-path','unused','--data',str(data),'--output',str(out),'--student',str(ckpt),'--n','2','--max-new','4','--max-model-len','10','--prompt-chunk-size','0','--batch','4','--batched-latent','--resume-from',str(resume)])
    generate.main()
    rows=[json.loads(x) for x in (out/'shard0.jsonl').read_text().splitlines()]
    summary=json.loads((out/'summary0.json').read_text())
    assert calls==[4,1] and rows[0]==old
    assert len(rows)==len({(r['id'],r['sample']) for r in rows})==6
    assert summary['resumed_samples']==1 and summary['avg_at_n']==5/6
    assert summary['pass_at_n']==1 and summary['trunc_rate']==1/6
    saved=json.loads((out/'resume-protocol.json').read_text())
    assert saved['student']==ckpt.name and saved['elapsed_seconds']>=0
    from hla.latent.eval_resume import load_completed
    samples=[(json.loads(line),k) for line in data.read_text().splitlines() for k in range(2)]
    assert len(load_completed(out,samples,protocol)[0])==6


@pytest.mark.parametrize('elapsed',[-1,float('nan'),float('inf'),'123',True])
def test_resume_rejects_invalid_elapsed_before_generation(tmp_path,elapsed):
    from hla.latent.eval_resume import load_completed
    (tmp_path/'resume-protocol.json').write_text(json.dumps(dict(elapsed_seconds=elapsed)))
    with pytest.raises(ValueError,match='elapsed_seconds'):
        load_completed(tmp_path,[],{})


def test_auto_concurrency_reserves_null_block_and_whole_problems():
    assert matheval.safe_request_batch(909872, 128, 10240, 4) == (22, 88)
    assert matheval.safe_request_batch(909872, 32, 10240, 4) == (8, 32)
    assert matheval.safe_request_batch(81920, 128, 10240, 4) == (1, 4)
    for pool in (None, 40960):
        with pytest.raises(ValueError):
            matheval.safe_request_batch(pool, 128, 10240, 4)
