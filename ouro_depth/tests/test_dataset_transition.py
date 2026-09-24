"""Dataset provenance survives optimizer checkpoint resume; mismatches stay fatal."""
from copy import deepcopy
import hashlib
from unittest.mock import patch
import pytest
import torch
from ouro_depth.tests.test_s6_direct_decode import make_inputs,reference_worker
from ouro_depth.latent import train_decode as trainer


def test_new_corpus_initialization_and_resume(tmp_path):
 model,_,data,stage1=make_inputs(tmp_path)
 old=hashlib.sha256((data/'manifest.json').read_bytes()).hexdigest()
 (data/'manifest.json').write_text('{"fixture":"new-data"}')
 new=hashlib.sha256((data/'manifest.json').read_bytes()).hexdigest()
 out=tmp_path/'out'
 common=['--mode','stage3','--model-path','tiny','--data-dir',str(data),'--steps','2','--global-batch-size','2','--tbptt','2','--max-prompt-length','8','--max-response-length','5','--save-every','1','--eval-every','2','--eval-records','2','--output-dir',str(out),'--expected-stage1-manifest',old]
 with patch('ouro_depth.latent.teacher.load_teacher',side_effect=lambda *a,**kw:deepcopy(model)),patch('ouro_depth.latent.vllm_rollout.VLLMRollout',reference_worker(model)):
  trainer.main(common+['--stage1-student',str(stage1),'--stop-after','1'])
  trainer.main(common+['--resume',str(out/'checkpoint-000001')])
  d=torch.load(out/'checkpoint-000002/training.pt',weights_only=False)
  assert d['metadata']['data_manifest_sha256']==new
  assert d['metadata']['stage1_data_manifest_sha256']==old
  assert d['metadata']['explicit_dataset_transition'] is True
  (data/'manifest.json').write_text('{"fixture":"unexpected-change"}')
  with pytest.raises(ValueError,match='recipe/data/distribution mismatch'):
   trainer.main(common+['--resume',str(out/'checkpoint-000002')])
