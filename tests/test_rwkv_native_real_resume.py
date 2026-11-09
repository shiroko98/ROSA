"""Exercise actual Lightning restore order, not a manually called callback hook."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
import pytorch_lightning as pl
from rwkv_rosa.native.integration import install


class RealLightningResumeTest(unittest.TestCase):
    def test_compact_restore_expands_external_base_before_strict_model_load(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp);base=tmp/'base.pth'
            class Toy(pl.LightningModule):
                def __init__(self):
                    super().__init__();self.emb=nn.Embedding(32,16);self.head=nn.Linear(16,32,bias=False)
                    self.bfloat16()
                    if not base.exists():torch.save(self.state_dict(),base)
                    self.load_state_dict(torch.load(base,weights_only=True))
                    args=SimpleNamespace(rosa_cpu=1,rosa_position='post_norm',rosa_layers='',n_layer=1,
                        devices=1,num_nodes=1,strategy='auto',precision='bf16',load_model=str(base),
                        train_stage=0,load_partial=0,resume_ckpt_path=None,rosa_resume='',rosa_scale=.05,
                        rosa_device='cpu',run_name='resume-test',proj_dir=str(tmp),lr_init=1e-3,
                        betas=(.9,.99),adam_eps=1e-8,weight_decay=.01,ctx_len=8,data_file='toy',epoch_steps=8,
                        rosa_checkpoint_adapter_only=1,rosa_windowed_data=0)
                    self.extension_callback=install(self,args)
                def training_step(self,batch,batch_idx):
                    ids=batch[0];self.rosa_cpu.begin(ids)
                    x=self.rosa_cpu.inject_post_norm(self.emb(ids))
                    return self.head(x).float().square().mean()
                def on_save_checkpoint(self,checkpoint):
                    self.extension_callback.on_save_checkpoint(self.trainer,self,checkpoint)
            def trainer(steps):return pl.Trainer(accelerator='cpu',devices=1,max_steps=steps,max_epochs=1,
                logger=False,enable_checkpointing=False,enable_model_summary=False,enable_progress_bar=False,
                num_sanity_val_steps=0)
            loader=DataLoader(TensorDataset(torch.tensor([[1,2,1,2,3,1,2,3]]*8)),batch_size=1)
            torch.manual_seed(42);continuous=Toy();t=trainer(3);t.fit(continuous,loader)
            class SaveAtTwo(pl.Callback):
                def on_train_batch_end(self, trainer, model, outputs, batch, batch_idx):
                    if trainer.global_step==2:
                        model.extension_callback.save_full_checkpoint(trainer,'test')
            torch.manual_seed(42);first=Toy();t1=trainer(3);t1.callbacks.append(SaveAtTwo());t1.fit(first,loader)
            # Exactly the same optimizer-boundary save used by production hourly
            # and user-stop checkpoints, not an already-closed fit-loop state.
            path=tmp/'native-last.ckpt'
            raw=torch.load(path,weights_only=False)
            self.assertTrue(all(k.startswith('rosa_cpu.') for k in raw['state_dict']))
            torch.manual_seed(999);restored=Toy();t2=trainer(3);t2.fit(restored,loader,ckpt_path=path)
            self.assertEqual(t2.global_step,3)
            for key,value in continuous.state_dict().items():
                torch.testing.assert_close(value,restored.state_dict()[key],atol=0,rtol=0)
            for group in ('dense','sparse'):
                a=t.optimizers[0].state_dict()[group]['state'];b=t2.optimizers[0].state_dict()[group]['state']
                self.assertEqual(a.keys(),b.keys())
                for index,state in a.items():
                    for name,value in state.items():
                        if isinstance(value,torch.Tensor):torch.testing.assert_close(value,b[index][name],atol=0,rtol=0)
                        else:self.assertEqual(value,b[index][name])


if __name__=='__main__':unittest.main()
