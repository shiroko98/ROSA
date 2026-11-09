"""The follow-up can only enqueue inference, inheriting the same matrix."""
import unittest
from unittest.mock import patch

from rwkv_rosa.native.gpu_needle_followup import followup_command, same_process_alive, three_way_report


class FollowupTest(unittest.TestCase):
    def request(self):
        return {'prompt_template':'rwkv','training':False,'train_dir':'/training',
                'checkpoint':'/snapshot/step200.pth','lengths':[8192,192512],
                'depths':[.1,.5,.9],'seeds':[20261109],'max_new_tokens':24,'wandb':True}

    def test_only_trained_chatml_same_seeds_checkpoint_and_scope(self):
        r=self.request();cmd=followup_command(r,'/chatml-output')
        values={cmd[i]:cmd[i+1] for i in range(2,len(cmd)-1,2)}
        self.assertTrue(cmd[1].endswith('/scripts/run_rwkv_gpu_needle.sh'))
        self.assertEqual(values['--prompt-template'],'chatml')
        self.assertEqual(values['--arms'],'trained')
        self.assertEqual(values['--checkpoint'],r['checkpoint'])
        self.assertEqual(values['--lengths'],'8192,192512')
        self.assertEqual(values['--depths'],'0.1,0.5,0.9')
        self.assertEqual(values['--seeds'],'20261109')
        self.assertEqual(values['--max-new-tokens'],'24')
        self.assertEqual(cmd[-1],'--wandb')
        self.assertNotIn('--rosa_resume',cmd)

    def test_rejects_training_or_wrong_template_dependency(self):
        for changes in ({'training':True},{'prompt_template':'chatml'},{'training':None}):
            with self.assertRaises(ValueError):followup_command({**self.request(),**changes},'/out')
        r=self.request();r['wandb']=False
        self.assertNotIn('--wandb',followup_command(r,'/out'))

    def test_three_way_report_keeps_only_requested_variants_and_protocol(self):
        r=self.request();r['source_step']=200
        rows=[{'arm':'base','passkey_match_rate':.1},{'arm':'trained','passkey_match_rate':.2}]
        def report(template):return {'request':{**r,'prompt_template':template},'status':{'status':'complete'},
            'generation_overall':rows,'generation_summary':[]}
        combined=three_way_report(report('rwkv'),report('chatml'))
        self.assertEqual([(x['arm'],x['prompt_template']) for x in combined['summary']],
            [('base','rwkv'),('trained','rwkv'),('trained','chatml')])
        self.assertEqual(combined['status'],'complete')
        other=report('chatml');other['request']['seeds']=[123]
        with self.assertRaisesRegex(ValueError,'seeds'):three_way_report(report('rwkv'),other)
        other=report('chatml');other['status']['status']='failed'
        self.assertEqual(three_way_report(report('rwkv'),other)['status'],'partial')

    @patch('rwkv_rosa.native.gpu_needle_followup.psutil.Process')
    def test_pid_reuse_is_not_same_worker(self, process):
        process.return_value.create_time.return_value=100
        self.assertFalse(same_process_alive(1,99))


if __name__=='__main__':unittest.main()
