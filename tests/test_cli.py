import subprocess
import sys
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


class CliTests(unittest.TestCase):
    def run_cli(self,*args):
        return subprocess.run([sys.executable,str(ROOT/'tools/infer.py'),*args],capture_output=True,text=True)

    def test_help_does_not_load_models(self):
        result=self.run_cli('--help')
        self.assertEqual(result.returncode,0)
        self.assertIn('--trajectory',result.stdout)

    def test_invalid_conditions_and_parameters_fail_before_loading(self):
        cases=[['--checkpoint','missing.pth'],
               ['--checkpoint','missing.pth','--prompt','walk','--frames','61'],
               ['--checkpoint','missing.pth','--prompt','walk','--tta-steps','400'],
               ['--checkpoint','missing.pth','--prompt','walk','--topk-threshold','1']]
        for args in cases:
            with self.subTest(args=args):
                result=self.run_cli(*args)
                self.assertEqual(result.returncode,2)
                self.assertNotIn('ModuleNotFoundError',result.stderr)


if __name__=='__main__':unittest.main()
