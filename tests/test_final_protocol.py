import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

import os
project_root = Path(os.environ.get('TEXT_MONITOR_ROOT', Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(project_root / 'src'))
import protocol
from features import Featurizer
from metrics import paired_problem_bootstrap

class FinalProtocolTests(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame([
            {'problem_id': f'g{g}', 'problem_text': f'uniqueword{g}',
             'step_text': 'valid yes' if j % 2 else 'invalid no',
             'label': 1 if j % 2 else -1}
            for g in range(9) for j in range(4)])

    def test_fold_vocabulary_never_sees_held_out_groups(self):
        fitted = []
        original = TfidfVectorizer.fit_transform
        def recording(vec, texts, *args, **kwargs):
            X = original(vec, texts, *args, **kwargs)
            fitted.append(set(vec.vocabulary_))
            return X
        with patch.object(TfidfVectorizer, 'fit_transform', recording):
            result = protocol.fit_and_eval('lr', Featurizer('tfidf-word'),
                self.df, self.df.iloc[:4].assign(problem_text='externaltestonly'))
        self.assertEqual(len(fitted), 4)
        splits = GroupKFold(3).split(self.df, self.df.label, self.df.problem_id)
        for vocabulary, (_, val) in zip(fitted[:3], splits):
            held = self.df.iloc[val].problem_text.unique()
            self.assertTrue(all(t not in vocabulary for t in held))
        self.assertTrue(all('externaltestonly' not in vocab for vocab in fitted))
        self.assertEqual(result['protocol_version'], protocol.PROTOCOL_VERSION)
        self.assertEqual(result['test']['auroc'], 1.0)

    def test_cache_tracks_text_content_and_order(self):
        def key(df): return protocol._full_train_key('tfidf-word', 'lr', 0, 'step', df)
        changed = self.df.copy()
        changed.loc[0, 'step_text'] += ' changed'
        self.assertNotEqual(key(self.df), key(changed))
        self.assertNotEqual(key(self.df), key(self.df.iloc[::-1]))
        with tempfile.TemporaryDirectory() as cache:
            with patch.object(protocol, 'CACHE', cache):
                first = protocol.fit_full_cached('tfidf-word', 'lr', self.df)
                with patch.object(protocol, 'select_C_tfidf', side_effect=AssertionError('cache miss')):
                    second = protocol.fit_full_cached('tfidf-word', 'lr', self.df)
                self.assertEqual(first[2], second[2])

    def test_problem_bootstrap_matches_explicit_whole_problem_duplication(self):
        y = np.array([-1, 1, -1, 1, -1, 1, -1, 1])
        groups = np.array(['a','a','b','b','c','c','d','d'])
        base = np.array([.3,.7,.6,.4,.2,.8,.1,.9])
        model = np.array([.2,.8,.3,.7,.6,.4,.1,.9])
        result = paired_problem_bootstrap(y, {'word':base, 'char':base, 'model':model},
                                          groups, ['word','char'], n_boot=30, seed=9)
        rng = np.random.RandomState(9)
        draws=[]
        for _ in range(30):
            sampled=rng.randint(4,size=4)
            idx=np.concatenate([np.flatnonzero(groups==np.unique(groups)[g]) for g in sampled])
            draws.append(roc_auc_score(y[idx],model[idx])-roc_auc_score(y[idx],base[idx]))
        np.testing.assert_allclose(result['comparisons']['model - word']['ci95'],
                                   np.percentile(draws,[2.5,97.5]),atol=1e-12)
        identical=paired_problem_bootstrap(y, {'word':base,'same':base},groups,['word'],n_boot=10)
        self.assertEqual(identical['comparisons']['same - word']['ci95'], [0.,0.])

    def test_invalid_score_alignment_and_single_class_folds_fail(self):
        with self.assertRaises(ValueError):
            paired_problem_bootstrap([1,-1],{'b':[.1,.2],'m':[.2]},['a','b'],['b'],n_boot=10)
        df=self.df.copy()
        df['label']=np.where(df.problem_id=='g0',1,-1)
        with self.assertRaisesRegex(ValueError, 'one class'):
            protocol.fit_and_eval('lr',Featurizer('tfidf-word'),df,df)

if __name__ == '__main__':
    unittest.main(verbosity=2)
