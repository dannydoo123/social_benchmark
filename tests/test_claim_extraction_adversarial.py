import unittest
from social_benchmark.pipeline.claim_extraction import split_claims
from social_benchmark.pipeline.normalization import normalize_evidence_text

A='GPT-6 Astra'
B='GPT-6.1 Sol'


class AspectRegressionTests(unittest.TestCase):
    def labels(self,text,context=None):
        return {(r['model_id'],r['category'],r['polarity_score']) for r in split_claims(text,context_text=context)}

    def test_aspect_matrix(self):
        for category,words in [('coding','coding'),('design','UI design'),('cost_efficiency','pricing'),('writing','prose'),('research','citations'),('agents','tool calling'),('vibe_coding','vibe coding')]:
            for adjective,sign in [('great',1),('terrible',-1),('not good',-1),('not bad',1)]:
                with self.subTest(category=category,adjective=adjective):
                    self.assertEqual(self.labels(f'{A} is {adjective} for {words}'),{('gpt-6-astra',category,sign)})

    def test_conjunction_has_local_polarity(self):
        self.assertEqual(self.labels(f'{A} is good for coding and terrible at design'),{('gpt-6-astra','coding',1),('gpt-6-astra','design',-1)})

    def test_elliptical_cost(self):
        self.assertEqual(self.labels(f'{A} is great for vibe coding but too expensive'),{('gpt-6-astra','vibe_coding',1),('gpt-6-astra','cost_efficiency',-1)})

    def test_multiple_subjects(self):
        self.assertEqual(self.labels(f'{A} is good for coding but {B} is terrible at design'),{('gpt-6-astra','coding',1),('gpt-6.1-sol','design',-1)})

    def test_direct_parent(self):
        rows=split_claims('It is terrible at coding.',context_text=f'I tested {A}.')
        self.assertEqual(rows[0]['attribution'],'direct_parent_pronoun')
        self.assertEqual(rows[0]['model_id'],'gpt-6-astra')

    def test_parent_ambiguity(self):
        self.assertEqual(self.labels('It is great for coding.',f'{A} versus {B}'),set())

    def test_parent_does_not_label_unrelated_statement(self):
        self.assertEqual(self.labels('My laptop is expensive.',A),set())

    def test_generic_parent_stays_generic(self):
        self.assertEqual(self.labels('It is great for coding.','Claude'),{('claude','coding',1)})

    def test_scope_resets(self):
        for separator in ['\n\n',' My laptop arrived. ']:
            self.assertEqual(len(split_claims(f'{A} is good for coding.{separator}It is expensive.')),1)

    def test_abstentions(self):
        for text in [f'Is {A} good for coding?',f'If {A} is good for coding, I will buy it.',f'{A} might be good for coding.',f'{A} is good and bad for coding.',f'> {A} is great for coding',f'"{A} is great for coding"',f'```{A} is great for coding```',f'{A} costs $5 per month.',f'An unnamed model is better than {A}.',f'I found it excellent compared to {A}.',f'{A} is great for coding /s',f'{A} is not better than {B}.']:
            with self.subTest(text=text):self.assertEqual(self.labels(text),set())

    def test_not_only(self):
        self.assertEqual(self.labels(f'{A} is not only good for coding'),{('gpt-6-astra','coding',1)})

    def test_unicode_negation(self):
        self.assertEqual(self.labels(f'{A} isn\u2019t good for coding'),{('gpt-6-astra','coding',-1)})

    def test_exact_offsets(self):
        text=f'  {A} is great for coding; it is too expensive.\n{B} is bad for design'
        for row in split_claims(text):self.assertEqual(text[row['span_start']:row['span_end']],row['evidence_text'])

    def test_comparison_direction(self):
        self.assertEqual(self.labels(f'{A} is worse than {B} at coding'),{('gpt-6-astra','coding',-1),('gpt-6.1-sol','coding',1)})

    def test_fast_quota_drain_is_not_praise(self):
        self.assertEqual(self.labels(f'{A} burns through our quota fast'),{('gpt-6-astra','cost_efficiency',-1)})

    def test_saved_metadata_is_not_sentiment(self):
        self.assertEqual(self.labels(f'Saved turn metadata records {A}'),set())

    def test_comparison_target_is_not_subject(self):
        self.assertEqual(self.labels(f'Not as fast as {B} on my math puzzle'),set())

    def test_benchmark_table_is_not_user_opinion(self):
        self.assertEqual(self.labels(f'| {A} | nobody lies | 96 | agents |'),set())

    def test_testing_models_is_not_coding(self):
        self.assertEqual(self.labels(f'I wanted to test models and found {A} good at identifying text'),{('gpt-6-astra','general',1)})

    def test_html_quote_does_not_swallow_author_opinion(self):
        text=normalize_evidence_text(f'<blockquote><p>{A} is great for coding</p></blockquote><p>{B} is terrible for design</p>')
        self.assertEqual(self.labels(text),{('gpt-6.1-sol','design',-1)})

    def test_html_code_is_not_an_opinion(self):
        text=normalize_evidence_text(f'<pre><code>{A} is great for coding</code></pre><p>{B} is expensive</p>')
        self.assertEqual(self.labels(text),{('gpt-6.1-sol','cost_efficiency',-1)})
