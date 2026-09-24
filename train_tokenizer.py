"""Train CrazeAI's own tokenizer from the supplied corpus.
Run: python train_tokenizer.py --input data.json --output tokenizer.model
"""
import argparse, json, pathlib
import sentencepiece as spm


def collect(path):
    p=pathlib.Path(path)
    if p.suffix.lower()=='.json':
        obj=json.loads(p.read_text(encoding='utf-8'))
        entries=obj.get('entries',obj if isinstance(obj,list) else [])
        lines=[]
        for e in entries:
            if isinstance(e,dict):
                lines += [str(e.get('instruction','')), str(e.get('input','')), str(e.get('output',''))]
            else: lines.append(str(e))
        return '\n'.join(x for x in lines if x.strip())
    return p.read_text(encoding='utf-8')

ap=argparse.ArgumentParser(); ap.add_argument('--input',required=True); ap.add_argument('--output',default='tokenizer.model'); ap.add_argument('--vocab-size',type=int,default=8192); a=ap.parse_args()
text=collect(a.input); corpus=pathlib.Path(a.output).with_suffix('.corpus.txt'); corpus.write_text(text,encoding='utf-8')
prefix=str(pathlib.Path(a.output).with_suffix(''))
spm.SentencePieceTrainer.Train(input=str(corpus), model_prefix=prefix, vocab_size=a.vocab_size, model_type='bpe', character_coverage=0.9995, pad_id=0, bos_id=1, eos_id=2, unk_id=3, byte_fallback=True, normalization_rule_name='identity', shuffle_input_sentence=False)
print(f'Created {a.output} from {len(text):,} characters')
