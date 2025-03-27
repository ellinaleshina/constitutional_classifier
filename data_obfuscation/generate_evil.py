"""
Generates attacks from .txt file with prompts using h4rm3l package. 
Returns .csv file with original prompt, obfuscated prompt and attack type
"""

from h4rm3l.decorators import make_prompt_decorator
from argparse import Namespace
import pandas as pd
from tqdm.auto import tqdm

primitives = pd.read_csv("primitivs.csv") # You shuld run get_primitives.py first
train_primitives = primitives[primitives['test'] == 0]


class AttackGenerator:
    def __init__(self, programs: list[str]):
        self.programs = programs
        self.args = Namespace(
            decorator_syntax_version=2,
            synthesis_model_name="gpt-3.5-turbo"
            )
        self.attacks = [make_prompt_decorator(program, credentials=None, args=self.args) for program in programs]

    def make_one_type_attacks(self, prompt: str) -> list[str]:
        decorated_prompts = []
        for attack in self.attacks:
            decorated_prompt = attack(prompt)
            decorated_prompts.append(decorated_prompt)
        return decorated_prompts


data_path = "/workspace/constitutions/input_data/Tarantino_2.txt"
output_path = "/workspace/constitutions/obfuscated_data/Tarantino_3_corrupted.csv"

OUT_TYPE = 'csv'
attacker = AttackGenerator(train_primitives.primitiv.tolist())
prompt_nature = 'Tarantino'


if OUT_TYPE == 'txt':
    with open(data_path, 'r') as ipt:
        with open(output_path, "w+") as out:
            for line in ipt.read().splitlines():
                dec = attacker.make_one_type_attacks(line)
                for d, name in zip(dec, train_primitives.primitiv.tolist()):
                    out.write("------------------------------------\n")
                    out.write(name + '\n')
                    out.write("------------------------------------\n")
                    out.write(d + '\n')
                    out.write('\n\n')



elif OUT_TYPE == 'csv':
    inds = train_primitives.index.tolist()
    obfuscated = []
    original = []
    indices = []
    with open(data_path, 'r') as ipt:
        for line in tqdm(ipt.read().splitlines()):
            dec = attacker.make_one_type_attacks(line)
            obfuscated.extend(dec)
            original.extend([line] * len(dec))
            indices.extend(inds)
        
    out_df = pd.DataFrame({
        "obfuscated_prompt" : obfuscated,
        "original_prompt" : original,
        "primitiv_index" : indices,
        "class" : prompt_nature
    })

    out_df.to_csv(output_path, index=False)
            
                