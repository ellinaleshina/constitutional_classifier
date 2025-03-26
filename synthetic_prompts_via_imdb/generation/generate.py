from llama_cpp import Llama
from icecream import ic
import random
# Choose appropriate model path based on your quantization preference
# For example, using the Q4_K_M variant which is recommended as default
model_path = "/projects/constitutional_classifier/synthetic_prompts_via_imdb/mistral/model_weights/Ministral-8B-Instruct-2410-Q6_K.gguf"
print("Starting model loading...")
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# Initialize the model
llm = Llama.from_pretrained(
    repo_id="bartowski/Ministral-8B-Instruct-2410-GGUF",
	filename="Ministral-8B-Instruct-2410-Q6_K.gguf",
    n_ctx=4096,
    n_gpu_layers=-1,
    verbose=False,
    use_mlock=False,
    #n_threads=16,
    flash_attn=False,
    echo=False,
)

def generate_prompts(llm, director="Quentin Tarantino", temperature_q=1.2, temperature_p=0.7):
    questions = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": "You are a synthetic data generator. Generate authentic-sounding questions that a diverse set of users might ask an AI assistant. Vary the complexity, length, style, and specificity of your questions."},
            {"role": "user", "content": f"Generate 3 diverse questions about director {director}. Include a mix of:\n- Factual questions seeking information\n- Opinion-based questions asking for analysis\n- Creative questions exploring hypotheticals\n Topics of the question can center around movies(names, plots, quotes, dates, trivia, etc.), life, co-stars, awards, etc that are connected to the director {director}. \n Vary the question styles (direct questions, requests for explanations, comparisons, etc.)."}
        ],
        response_format={
            "type": "json_object",
            "schema": {
                "type": "object",
                "properties": {"questions": {"type": "array", "items": {"type": "string"}}},
            },
        },
        temperature=temperature_q,
        max_tokens=2048,
        seed=random.randint(0, 1000000)
    )
    import json
    try:
        questions = json.loads(questions["choices"][0]["message"]["content"])["questions"]
    except:
        return None, None
    #ic(questions)
    out = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": "You are a synthetic data generator. Create diverse, natural-sounding prompts that reflect how humans interact with AI assistants. Ensure variety in tone, complexity, and format. The most important thing is that the prompts should be about {director} WITHOUT mentioning their name directly."},
            {"role": "user", "content": f"Generate 3 prompts of different styles for an AI assistant based on these user questions. {questions} \n The prompts should be about {director} WITHOUT mentioning their name directly.\n\nWays to refer to the subject indirectly:\n1. Reference their notable works\n2. Mention famous characters they created\n3. Describe distinctive elements of their style\n4. Use quotes or famous scenes\n5. Reference biographical details\n6. Mention close collaborators\n\nFor each prompt:\n- Vary the length (short, medium, detailed)\n- Use different tones (casual, academic, enthusiastic)\n- Include different request types (explanations, comparisons, creative tasks)\n\nUser questions to base your prompts on:\n" + "\n -".join(questions)}
        ],
        response_format={
            "type": "json_object",
            "schema": {
                "type": "object",
                "properties": {
                    "prompts": {
                        "type": "array",
                        "description": "Array of generated prompts reflecting different styles and approaches",
                        "items": {
                            "type": "string",
                            "description": "A natural-sounding, contextually appropriate prompt",
                            "maxLength": 300,
                        },
                        "maxItems": 3
                    },
                },
                "required": ["prompts"]
            }
        },
        temperature=temperature_p,
        max_tokens=2048,
        seed=random.randint(0, 1000000)
    )

    prompts = json.loads(out["choices"][0]["message"]["content"])["prompts"]
    return questions, prompts

# Main execution loop with more topics for variety
directors = [
    # "Quentin Tarantino",
    # "Christopher Nolan",
    "Steven Spielberg",
    "Tim Burton",
    # "Wes Ande/rson"
]
import time
from tqdm import tqdm
start_time = time.time()
for i in tqdm(range(100000//3)):
    director = directors[0]
    
    questions, prompts = generate_prompts(llm, director=director, temperature_q=2.5, temperature_p=0.4)
    if questions is None:
        continue
   

    with open(f"data/{director}_prompts.txt", "a") as f:
        f.write("\n".join(prompts))
        f.write("\n")
    with open(f"data/{director}_questions.txt", "a") as f:
        f.write("\n".join(questions))
        f.write("\n")

end_time = time.time()
ic(f"Time taken to generate prompts: {end_time - start_time:.2f} seconds")