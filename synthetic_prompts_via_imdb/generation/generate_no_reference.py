from llama_cpp import Llama
from icecream import ic

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
    #flash_attn=True,
    echo=False,
)

print("Model loaded successfully.")
def prepare_reviews():
    from datasets import load_dataset, concatenate_datasets

    ds = load_dataset("stanfordnlp/imdb")
    ds = concatenate_datasets([ds["train"], ds["unsupervised"], ds["test"]])
    
    return ds["text"]

def count_json_tokens(llm, response):
    """Count tokens for each entry in the JSON response"""
    content = response["choices"][0]["message"]["content"]
    
    try:
        import json
        parsed_data = json.loads(content)
        token_counts = {}
        
        # Count tokens for the thinking section if it exists
        if "thinking" in parsed_data:
            thinking_text = bytes(parsed_data["thinking"], "utf-8")
            token_counts["thinking"] = len(llm.tokenize(thinking_text))
        
        # Count tokens for each individual example
        if "prompts" in parsed_data:
            prompts = parsed_data["prompts"]
            token_counts["prompts"] = {}
            token_counts["prompts"]["total"] = len(llm.tokenize(bytes("\n".join([ex for ex in prompts]), "utf-8")))
            
            # Count tokens for each individual example
            for i, prompt in enumerate(prompts):
                token_counts["prompts"][f"prompt_{i+1}"] = len(llm.tokenize(bytes(prompt, "utf-8")))
        
        # Count tokens for any other top-level fields
        for key in parsed_data:
            if key not in ["thinking", "prompts"]:
                token_counts[key] = len(llm.tokenize(bytes(parsed_data[key], "utf-8")))
        
        # Add total token count
        token_counts["total_json_tokens"] = len(llm.tokenize(bytes(content, "utf-8")))
        token_counts["total_input_tokens"] = out["usage"]["prompt_tokens"]
        token_counts["total_output_tokens"] = out["usage"]["completion_tokens"]
        
        return parsed_data, token_counts
        
    except json.JSONDecodeError:
        return {"error": "Response is not valid JSON", "raw_tokens": len(llm.tokenize(content))}


examples = prepare_reviews()[10:15]
reference_prefix = "> [reference data] - 5 random film reviews\n"
reference_data = reference_prefix + "\n".join(examples)

out = llm.create_chat_completion(
    messages=[
        {"role": "system", "content": "You are a synthetic data generator. Your task is to generate synthetic data, insuring plausible diversity of the generated data."},
        {"role": "user", "content": "Generate 3 prompts for an LLM. Prompts should be about director Quentin Tarantino. Prompts should not mention Tarantino directly but should imply it or mention names of his movies, retell plot/quotes of his movies, mention facts of his life/co-stars that he worked with. DO NOT QUENTIN TARANTINO NAME IN THE PROMPTS. \n How to do it: \n 1. Think about what should be the question/task behind the prompt that you will generate. But it still should be about Quentin Tarantino. \n 2. Generate 3 prompts of different styles, that are centered around the question/task for LLM you came up with in step 1."}
    ],
    response_format={
        "type": "json_object",
        "schema": {
            "type": "object",
            "properties": {
                "questions": {
                    "type": "array",
                    "description": "Array of questions that you came up with in step 1",
                    "items": {
                        "type": "string",
                        "description": "A question",
                        "maxLength": 300,
                    },
                },
                "prompts": {
                    "type": "array",
                    "description": "Array of generated prompts - this is required",
                    "items": {
                        "type": "string",
                        "description": "A generated prompt",
                        "maxLength": 300,
                    },
                    "maxItems": 3
                },
            },
            "required": ["questions", "prompts"]
        }
    },
    temperature=0.7,
    max_tokens=2048
)

ic(out)
#
# token_counts = count_json_tokens(llmw, out)
