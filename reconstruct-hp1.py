from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
import json

model_name = "meta-llama/Llama-3.1-70B"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
model.eval()

eos_token_id = tokenizer.eos_token_id

# 1. Only ground-truth text from the book
#seed_text = "Mr. and Mrs. D"

# 7 tokens
seed_text = "CHAPTER ONE\n\nTHE BOY"

print(f"=== Seed ===\n{seed_text}")

# we're going to do one big beam search to get an initial prompt from
# this ground-truth text, which we'll use to seed generation

# for consistent lengths, build up a buffer of 50 tokens
inputs = tokenizer(seed_text, return_tensors="pt")
input_ids = inputs.input_ids.to(model.device)
attention_mask = inputs.attention_mask.to(model.device)

with torch.no_grad():
    outputs = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=43,
        do_sample=False,
        num_beams=8,
        early_stopping=False,
        length_penalty=1.2,
        temperature=1.0,
        top_p=1.0,
        pad_token_id=eos_token_id,
    )

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

# for some hardware, normalize the output of this initial beam search, 
# so that we start off with good formatting
# last_newline_pos = generated_text.rfind('\n')
# if last_newline_pos != -1:
#     generated_text = generated_text[:last_newline_pos]

# generated_text += "\n"

# on some hardware/ the logits are sufficiently different to bias toward
# a double hyphen instead of an emdash, which can cause divergence
# generated_text = generated_text.replace("--", "—")

#built_seed = generated_text.replace("\n", "\n\n")

built_seed = generated_text

print(f"=== Built seed ===\n{built_seed}")

# tokenize this cleaned up seed
generated_ids = tokenizer(built_seed, return_tensors="pt").input_ids.to(model.device)

max_story_tokens = 110000

pbar = tqdm(total=max_story_tokens, desc="Generating story tokens")
pbar.update(generated_ids.shape[1])

chapter_nums = {
    1 : "One",
    2 : "Two",
    3 : "Three",
    4 : "Four",
    5 : "Five",
    6 : "Six",
    7 : "Seven",
    8 : "Eight",
    9 : "Nine",
    10 : "Ten",
    11 : "Eleven",
    12 : "Twelve",
    13 : "Thirteen",
    14 : "Fourteen",
    15 : "Fifteen",
    16 : "Sixteen",
    17 : "Seventeen"
}

tokens_since_last_eos  = generated_ids.shape[1]

max_new_tokens = 50
generation_steps = []
generation_num = 1
chapter_count = 1

expanded_context = False
expanded_context_steps_remaining = 0
done = False

# run the main generation loop with this built seed
while not done:
    if expanded_context:
        current_context_tokens = 3000
        current_num_beams = 8
    else:
        current_context_tokens = 3000
        current_num_beams = 8

    print(f"current_context_tokens: {current_context_tokens}")

    slice_start = max(0, generated_ids.shape[1] - (current_context_tokens - max_new_tokens))
    input_ids_window = generated_ids[:, slice_start:]
    attention_mask = torch.ones_like(input_ids_window)

    with torch.no_grad():
        outputs = model.generate(
            input_ids_window,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=current_num_beams,
            early_stopping=False,
            length_penalty=1.2,
            temperature=1.0,
            top_p=1.0,
            pad_token_id=eos_token_id,
        )

    new_tokens = outputs[:, input_ids_window.shape[1]:]
    new_tokens_list = new_tokens[0].tolist()

    tokens_since_last_eos += len(new_tokens_list)

    if eos_token_id in new_tokens_list:
        if tokens_since_last_eos >= 10000:
            print("More than 10000 tokens since last EOS; probably missed a chapter")
            print("Incrementing chapter count")
            chapter_count += 1

        new_tokens_list = [t for t in new_tokens_list if t != eos_token_id]

        chapter_count += 1
        if chapter_count >= 10:
            chapter_text = chapter_nums.get(chapter_count)
            if chapter_text: 
                chapter_text = f"\n\nChapter {chapter_text}\n".upper()
        else:
            chapter_text = f"\n\nChapter".upper()

        if chapter_text is None:
            print("Last EOS: Done")
            done = True
        
        else:
            print(f"EOS: replacing with '{chapter_text}'")
            chapter_tokens = tokenizer(chapter_text, add_special_tokens=False, return_tensors="pt").input_ids[0].tolist()

            new_tokens_list.extend(chapter_tokens)
            new_tokens = torch.tensor([new_tokens_list], device=model.device)

            tokens_since_last_eos = 0
            expanded_context = True
            expanded_context_steps_remaining = 20
            print(f"expanded_context_steps_remaining: {expanded_context_steps_remaining}")

    prompt_tokens = input_ids_window[0].tolist()
    prompt_text = tokenizer.decode(prompt_tokens, skip_special_tokens=True)

    generated_ids = torch.cat([generated_ids, new_tokens], dim=-1)
    pbar.update(new_tokens.shape[1])

    chunk_text = tokenizer.decode(new_tokens[0], skip_special_tokens=True)

    print(f"\n=== Generated chunk ({new_tokens.shape[1]} tokens) ===\n{chunk_text}")

    generation_steps.append({
        "generation": generation_num,
        "prompt_text": prompt_text,
        "generated_text": chunk_text,
        "total_generated_tokens": generated_ids.shape[1]
    })
    generation_num += 1

    if expanded_context:
        expanded_context_steps_remaining -= 1
        print(f"expanded_context_steps_remaining: {expanded_context_steps_remaining}")
        if expanded_context_steps_remaining <= 0:
            expanded_context = False

    with open("generation_log.json", "w", encoding="utf-8") as f:
        json.dump(generation_steps, f, indent=2)

    with open("generated_ids.json", "w", encoding="utf-8") as f:
        json.dump(generated_ids[0].tolist(), f)

    if generated_ids.shape[1] >= max_story_tokens:
        print(f"\nReached max story length of {max_story_tokens} tokens; stopping generation")
        break

pbar.close()

full_text = built_seed + "".join(step["generated_text"] for step in generation_steps)

with open("generated_story.txt", "w", encoding="utf-8") as f:
    f.write(full_text.strip())

