from transformers import AutoTokenizer, AutoModelForCausalLM

global_model = None
global_tokenizer = None

def product_name_list(df):
    return list(df['Product_Name'].unique())

def get_product_details(df,product_name):
    data = df[df['Product_Name'] == product_name]
    selected_product = data.iloc[0].to_dict()
    selected_product = str(selected_product)
    selected_product = selected_product.replace('{','[').replace('}',']')
    selected_product = selected_product.replace('\n','')
    return selected_product.strip()

def loading_model_and_tokenizer(model_dir,adapter_dir):
    if global_tokenizer is None or global_model is None:
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForCausalLM.from_pretrained(model_dir)
        model.load_adapter(adapter_dir)
        global global_model
        global global_tokenizer
        global_model = model
        global_tokenizer = tokenizer
    return global_tokenizer, global_model

def input_prompt(product_details,question):
    alpaca_prompt = f"""
### Instruction : Help user to solve is query
### Context : {product_details}
### Input : {question}
### Response : ''
"""
    return alpaca_prompt

def get_response_from_model(tokenizer,model,alpaca_prompt):
    inputs = tokenizer(alpaca_prompt, return_tensors="pt").input_ids
    outputs = model.generate(inputs, max_length=350)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def filtering_response(response):
    response_start = "### Response :"
    # Find the index where the response starts
    start_index = response.find(response_start)

    if start_index != -1:
        # Extract everything after the response start index
        extracted_response = response[start_index + len(response_start):].strip()
        return extracted_response
    else:
        return "Something Went Wrong"