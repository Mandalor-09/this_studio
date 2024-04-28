import streamlit as st
import pandas as pd
from helper import product_name_list,get_product_details,input_prompt,get_response_from_model,filtering_response,loading_model_and_tokenizer

df = pd.read_csv('dataset/cleaned_data.csv')
df = df.iloc[:,1:-2]

tokenizer,model = loading_model_and_tokenizer('models/model','models/adapter')

product_name = product_name_list(df)

def reset_session(selected_product):
    st.session_state.messages = []
    st.session_state.messages.append({
        'role': 'assistant',
        'content': f'Hello  How may I help You with Product {selected_product}'
    })
    
def main():
    # Get product selection from sidebar
    selected_product = st.sidebar.selectbox('Select a Product', product_name)

    # Initialize messages list if it doesn't exist
    if "messages" not in st.session_state:
        reset_session(selected_product)

    # Update assistant message based on selection (if any)
    if selected_product is not None:
        st.session_state.messages[0]['content'] = f'Hello  How may I help You with Product {selected_product}'

    # Display subheader based on selection
    st.subheader(f'You have Query Related to {selected_product}')

    # Get product details from a function (assuming it exists)
    product_details = get_product_details(df, selected_product)

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Generate and display assistant response based
        input_text = input_prompt(product_details, prompt)
        response = get_response_from_model(model,tokenizer,input_text)
        filter_res = filtering_response(response)
        st.session_state.messages.append({
        'role': 'assistant',
        'content': f'{filter_res}'
        })

        #Reset Session as soon as new product is selected
        reset_session(selected_product)



if __name__ == '__main__':
    main()