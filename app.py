import streamlit as st

PAGE_CONFIG = {"page_title":"StColab.io","page_icon":":smiley:","layout":"centered"}
st.set_page_config(**PAGE_CONFIG)


def main():
    st.title("Awesome Streamlit for ML")
    st.subheader("How to run streamlit from colab")


    menu = ["Home","About"]
    choice = st.sidebar.selectbox('Menu',menu)
    if choice == 'Home':
        st.subheader("Streamlit From Colab")



if __name__ == '__main__':
    main()