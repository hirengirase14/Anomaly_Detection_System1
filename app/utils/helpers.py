import streamlit as st
import matplotlib.pyplot as plt

def identify_roles(files):
    roles = {"train":None,"test":None,"label":None,"single":None}
    if len(files)==1:
        roles["single"]=files[0]
        return roles
    for f in files:
        n=f.name.lower()
        if "train" in n: roles["train"]=f
        elif "label" in n: roles["label"]=f
        elif "test" in n: roles["test"]=f
    if not roles["train"] and not roles["test"]:
        roles["single"]=files[0]
    return roles

def sfig(fig, axes=None):
    return fig

def glass_chart(fig):
    st.pyplot(fig)
    plt.close(fig)