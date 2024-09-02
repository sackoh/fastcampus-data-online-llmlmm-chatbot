import streamlit as st

from nlu import IntentClassifier
from nlg import WeatherAgent, AccmAgent, CommunityAgent, FaqAgent, chroma_client, embedding_function


st.title("🚀 패스트투어 Chatbot")

# initialize sessions states
if 'intent_cls' not in st.session_state:
    st.session_state.intent_cls = IntentClassifier()
if 'chroma_client' not in st.session_state:
    st.session_state.chroma_client = chroma_client
    st.session_state.embedding_function = embedding_function

if 'predicted_intent' not in st.session_state:
    st.session_state.predicted_intent = None
if 'assistant' not in st.session_state:
    st.session_state.assistant = None


if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "안녕하세요, 패스트투어 챗봇입니다. 무엇을 도와드릴까요?"}]

for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if query := st.chat_input(key='query'):
    st.session_state['messages'].append({"role": "user", "content": st.session_state.query})
    with st.chat_message("user"):
        st.markdown(st.session_state.query)
    
    # classify an intent of the first query
    if len(st.session_state["messages"]) == 2 and st.session_state.predicted_intent is None:
        st.session_state.predicted_intent = st.session_state.intent_cls.classify_intent(st.session_state.query)    
    
    if st.session_state.predicted_intent == "날씨 정보조회":
        if not st.session_state.assistant:
            st.session_state.assistant = WeatherAgent()
            print('Created: weather assistant')
        message = st.session_state.assistant.process_user_message(st.session_state.query)

    elif st.session_state.predicted_intent == "서비스 이용문의":
        if not st.session_state.assistant:
            st.session_state.assistant = FaqAgent(st.session_state.chroma_client, st.session_state.embedding_function)
            print('Created: faq assistant')
        message = st.session_state.assistant.process_user_message(st.session_state.query)
        
    elif st.session_state.predicted_intent == "여행지 정보탐색":
        if not st.session_state.assistant:
            st.session_state.assistant = CommunityAgent(st.session_state.chroma_client, st.session_state.embedding_function)
            print('Created: community assistant')
        message = st.session_state.assistant.process_user_message(st.session_state.query)

    elif st.session_state.predicted_intent == "숙소 정보탐색":
        if not st.session_state.assistant:
            st.session_state.assistant = AccmAgent(st.session_state.chroma_client, st.session_state.embedding_function)
            print('Created: accm assistant')
        message = st.session_state.assistant.process_user_message(st.session_state.query)

    else:
        message = '죄송합니다, 저는 패스트투어 고객지원 챗봇으로 `날씨 정보조회`, `서비스 이용문의`, `여행지 및 숙소 정보탐색`을 도와드리고 있습니다. 세션을 초기하고 다시 말씀해주세요.'

    st.session_state['messages'].append({"role": "assistant", "content": message})
    with st.chat_message("assistant"):
        st.markdown(message)


if st.button(label='Reset Chat'):
    for key in ['predicted_intent', 'assistant', 'messages', 'query']:
        st.session_state.pop(key)