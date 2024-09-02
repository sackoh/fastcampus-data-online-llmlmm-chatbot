import os
import requests

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from openai import OpenAI


chroma_client = chromadb.PersistentClient(
    path='./db',
    settings=Settings(),
)

embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="intfloat/multilingual-e5-large"
)


class Agent:
    def __init__(self,):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.messages = []
        self.end = False

    def chat_completion(self, model='gpt-4o-mini', max_tokens=256, temperature=1.):
        response = self.client.chat.completions.create(
            messages=self.messages, 
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            )
        assistant_message = response.choices[0].message.content
        return assistant_message


class WeatherAgent(Agent):
    def __init__(self, ):
        super().__init__()
        self.weather_api_key = os.getenv("WEATHER_API_KEY")
        self.messages = [{"role": "system", "content": self.initialize_system_message()}]
    
    @staticmethod
    def initialize_system_message():
        return """당신은 패스트투어 여행사의 날씨정보를 제공하는 AI Assistant입니다. 사용자의 Message History와 Weather Info를 참고하여 아래에 주어진 Task Description에 따라 단계적으로 업무를 수행합니다.

## Task Description:
1. Step 1: 사용자 메시지에 필수 엔티티 "도시명(City Name)"이 없으면 요청 메시지를 작성합니다. 응답 메시지의 마지막에 <|require-city|> 토큰을 붙입니다.
    <example>정확하게 도시의 이름을 입력해주세요. (ex. 서울, 베이징)</example>
    <example>어느 도시의 날씨 정보가 궁금하신가요?</example>

2. Step 2: 사용자 메시지에 "도시명(City Name)"이 있으면 도시명만 영어로 결과를 작성합니다. 응답 메시지의 마지막에 <|require-weather|> 토큰을 붙입니다.
    <example>서울입니다. -> Seoul</example>
    <example>토론토 -> Toronto</example>
    <example>홍콩 -> Hongkong</example>

3. Step 3: Weather Info에 날씨 정보가 있다면 이를 참조하여 사용자의 날씨정보 문의에 대한 응답 메시지를 한글로 작성합니다. 응답 메시지의 마지막에 <|end-weather|> 토큰을 붙입니다.
    <example>토론토의 현재 날씨는 맑고 ☀️  기온은 25.1°C 입니다.</example>
    <example>도쿄는 지금 구름이 끼었고 🌥️  기온은 31.8°C 입니다.</example>

4. Step 4: <|end-weather|> 상태이고, 추가적인 사용자의 메시지가 있다면 해당 메시지에 대한 올바른 답변을 작성합니다.

5. Step 5: 별다른 추가 문의사항이 있는지 물어보고 없다고 한다면, 감사합니다라는 문구로 대화를 마무리합니다.

## Weather Info:
"""

    def process_user_message(self, user_message):
        self.messages.append({"role": "user", "content": user_message})
        assistant_message = self.chat_completion()
        
        if '<|require-city|>' in assistant_message:
            assistant_message = assistant_message.replace('<|require-city|>', '').strip()
            self.messages.append({"role": "assistant", "content": assistant_message})

        elif '<|require-weather|>' in assistant_message:
            assistant_message = assistant_message.replace('<|require-weather|>', '').strip()
            weather_info = self._get_weather(assistant_message)
            if weather_info:
                self._update_weather_info(weather_info)
                self.messages.append({"role": "assistant", "content": assistant_message})
                assistant_message = self.chat_completion(temperature=0)
                if '<|end-weather|>' in assistant_message:
                    self.messages.append({"role": "assistant", "content": assistant_message})
                    pass
                else:
                    assistant_message = "⚠️ 날씨 API 통신 이슈가 있었습니다. 동일한 내용을 다시 말씀해주시겠어요?"
            else:
                assistant_message = "⚠️ 입력하신 정보의 날씨 조회 서비스는 지원되지 않습니다."
                self.messages.append({"role": "assistant", "content": assistant_message})

        else:
            self.messages.append({"role": "assistant", "content": assistant_message})
        
        # make sure remove token
        return assistant_message.replace('<|end-weather|>', '').replace('<|require-weather|>', '')
            
    def _get_weather(self, city):
        api_url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={self.weather_api_key}&units=metric&lang=kr"
        response = requests.get(api_url)
        if response.status_code == 200:
            data = response.json()
            description = data.get('weather', [{}])[0].get('description', '')
            temp = data.get('main', {}).get('temp', '')
            return {'description': description, 'temp': temp}
        return {}
    
    def _update_weather_info(self, weather_info):
        description = weather_info["description"]
        temperature = weather_info["temp"]
        updated_system = self.messages[0]["content"] + f'- description: {description}\n- temperature: {temperature}'
        self.messages[0] = {"role": "system", "content": updated_system}


class FaqAgent(Agent):
    def __init__(self, chroma_client, embedding_function):
        super().__init__()
        self.collection = chroma_client.get_collection(name="faq", embedding_function=embedding_function)
    
    @staticmethod
    def initialize_system_message():
        return """You are the helpful AI assistant to answer the given User message. You should reference the relevant FAQs below."""

    def search(self, query):
        results = self.collection.query(
            query_texts=[query],
            n_results=5,
        )
        results = results['documents'][0]
        documents = "\n\n<FAQ>\n"
        documents += "\n\n".join([f"{idx+1}. relevant document: {doc}" for idx,  doc in enumerate(results)])
        self.messages.append({"role": "system", "content": self.initialize_system_message() + documents})

    def process_user_message(self, user_message):
        if len(self.messages) == 0:
            self.search(user_message)

        self.messages.append({"role": "user", "content": user_message})
        assistant_message = self.chat_completion()
        self.messages.append({"role": "assistant", "content": assistant_message})

        return assistant_message


class CommunityAgent(Agent):
    def __init__(self, chroma_client, embedding_function):
        super().__init__()
        self.collection = chroma_client.get_collection(name="community", embedding_function=embedding_function)
        
    @staticmethod
    def initialize_system_message():
        return """You are the helpful AI assistant to answer the given User message. You should reference the relevant documents below. If the documents are not relevant to the User message, do not reference the documents.

<Retrieved Documents>
{}

Consideration: If the document is not relevant to the User message, do not reference the documents.
"""

    def search(self, query):
        results = self.collection.query(
            query_texts=[query],
            n_results=3,
        )
        results = results['documents'][0]
        documents = "\n\n".join([f"{idx+1}. relevant document: {doc}" for idx,  doc in enumerate(results)])
        self.messages.append({"role": "system", "content": self.initialize_system_message().format(documents)})

    def process_user_message(self, user_message):
        if len(self.messages) == 0:
            self.search(user_message)

        self.messages.append({"role": "user", "content": user_message})
        assistant_message = self.chat_completion()
        self.messages.append({"role": "assistant", "content": assistant_message})

        return assistant_message


class AccmAgent(Agent):
    def __init__(self, chroma_client, embedding_function):
        super().__init__()
        self.collection = chroma_client.get_collection(name="accommodation", embedding_function=embedding_function)

    @staticmethod
    def initialize_system_message():
        return """You are the helpful AI assistant to answer the given User message. You should reference the relevant documents below. The documents are the information about the accommodations. And you should summarize and provide the essential information to the customer about the retrieved accommodations. Please follow the below Accommodation Template.

<Accommodation Template>
1. '''Input 숙소 이름 in the accommodatio document'''
  - 위치: '''Input 위치정보 in the accommodation document'''
  - 숙소정보: '''Input summarized text of 숙소개요 in the accommodation document
  - 추천 이유: '''Input text to recommend the accommodation'''
  - 기타 제공 서비스: '''Input few other 숙소 제공 서비스 to get attraction in the accommodation document'''
2. ...


<Retrieved Documnets>
{}

Consideration: If 위치 of the accommodation is not relevant to the User message, do not reference the documents.
"""

    def search(self, query):
        results = self.collection.query(
            query_texts=[query],
            n_results=3,
        )
        results = results['documents'][0]
        documents = "\n\n".join([f"{idx+1}. relevant document: {doc}" for idx,  doc in enumerate(results)])
        self.messages.append({"role": "system", "content": self.initialize_system_message().format(documents)})

    def process_user_message(self, user_message):
        if len(self.messages) == 0:
            self.search(user_message)

        self.messages.append({"role": "user", "content": user_message})
        assistant_message = self.chat_completion(max_tokens=1024)
        self.messages.append({"role": "assistant", "content": assistant_message})

        return assistant_message 
