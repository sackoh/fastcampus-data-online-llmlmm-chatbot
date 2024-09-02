import os

from dotenv import load_dotenv
from openai import OpenAI


class IntentClassifier:
    def __init__(self):
        load_dotenv()
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        self.INTENT_LIST = [
            "숙소 정보탐색", 
            "여행지 정보탐색", 
            "서비스 이용문의", 
            "날씨 정보조회", 
            "가드레일"
        ]

        self.system_prompt = """As an AI-driven assistant, categorize the provided message into the intents from the provided categories:

Intent Categories:

1. {}: 국내외에 위치한 지역의 숙소를 찾고 싶거나 숙소의 정보를 알고 싶어하는 메시지
    <example>오사카에 료칸 형태의 숙소 좀 알려줄래</example>
        
2. {}: 특정 지역과 관련된 여행 관련 정보와 팁 등에 대한 내용을 알고 싶어하는 메시지
    <example>오사카 놀러가면 가봐야 할 곳 좀 추천해줘</example>

3. {}: 패스트투어 서비스를 이용하는데 있어 궁금한 사항에 대한 문의 메시지
    <example>현금영수증 발급받고 싶어요.</example>
    <example>제 비행 일정상 내일 오전 6시경에 도착 예정인데요, 조금 이른 시간에 체크인이 가능할까요? 혹시 추가 요금이 발생하는지도 궁금합니다.</example>

4. {}: 특정 지역의 기온과 온도 등의 날씨 정보에 대한 문의 메시지
    <example>오사카 날씨 어때?</example>

5. {}: 기타 관련 없는 메시지

Output Requirement: Provide the detected intent name only.
""".format(*self.INTENT_LIST)

    def classify_intent(self, message):
        response = self.client.chat.completions.create(
            model='gpt-4o-mini',
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": message},
            ],
        )

        intent = response.choices[0].message.content.strip()
        if intent not in self.INTENT_LIST:
            raise ValueError(f"Unexpected intent: {intent}")
        return intent
