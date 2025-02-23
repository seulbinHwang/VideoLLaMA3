import torch
from transformers import AutoModelForCausalLM, AutoProcessor

# NOTE: transformers==4.46.3 is recommended for this script
"""
Transformers 라이브러리에서 **AutoModelForCausalLM** 클래스를 별도로 제공하는 이유는 
    모델의 **사용 목적과 아키텍처**에 따라 "적절한 헤드(head)와 기능"을 자동으로 구성해주기 위해서

### 1. 기본 AutoModel과의 차이점
- **AutoModel**: 
    - 기본적인 트랜스포머 인코더나 디코더의 레이어만 로드하며, 특정 작업에 최적화된 헤드가 붙어있지 않습니다. 
    - 단순히 모델의 “본체”만 불러오는 역할을 하죠.
- **AutoModelForCausalLM**: 
    - **인과(autoregressive) 언어모델**에 특화된 클래스로, 
    - 텍스트 생성과 같이 시퀀스의 다음 토큰을 예측하는 작업에 적합한 **언어 모델링(head)**을 포함
    - 이 헤드는 자동으로 생성 함수(`generate`)와 같은 기능들을 지원하도록 설계되어 있습니다.

### 2. 작업별 맞춤형 헤드 제공
딥러닝 모델은 동일한 트랜스포머 본체를 사용하더라도, 이후에 붙는 “헤드”에 따라 용도가 달라집니다. 
- 예를 들어:
    - **AutoModelForSequenceClassification**: 분류 작업에 적합한 헤드.
    - **AutoModelForSeq2SeqLM**: 인코더-디코더 구조를 사용하는 번역, 요약 등 작업에 적합한 헤드.
    - **AutoModelForCausalLM**: 텍스트 생성, 즉 다음 토큰 예측에 최적화된 헤드.

- 각각의 AutoModel 클래스는 해당 작업에 맞는 사전학습된 헤드를 포함하여 모델을 초기화하므로, 
    - 사용자 입장에서는 별도의 추가 구성 없이 바로 해당 작업을 수행할 수 있게 됩니다.

### 3. 사용자 편의성과 안전성
- **API 일관성**: 
- **생성 함수 지원**: 
    - AutoModelForCausalLM은 자동으로 텍스트 생성 메서드(`generate`)를 지원
    - 일반 AutoModel은 이러한 작업을 위해 별도의 추가 작업이 필요할 수 있습니다.
- **내부 최적화**: 
    - 인과 언어 모델링에 필요한 다양한 최적화(예: 어텐션 스킴, 토큰 마스킹 처리 등)를 내부적으로 처리하기 때문에, 
    - 사용자는 복잡한 세부 구현에 대해 고민할 필요 없이 모델을 바로 활용할 수 있습니다.
"""
model_path = "DAMO-NLP-SG/VideoLLaMA3-7B"
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    device_map={"": "cuda:0"},
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)
"""
이 processor는 텍스트 외에도 이미지와 영상 데이터를 모델이 이해할 수 있는 토큰 형태로 인코딩하는 역할
"""
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)


@torch.inference_mode()
def infer(conversation):
    """
    args:
        conversation: 멀티모달 입력(텍스트, 이미지, 영상 등)이 포함된 대화 데이터를 전달
    시스템 프롬프트(system prompt):
        - 모델의 전체 동작이나 역할, 제한 사항 등을 정의
        - 예를 들어 "You are a helpful assistant." 같은 지시문은
          - 모델이 대화 중 어떠한 태도로 응답해야 하는지를 명확히 합니다.
        - 이 옵션이 활성화되면, processor는 대화의 시작 부분에 사전에 정의된 시스템 프롬프트를 자동으로 삽입
        - 컨텍스트 명료화:
          - 사용자 입력과 혼동되지 않도록 시스템 메시지를 명확하게 구분
    생성 프롬프트(generation prompt):
      - 실제 응답을 생성할 때 모델이 참고해야 할 추가 지침이나 서식을 제공
      - 이는 모델이 응답 시작 시점을 명확히 인식하도록 도와줌
      - 이 옵션은 모델이 응답을 생성하기 전, 적절한 지침(예: "Answer:" 또는 특별한 토큰)을 자동으로 추가합니다.
    return_tensors="pt":
      - 전처리 결과를 PyTorch 텐서 형태로 반환

    """
    inputs = processor(conversation=conversation,
                       add_system_prompt=True,
                       add_generation_prompt=True,
                       return_tensors="pt")
    inputs = {
        k: v.cuda() if isinstance(v, torch.Tensor) else v
        for k, v in inputs.items()
    }
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
    """
    전처리된 입력을 바탕으로 최대 1024 토큰 길이의 응답을 생성
    """
    output_ids = model.generate(**inputs, max_new_tokens=1024)
    """
    processor.batch_decode: 생성된 토큰 시퀀스를 사람이 읽을 수 있는 문자열로 변환
    kip_special_tokens=True 옵션은 특수 토큰들을 제거
    """
    response = processor.batch_decode(output_ids,
                                      skip_special_tokens=True)[0].strip()
    return response


# Video conversation
conversation = [
    {
        "role": "system",
        "content": "You are a helpful assistant."
    },
    {
        "role":
            "user",
        "content": [
            {
                "type": "video",
                "video": {
                    "video_path": "./assets/cat_and_chicken.mp4",
                    "fps": 1,
                    "max_frames": 180
                }
            },
            {
                "type":
                    "text",
                "text":
                    "What is the cat doing? Please describe the scene, the obejcts and the actions in detail."
            },
        ]
    },
]
print(infer(conversation))

# Image conversation
conversation = [{
    "role":
        "user",
    "content": [
        {
            "type": "image",
            "image": {
                "image_path": "./assets/sora.png"
            }
        },
        {
            "type": "text",
            "text": "Please describe the model?"
        },
    ]
}]
print(infer(conversation))

# Mixed conversation
conversation = [{
    "role":
        "user",
    "content": [
        {
            "type": "video",
            "video": {
                "video_path": "./assets/cat_and_chicken.mp4",
                "fps": 1,
                "max_frames": 180
            }
        },
        {
            "type":
                "text",
            "text":
                "What is the relationship between the video and the following image?"
        },
        {
            "type": "image",
            "image": {
                "image_path": "./assets/sora.png"
            }
        },
    ]
}]
print(infer(conversation))

# Plain text conversation
conversation = [{
    "role": "user",
    "content": "What is the color of bananas?",
}]
print(infer(conversation))
