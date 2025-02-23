import torch
from transformers import AutoModelForCausalLM, AutoProcessor

# NOTE: transformers==4.46.3 is recommended for this script
"""
Transformers 라이브러리에서 **AutoModelForCausalLM** 클래스를 별도로 제공하는 이유는 모델의 **사용 목적과 아키텍처**에 따라 적절한 헤드(head)와 기능을 자동으로 구성해주기 위해서입니다.

### 1. 기본 AutoModel과의 차이점
- **AutoModel**: 기본적인 트랜스포머 인코더나 디코더의 레이어만 로드하며, 특정 작업에 최적화된 헤드가 붙어있지 않습니다. 단순히 모델의 “본체”만 불러오는 역할을 하죠.
- **AutoModelForCausalLM**: **인과(autoregressive) 언어모델**에 특화된 클래스로, 텍스트 생성과 같이 시퀀스의 다음 토큰을 예측하는 작업에 적합한 **언어 모델링(head)**을 포함합니다. 이 헤드는 자동으로 생성 함수(`generate`)와 같은 기능들을 지원하도록 설계되어 있습니다.

### 2. 작업별 맞춤형 헤드 제공
딥러닝 모델은 동일한 트랜스포머 본체를 사용하더라도, 이후에 붙는 “헤드”에 따라 용도가 달라집니다. 예를 들어:
- **AutoModelForSequenceClassification**: 분류 작업에 적합한 헤드.
- **AutoModelForSeq2SeqLM**: 인코더-디코더 구조를 사용하는 번역, 요약 등 작업에 적합한 헤드.
- **AutoModelForCausalLM**: 텍스트 생성, 즉 다음 토큰 예측에 최적화된 헤드.

각각의 AutoModel 클래스는 해당 작업에 맞는 사전학습된 헤드를 포함하여 모델을 초기화하므로, 사용자 입장에서는 별도의 추가 구성 없이 바로 해당 작업을 수행할 수 있게 됩니다.

### 3. 사용자 편의성과 안전성
- **API 일관성**: 모델을 불러올 때, 어떤 작업을 위해 모델을 사용할 것인지 명시함으로써, 잘못된 헤드를 붙이거나 사용 실수를 줄일 수 있습니다.
- **생성 함수 지원**: AutoModelForCausalLM은 자동으로 텍스트 생성 메서드(`generate`)를 지원합니다. 일반 AutoModel은 이러한 작업을 위해 별도의 추가 작업이 필요할 수 있습니다.
- **내부 최적화**: 인과 언어 모델링에 필요한 다양한 최적화(예: 어텐션 스킴, 토큰 마스킹 처리 등)를 내부적으로 처리하기 때문에, 사용자는 복잡한 세부 구현에 대해 고민할 필요 없이 모델을 바로 활용할 수 있습니다.

### 결론
즉, Transformers 라이브러리에서 **AutoModelForCausalLM**을 별도로 분리해 관리하는 이유는 **모델의 사용 목적에 따른 최적화와 사용자 편의성을 극대화하기 위해서**입니다. 모델 경로만 지정하는 것과는 달리, 이 클래스는 자동으로 인과 언어 모델에 필요한 구성 요소(헤드, 생성 함수 등)를 포함하여 불러오기 때문에, 올바른 아키텍처와 기능이 함께 로드되도록 보장합니다. 이는 복잡한 딥러닝 파이프라인에서 실수를 줄이고, 작업에 최적화된 모델을 손쉽게 활용할 수 있도록 돕는 중요한 설계 철학입니다.
"""
model_path = "DAMO-NLP-SG/VideoLLaMA3-7B"
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    device_map={"": "cuda:0"},
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)


@torch.inference_mode()
def infer(conversation):
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
    output_ids = model.generate(**inputs, max_new_tokens=1024)
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
