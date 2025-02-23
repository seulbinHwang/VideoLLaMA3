import sys
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoProcessor, AutoImageProcessor

from .configuration_videollama3 import Videollama3Qwen2Config
from .configuration_videollama3_encoder import Videollama3VisionEncoderConfig
from .modeling_videollama3 import Videollama3Qwen2Model, Videollama3Qwen2ForCausalLM
from .modeling_videollama3_encoder import Videollama3VisionEncoderModel
from .processing_videollama3 import Videollama3Qwen2Processor
from .image_processing_videollama3 import Videollama3ImageProcessor

AutoConfig.register("videollama3_vision_encoder",
                    Videollama3VisionEncoderConfig)
AutoModel.register(Videollama3VisionEncoderConfig,
                   Videollama3VisionEncoderModel)
AutoImageProcessor.register(Videollama3VisionEncoderConfig,
                            Videollama3ImageProcessor)

AutoConfig.register("videollama3_qwen2", Videollama3Qwen2Config)
AutoModel.register(Videollama3Qwen2Config, Videollama3Qwen2Model)
AutoModelForCausalLM.register(Videollama3Qwen2Config,
                              Videollama3Qwen2ForCausalLM)
AutoProcessor.register(Videollama3Qwen2Config, Videollama3Qwen2Processor)

Videollama3VisionEncoderConfig.register_for_auto_class()
Videollama3VisionEncoderModel.register_for_auto_class("AutoModel")
Videollama3ImageProcessor.register_for_auto_class("AutoImageProcessor")

Videollama3Qwen2Config.register_for_auto_class()
Videollama3Qwen2Model.register_for_auto_class("AutoModel")
Videollama3Qwen2ForCausalLM.register_for_auto_class("AutoModelForCausalLM")
Videollama3Qwen2Processor.register_for_auto_class("AutoProcessor")
