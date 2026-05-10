from transformers import AutoConfig, AutoModel, AutoModelForCausalLM
from kf_model.config import KiteFishConfig
from kf_model.model import KiteFishModel, KiteFishForCausalLM



AutoConfig.register("kitefish_v1", KiteFishConfig)
AutoModel.register(KiteFishConfig, KiteFishModel)
AutoModelForCausalLM.register(KiteFishConfig, KiteFishForCausalLM)